from typing import Any

import torch
import torch.distributed as dist

from .common import _apply_error_feedback, _ensure_residual
from perf.comm_stats import add_bytes as _comm_add_bytes


class GradQuantState:
    def __init__(self, num_bits: int = 8, error_feedback: bool = False, variant: str = "linear", ef_local: bool = False) -> None:
        self.num_bits = num_bits
        self.error_feedback = error_feedback
        self._ef_local = ef_local
        # variant: "linear" (symmetric scale=global_max) or "dynamic_tree" (normalize to [0,1], 7-bit stochastic)
        self.variant = (variant or "linear").lower().strip()


def _int8_linear_quantize(g: torch.Tensor, global_max: torch.Tensor, world_size: int) -> torch.Tensor:
    q = 127
    qr = max(1, q // world_size)
    scale = qr / torch.clamp(global_max, min=1e-8)
    return torch.clamp((g * scale).round(), -qr, qr).to(torch.int8)


def _int8_dynamic_tree_quantize(g: torch.Tensor, scale: float) -> torch.Tensor:
    """Normalize to [0,1] by max, 7-bit stochastic rounding; store as uint8: sign*128 + level (paper 3.1)."""
    if scale < 1e-12:
        return torch.zeros_like(g, dtype=torch.uint8)
    g_norm = (g.abs() / scale).clamp(0.0, 1.0)
    level_float = g_norm * 127.0
    lo = level_float.floor().clamp(0, 126)
    hi = level_float.ceil().clamp(1, 127)
    u = torch.empty_like(g).uniform_(0, 1)
    lev = torch.where(u < (level_float - lo), hi, lo)
    lev = lev.to(torch.int64)
    sign_bit = (g >= 0).to(torch.int64)
    return (sign_bit * 128 + lev).clamp(0, 255).to(torch.uint8)


def _int8_dynamic_tree_dequantize(q_grad: torch.Tensor, scale: float) -> torch.Tensor:
    """Dequant uint8 (sign*128 + 7-bit level) back to float."""
    q = q_grad.to(torch.int64)
    sign = torch.where(q >= 128, 1.0, -1.0)
    lev = (q & 127).float()
    return sign * (lev / 127.0) * scale


def fsdp_quantized_comm_hook(
    state: GradQuantState,
    full_flat_grad: torch.Tensor,
    shard_out: torch.Tensor,
) -> None:
    """
    FSDP communication hook for int8 quantization before reduce-scatter.
    variant=linear: symmetric scale=global_max; variant=dynamic_tree: normalize [0,1] + 7-bit stochastic (paper 3.1).
    With error_feedback: compresses (g + residual) and updates residual.
    """
    pg = dist.group.WORLD
    world_size = dist.get_world_size(pg)

    if world_size == 1:
        shard_out.copy_(full_flat_grad)
        return

    g = full_flat_grad.contiguous().view(-1)
    if state.error_feedback:
        residual, start, end = _ensure_residual(state, g, pg)
        if start is None:
            g = (g + residual).to(g.dtype)
        else:
            g[start:end] += residual

    numel = g.numel()
    assert numel % world_size == 0, (
        f"flat grad numel {numel} must be divisible by world_size {world_size}"
    )

    local_max = g.abs().max().to(torch.float32)
    global_max = local_max.clone()
    dist.all_reduce(global_max, op=dist.ReduceOp.MAX, group=pg)
    scale_val = global_max.clamp(min=1e-8).item()

    if getattr(state, "variant", "linear") == "dynamic_tree":
        # 动态树 8bit：保持 7-bit 随机舍入的量化形式，但通信改为 int16 + reduce_scatter，
        # 使每个 rank 只处理自己 shard 的数据，避免 all_gather 的 O(world_size * numel) 开销。
        if scale_val < 1e-8:
            shard_out.zero_()
        else:
            # 生成 7-bit level 和符号，并编码为有符号整型：signed_level ∈ [-127, 127]
            g_abs = g.abs()
            g_norm = (g_abs / scale_val).clamp(0.0, 1.0)
            level_float = g_norm * 127.0
            lo = level_float.floor().clamp(0, 126)
            hi = level_float.ceil().clamp(1, 127)
            u = torch.empty_like(g, dtype=torch.float32).uniform_(0, 1)
            lev = torch.where(u < (level_float - lo), hi, lo).to(torch.int32)
            sign = torch.where(g >= 0, 1, -1).to(torch.int32)
            signed_level = (sign * lev).view(-1)

            shard_size = numel // world_size
            temp_shard_level = torch.empty(shard_size, device=g.device, dtype=torch.int32)
            if hasattr(dist, "reduce_scatter_tensor"):
                dist.reduce_scatter_tensor(
                    temp_shard_level, signed_level, op=dist.ReduceOp.SUM, group=pg
                )
            else:
                chunks = list(signed_level.chunk(world_size, dim=0))
                dist.reduce_scatter(
                    temp_shard_level, chunks, op=dist.ReduceOp.SUM, group=pg
                )

            # 近似统计：按 signed_level 元素数估算 reduce_scatter 负载
            try:
                _comm_add_bytes(state, signed_level.numel() * signed_level.element_size())
            except Exception:
                pass

            deq_sum = temp_shard_level.to(torch.float32) * (scale_val / 127.0)
            deq_avg = (deq_sum / float(world_size)).to(full_flat_grad.dtype)
            shard_out.copy_(deq_avg)
    else:
        qr = max(1, 127 // world_size)
        scale = qr / torch.clamp(global_max, min=1e-8)
        q_grad = _int8_linear_quantize(g, global_max, world_size)
        temp_shard_out = torch.empty_like(shard_out, dtype=torch.int8)
        if hasattr(dist, "reduce_scatter_tensor"):
            dist.reduce_scatter_tensor(temp_shard_out, q_grad, op=dist.ReduceOp.SUM, group=pg)
        else:
            chunks = list(q_grad.chunk(world_size, dim=0))
            dist.reduce_scatter(temp_shard_out, chunks, op=dist.ReduceOp.SUM, group=pg)

        # 近似统计：按 q_grad 元素数估算 reduce_scatter 负载
        try:
            _comm_add_bytes(state, q_grad.numel() * q_grad.element_size())
        except Exception:
            pass
        deq_sum = temp_shard_out.float() / scale
        deq_avg = (deq_sum / float(world_size)).to(full_flat_grad.dtype)
        shard_out.copy_(deq_avg)

    if state.error_feedback:
        _apply_error_feedback(state, full_flat_grad, shard_out, g, pg)


__all__ = ["GradQuantState", "fsdp_quantized_comm_hook"]


