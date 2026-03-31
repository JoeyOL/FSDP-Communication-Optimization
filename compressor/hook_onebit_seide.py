from typing import Tuple

import torch
import torch.distributed as dist

from .common import _ensure_residual, logger
from perf.comm_stats import add_bytes as _comm_add_bytes

try:
    # C++ 实现的 1-bit Seide 辅助算子（如可用则优先使用）
    from .onebit_seide_ops import (
        onebit_seide_per_column as cpp_onebit_seide_per_column,
        onebit_seide_reconstruct_from_gathered as cpp_onebit_seide_reconstruct,
    )

    _HAS_ONEBIT_CPP = True
except Exception:
    _HAS_ONEBIT_CPP = False

# 运行时探测：如果加载/调用 C++ 扩展失败，则退回 Python 实现，且不再重试。
_ONEBIT_CPP_RUNTIME_AVAILABLE = True


def _pack_signs_to_bytes(signs: torch.Tensor) -> torch.Tensor:
    """Pack signs (1 = non-negative, 0 = negative) into uint8, 8 per byte, LSB first."""
    flat = (signs >= 0).to(torch.uint8).view(-1)
    n = flat.numel()
    n_pad = (8 - n % 8) % 8
    if n_pad:
        flat = torch.cat([flat, torch.zeros(n_pad, device=flat.device, dtype=torch.uint8)])
    powers = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128], device=flat.device, dtype=torch.uint8)
    return (flat.view(-1, 8) * powers).sum(dim=1).to(torch.uint8)


def _unpack_signs_from_bytes_all(packed_all: torch.Tensor, numel: int) -> torch.Tensor:
    """Vectorized unpack for all ranks at once.

    packed_all: (world_size, packed_bytes) uint8
    return: (world_size, numel) float32 in {+1, -1}
    """
    world_size, n_bytes = packed_all.shape
    n_bits = min(numel, n_bytes * 8)
    if n_bits == 0:
        return torch.zeros(world_size, numel, device=packed_all.device, dtype=torch.float32)

    packed_int = packed_all.to(torch.int32)  # (world_size, n_bytes)
    bit_pos = torch.arange(8, device=packed_all.device, dtype=torch.int32)  # (8,)
    # (world_size, n_bytes, 8)
    bits_expanded = (packed_int.unsqueeze(-1) >> bit_pos.unsqueeze(0).unsqueeze(0)) & 1
    bits_flat = bits_expanded.view(world_size, -1)[:, :n_bits]  # (world_size, n_bits)

    out = torch.zeros(world_size, numel, device=packed_all.device, dtype=torch.float32)
    out[:, :n_bits] = torch.where(bits_flat == 1, 1.0, -1.0)
    return out


def _onebit_seide_per_column(g: torch.Tensor, col_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """For each column: signs (1/-1), a=mean(positive), b=mean(negative)."""
    global _ONEBIT_CPP_RUNTIME_AVAILABLE
    # 如果有 C++ 实现且运行时可用，优先使用
    if _HAS_ONEBIT_CPP and _ONEBIT_CPP_RUNTIME_AVAILABLE:
        try:
            return cpp_onebit_seide_per_column(g, col_size)
        except Exception as e:
            logger.warning(
                f"[1bit_seide] C++ onebit_seide_per_column unavailable at runtime, "
                f"falling back to Python implementation: {e}"
            )
            _ONEBIT_CPP_RUNTIME_AVAILABLE = False
    numel = g.numel()
    num_cols = (numel + col_size - 1) // col_size
    device = g.device
    dtype = g.dtype
    signs = g.sign()
    signs_pos = (signs >= 0).to(dtype)
    signs_neg = 1.0 - signs_pos

    # 向量化计算每列的 a 和 b
    n_pad = (col_size - numel % col_size) % col_size
    if n_pad > 0:
        g_padded = torch.cat([g, torch.zeros(n_pad, device=device, dtype=dtype)])
        signs_pos_padded = torch.cat([signs_pos, torch.zeros(n_pad, device=device, dtype=dtype)])
        signs_neg_padded = torch.cat([signs_neg, torch.zeros(n_pad, device=device, dtype=dtype)])
    else:
        g_padded = g
        signs_pos_padded = signs_pos
        signs_neg_padded = signs_neg

    g_cols = g_padded.view(num_cols, col_size)
    pos_mask = signs_pos_padded.view(num_cols, col_size)
    neg_mask = signs_neg_padded.view(num_cols, col_size)

    pos_sum = (g_cols * pos_mask).sum(dim=1)
    pos_count = pos_mask.sum(dim=1)
    neg_sum = (g_cols * neg_mask).sum(dim=1)
    neg_count = neg_mask.sum(dim=1)

    a_vec = torch.where(pos_count > 0, pos_sum / pos_count.clamp(min=1e-8), torch.zeros_like(pos_sum))
    b_vec = torch.where(neg_count > 0, neg_sum / neg_count.clamp(min=1e-8), torch.zeros_like(neg_sum))

    packed = _pack_signs_to_bytes(signs)
    return packed, a_vec, b_vec


def _onebit_seide_reconstruct_from_gathered(
    signs_all: torch.Tensor,
    a_all: torch.Tensor,
    b_all: torch.Tensor,
    numel: int,
    col_size: int,
    world_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """full_sum[i] = sum_r (a_r[c] if sign_r[i] >= 0 else b_r[c])."""
    global _ONEBIT_CPP_RUNTIME_AVAILABLE
    if _HAS_ONEBIT_CPP and _ONEBIT_CPP_RUNTIME_AVAILABLE:
        try:
            return cpp_onebit_seide_reconstruct(signs_all, a_all, b_all, numel, col_size)
        except Exception as e:
            logger.warning(
                f"[1bit_seide] C++ onebit_seide_reconstruct unavailable at runtime, "
                f"falling back to Python implementation: {e}"
            )
            _ONEBIT_CPP_RUNTIME_AVAILABLE = False

    num_cols = a_all.shape[1]
    col_idx = torch.arange(numel, device=device, dtype=torch.long) // col_size
    col_idx = col_idx.clamp(max=num_cols - 1)

    signs_unpacked = _unpack_signs_from_bytes_all(signs_all, numel)

    col_idx_expanded = col_idx.unsqueeze(0).expand(world_size, -1)
    col_idx_expanded = col_idx_expanded.clamp(min=0, max=num_cols - 1)

    a_expanded = torch.gather(a_all, dim=1, index=col_idx_expanded)
    b_expanded = torch.gather(b_all, dim=1, index=col_idx_expanded)

    recon_all = torch.where(signs_unpacked >= 0, a_expanded, b_expanded)
    return recon_all.sum(dim=0)


class OneBitSeideState:
    def __init__(self, col_size: int = 256, error_feedback: bool = True, use_cpp: bool = True, ef_local: bool = False) -> None:
        self.col_size = col_size
        self.error_feedback = error_feedback
        self._ef_local = ef_local
        self.use_cpp = use_cpp


def fsdp_onebit_seide_comm_hook(
    state: OneBitSeideState,
    full_flat_grad: torch.Tensor,
    shard_out: torch.Tensor,
) -> None:
    """1-bit Seide: per-column (a,b) minimize squared error, transmit signs + (a,b), EF.

    优化：移除 3 个冗余的 all_reduce（packed_bytes, ab_per_rank, num_cols），
    因为 FSDP 保证各 rank 的梯度形状一致，这些维度天然相同。
    """
    global _ONEBIT_CPP_RUNTIME_AVAILABLE

    pg = dist.group.WORLD
    world_size = dist.get_world_size(pg)
    rank = dist.get_rank(pg)

    if world_size == 1:
        shard_out.copy_(full_flat_grad)
        return

    if not getattr(state, "use_cpp", True):
        _ONEBIT_CPP_RUNTIME_AVAILABLE = False

    if not hasattr(state, "_logged_once"):
        logger.info(
            f"[1bit_seide R{rank}] Hook initialized: col_size={state.col_size}, error_feedback={state.error_feedback}"
        )
        state._logged_once = True

    g = full_flat_grad.contiguous().view(-1)

    if state.error_feedback:
        residual, start, end = _ensure_residual(state, g, pg)
        if start is None:
            g = (g + residual).to(g.dtype)
        else:
            g[start:end] += residual

    numel = g.numel()
    col_size = min(state.col_size, numel)
    if col_size < 1:
        col_size = 1
    num_cols = (numel + col_size - 1) // col_size

    packed, a_vec, b_vec = _onebit_seide_per_column(g, col_size)
    packed_bytes = packed.numel()
    ab_per_rank = num_cols * 2

    # 优化：移除了 3 个 all_reduce（packed_bytes, ab_per_rank, num_cols）。
    # FSDP 保证各 rank 的 full_flat_grad 形状一致 → numel 一致 → col_size, num_cols,
    # packed_bytes, ab_per_rank 天然一致，无需跨 rank 同步。

    ab_vec = torch.cat([a_vec, b_vec])

    # 记录本次 all-gather 的通信字节数
    per_rank_bytes = int(packed_bytes) + int(ab_per_rank) * g.element_size()
    _comm_add_bytes(state, per_rank_bytes)

    # All-gather（这是主要的通信开销）
    if hasattr(dist, "all_gather_into_tensor"):
        signs_buf = torch.empty(world_size * packed_bytes, device=g.device, dtype=torch.uint8)
        ab_buf = torch.empty(world_size * ab_per_rank, device=g.device, dtype=g.dtype)
        dist.all_gather_into_tensor(signs_buf, packed.contiguous(), group=pg)
        dist.all_gather_into_tensor(ab_buf, ab_vec.contiguous(), group=pg)
        signs_all = signs_buf.view(world_size, packed_bytes)
        ab_all = ab_buf.view(world_size, ab_per_rank)
        a_all = ab_all[:, :num_cols]
        b_all = ab_all[:, num_cols : num_cols * 2] if num_cols * 2 <= ab_per_rank else ab_all[:, num_cols:]
    else:
        signs_list = [torch.empty(packed_bytes, device=g.device, dtype=torch.uint8) for _ in range(world_size)]
        ab_list = [torch.empty(ab_per_rank, device=g.device, dtype=g.dtype) for _ in range(world_size)]
        dist.all_gather(signs_list, packed.contiguous(), group=pg)
        dist.all_gather(ab_list, ab_vec.contiguous(), group=pg)
        signs_all = torch.stack(signs_list, dim=0)
        ab_all = torch.stack(ab_list, dim=0)
        a_all = ab_all[:, :num_cols]
        b_all = ab_all[:, num_cols : num_cols * 2] if num_cols * 2 <= ab_per_rank else ab_all[:, num_cols:]

    # 重建（已优化为批量操作）
    full_sum = _onebit_seide_reconstruct_from_gathered(
        signs_all, a_all, b_all, numel, col_size, world_size, g.device, g.dtype
    )

    if full_sum.shape[0] != numel:
        raise RuntimeError(f"full_sum shape mismatch: {full_sum.shape} vs {numel}")

    shard_size = numel // world_size
    shard_start = rank * shard_size
    shard_end = min(shard_start + shard_size, numel)

    deq_avg = (full_sum[shard_start:shard_end] / float(world_size)).to(full_flat_grad.dtype)
    shard_out.copy_(deq_avg)

    if state.error_feedback:
        full_reconstructed = full_sum / float(world_size)
        idx = getattr(state, "_ef_index", 1) - 1
        res_list = getattr(state, "_ef_residual_list", None)
        if idx >= 0 and res_list is not None and idx < len(res_list) and res_list[idx] is not None:
            res = res_list[idx]
            if getattr(state, "_ef_local", False):
                ef_start = getattr(state, "_ef_shard_start", 0)
                ef_end = getattr(state, "_ef_shard_end", 0)
                if res.numel() == (ef_end - ef_start) and ef_end <= g.numel():
                    res.copy_(g[ef_start:ef_end] - full_reconstructed[ef_start:ef_end])
            elif res.numel() == g.numel():
                res.copy_(g - full_reconstructed)


__all__ = ["OneBitSeideState", "fsdp_onebit_seide_comm_hook"]
