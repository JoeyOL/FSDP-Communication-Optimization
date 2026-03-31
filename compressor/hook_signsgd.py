import torch
import torch.distributed as dist

from .common import _apply_error_feedback, _ensure_residual
from perf.comm_stats import add_bytes as _comm_add_bytes


class SignSGDState:
    def __init__(self, error_feedback: bool = True, use_delta_scale: bool = False, ef_local: bool = False) -> None:
        self.error_feedback = error_feedback
        self._ef_local = ef_local
        # use_delta_scale: True = paper "x - δ·sign(g)": output direction only (scale=1), δ from optimizer lr
        self.use_delta_scale = use_delta_scale


def fsdp_signsgd_comm_hook(
    state: SignSGDState,
    full_flat_grad: torch.Tensor,
    shard_out: torch.Tensor,
) -> None:
    """1-bit / SignSGD: communicate sign(g) as int8 (4× compression vs float32),
    majority vote; scale by L1 or by δ (lr) when use_delta_scale.

    优化：sign 值域为 {-1, 0, 1}，使用 int8 传输代替 float32，
    通信量减少 4 倍。reduce_scatter(SUM) 后 sign_sum 取 sign 即为多数投票结果。
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

    # 计算缩放因子（在量化前使用完整梯度）
    if getattr(state, "use_delta_scale", False):
        scale = 1.0
    else:
        norm_g = g.norm(p=1) / g.numel()
        scale = norm_g * world_size

    # 用 int8 传输 sign：{-1, 0, 1} -> int8，通信量从 4 bytes/elem 降为 1 byte/elem
    sign_g_int8 = g.sign().to(torch.int8)
    shard_size = g.numel() // world_size
    # reduce_scatter 在 int8 上做 SUM：各 rank 的 sign 求和，结果在 [-world_size, world_size]，int8 足够
    temp_shard_int8 = torch.empty(shard_size, device=g.device, dtype=torch.int8)
    if hasattr(dist, "reduce_scatter_tensor"):
        dist.reduce_scatter_tensor(temp_shard_int8, sign_g_int8, op=dist.ReduceOp.SUM, group=pg)
    else:
        chunks = list(sign_g_int8.chunk(world_size, dim=0))
        dist.reduce_scatter(temp_shard_int8, chunks, op=dist.ReduceOp.SUM, group=pg)
    # 通信统计：按 int8 元素数计算（原来是 float32）
    try:
        _comm_add_bytes(state, sign_g_int8.numel() * sign_g_int8.element_size())
    except Exception:
        pass

    # 多数投票：对求和结果取 sign，然后乘以缩放因子
    sign_sum = temp_shard_int8.float()
    deq_avg = (sign_sum.sign() * scale / float(world_size)).to(full_flat_grad.dtype)
    shard_out.copy_(deq_avg)

    if state.error_feedback:
        _apply_error_feedback(state, full_flat_grad, shard_out, g, pg)


__all__ = ["SignSGDState", "fsdp_signsgd_comm_hook"]
