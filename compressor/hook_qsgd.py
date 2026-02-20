import torch
import torch.distributed as dist

from .common import _apply_error_feedback, _ensure_residual


class QSGDState:
    def __init__(self, s: int = 4, error_feedback: bool = False, bucket_size: int = 0) -> None:
        self.s = max(2, int(s))
        self.error_feedback = error_feedback
        # bucket_size <= 0 or >= numel: whole vector one bucket; else per-bucket QSGD (paper Section 4)
        self.bucket_size = bucket_size


def _qsgd_quantize(v: torch.Tensor, s: int) -> torch.Tensor:
    """QSGD stochastic quantization: scale by norm, round to s levels stochastically."""
    norm = v.norm(p=2)
    if norm < 1e-12:
        return v
    v_n = v / norm
    # v_n in [-1,1]; quantize to {-1, -(s-1)/s, ..., (s-1)/s, 1}
    abs_v = v_n.abs()
    level_float = abs_v * s
    lo = level_float.floor().clamp(0, s - 1)
    hi = level_float.ceil().clamp(0, s)
    p = level_float - lo
    # stochastic rounding
    u = torch.empty_like(v).uniform_(0, 1)
    lev = torch.where(u < p, hi, lo)
    q_abs = lev.float() / s
    q_n = q_abs * v_n.sign()
    return q_n * norm


def _qsgd_quantize_bucketed(g: torch.Tensor, s: int, bucket_size: int) -> torch.Tensor:
    """QSGD per bucket: each bucket of size bucket_size gets independent L2 norm and s-level quantization."""
    numel = g.numel()
    if bucket_size <= 0 or bucket_size >= numel:
        return _qsgd_quantize(g, s)
    out = []
    for start in range(0, numel, bucket_size):
        end = min(start + bucket_size, numel)
        out.append(_qsgd_quantize(g[start:end], s))
    return torch.cat(out)


def fsdp_qsgd_comm_hook(
    state: QSGDState,
    full_flat_grad: torch.Tensor,
    shard_out: torch.Tensor,
) -> None:
    """QSGD: stochastic quantization with s levels (per-bucket if bucket_size set), then reduce-scatter in float."""
    pg = dist.group.WORLD
    world_size = dist.get_world_size(pg)

    if world_size == 1:
        shard_out.copy_(full_flat_grad)
        return

    g = full_flat_grad.contiguous().view(-1)
    if state.error_feedback:
        residual = _ensure_residual(state, g)
        g = (g + residual).to(g.dtype)

    q_g = _qsgd_quantize_bucketed(g, state.s, state.bucket_size)
    shard_size = g.numel() // world_size
    temp_shard = torch.empty(shard_size, device=g.device, dtype=g.dtype)
    if hasattr(dist, "reduce_scatter_tensor"):
        dist.reduce_scatter_tensor(temp_shard, q_g, op=dist.ReduceOp.SUM, group=pg)
    else:
        chunks = list(q_g.chunk(world_size, dim=0))
        dist.reduce_scatter(temp_shard, chunks, op=dist.ReduceOp.SUM, group=pg)

    deq_avg = (temp_shard / float(world_size)).to(full_flat_grad.dtype)
    shard_out.copy_(deq_avg)

    if state.error_feedback:
        _apply_error_feedback(state, full_flat_grad, shard_out, g, pg)


__all__ = ["QSGDState", "fsdp_qsgd_comm_hook"]


