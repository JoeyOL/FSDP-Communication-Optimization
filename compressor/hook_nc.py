import torch
import torch.distributed as dist

from .common import _apply_error_feedback, _ensure_residual


class NCState:
    def __init__(self, error_feedback: bool = False) -> None:
        self.error_feedback = error_feedback


def _nc_compress_scalar(t: torch.Tensor) -> torch.Tensor:
    """Natural compression: randomized rounding to nearest ±2^k."""
    out = torch.empty_like(t)
    zero = t == 0
    out[zero] = 0
    t_nz = t[~zero]
    abs_t = t_nz.abs()
    alpha = torch.log2(abs_t)
    lo = torch.pow(2.0, alpha.floor())
    hi = torch.pow(2.0, alpha.ceil())
    p = (abs_t - lo) / (hi - lo + 1e-12)
    u = torch.empty_like(t_nz, device=t.device).uniform_(0, 1)
    choice = (u < p).float() * hi + (u >= p).float() * lo
    out[~zero] = choice * t_nz.sign()
    return out


def fsdp_nc_comm_hook(
    state: NCState,
    full_flat_grad: torch.Tensor,
    shard_out: torch.Tensor,
) -> None:
    """Natural compression: unbiased round to ±2^k, then reduce-scatter."""
    pg = dist.group.WORLD
    world_size = dist.get_world_size(pg)

    if world_size == 1:
        shard_out.copy_(full_flat_grad)
        return

    g = full_flat_grad.contiguous().view(-1)
    if state.error_feedback:
        residual = _ensure_residual(state, g)
        g = (g + residual).to(g.dtype)

    q_g = _nc_compress_scalar(g)
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


__all__ = ["NCState", "fsdp_nc_comm_hook"]


