import torch
import torch.distributed as dist

from .common import _apply_error_feedback, _ensure_residual


class SignSGDState:
    def __init__(self, error_feedback: bool = True, use_delta_scale: bool = False) -> None:
        self.error_feedback = error_feedback
        # use_delta_scale: True = paper "x - δ·sign(g)": output direction only (scale=1), δ from optimizer lr
        self.use_delta_scale = use_delta_scale


def fsdp_signsgd_comm_hook(
    state: SignSGDState,
    full_flat_grad: torch.Tensor,
    shard_out: torch.Tensor,
) -> None:
    """1-bit / SignSGD: communicate sign(g), majority vote; scale by L1 or by δ (lr) when use_delta_scale."""
    pg = dist.group.WORLD
    world_size = dist.get_world_size(pg)

    if world_size == 1:
        shard_out.copy_(full_flat_grad)
        return

    g = full_flat_grad.contiguous().view(-1)
    if state.error_feedback:
        residual = _ensure_residual(state, g)
        g = (g + residual).to(g.dtype)

    sign_g = g.sign()
    shard_size = g.numel() // world_size
    temp_shard = torch.empty(shard_size, device=g.device, dtype=g.dtype)
    if hasattr(dist, "reduce_scatter_tensor"):
        dist.reduce_scatter_tensor(temp_shard, sign_g, op=dist.ReduceOp.SUM, group=pg)
    else:
        chunks = list(sign_g.chunk(world_size, dim=0))
        dist.reduce_scatter(temp_shard, chunks, op=dist.ReduceOp.SUM, group=pg)

    sign_sum = temp_shard
    if getattr(state, "use_delta_scale", False):
        scale = 1.0
    else:
        norm_g = g.norm(p=1) / g.numel()
        scale = norm_g * world_size
    deq_avg = (sign_sum.sign() * scale / float(world_size)).to(full_flat_grad.dtype)
    shard_out.copy_(deq_avg)

    if state.error_feedback:
        _apply_error_feedback(state, full_flat_grad, shard_out, g, pg)


__all__ = ["SignSGDState", "fsdp_signsgd_comm_hook"]


