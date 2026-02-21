import torch
import torch.distributed as dist

from .common import _apply_error_feedback, _ensure_residual


class FP16State:
    def __init__(self, error_feedback: bool = False, ef_local: bool = False) -> None:
        self.error_feedback = error_feedback
        self._ef_local = ef_local


def fsdp_fp16_comm_hook(
    state: FP16State,
    full_flat_grad: torch.Tensor,
    shard_out: torch.Tensor,
) -> None:
    """Compress gradient to float16 for communication, then dequantize to full precision."""
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

    g_fp16 = g.half()
    shard_size = g.numel() // world_size
    temp_shard = torch.empty(shard_size, device=g.device, dtype=torch.float16)
    if hasattr(dist, "reduce_scatter_tensor"):
        dist.reduce_scatter_tensor(temp_shard, g_fp16, op=dist.ReduceOp.SUM, group=pg)
    else:
        chunks = list(g_fp16.chunk(world_size, dim=0))
        dist.reduce_scatter(temp_shard, chunks, op=dist.ReduceOp.SUM, group=pg)

    deq_avg = (temp_shard.float() / float(world_size)).to(full_flat_grad.dtype)
    shard_out.copy_(deq_avg)

    if state.error_feedback:
        _apply_error_feedback(state, full_flat_grad, shard_out, g, pg)


__all__ = ["FP16State", "fsdp_fp16_comm_hook"]


