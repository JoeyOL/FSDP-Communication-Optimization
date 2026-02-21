import torch
import torch.distributed as dist

from .common import _apply_error_feedback, _ensure_residual


class HybridTopKInt8State:
    def __init__(self, k: int = 0, ratio: float = 0.01, error_feedback: bool = True, ef_local: bool = False) -> None:
        self.k = k
        self.ratio = ratio
        self.error_feedback = error_feedback
        self._ef_local = ef_local


def fsdp_hybrid_topk_int8_comm_hook(
    state: HybridTopKInt8State,
    full_flat_grad: torch.Tensor,
    shard_out: torch.Tensor,
) -> None:
    """Hybrid: apply top-k sparsification then int8 quantize the sparse vector; reduce-scatter in int8."""
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
    k = state.k if state.k > 0 else max(1, int(numel * state.ratio))
    k = min(k, numel)
    _, indices = g.abs().topk(k, largest=True, sorted=False)
    sparse = torch.zeros_like(g)
    sparse[indices] = g[indices]

    local_max = sparse.abs().max().to(torch.float32)
    global_max = local_max.clone()
    dist.all_reduce(global_max, op=dist.ReduceOp.MAX, group=pg)
    qr = max(1, 127 // world_size)
    scale = qr / torch.clamp(global_max, min=1e-8)
    q_grad = torch.clamp((sparse * scale).round(), -qr, qr).to(torch.int8)

    temp_shard_out = torch.empty_like(shard_out, dtype=torch.int8)
    if hasattr(dist, "reduce_scatter_tensor"):
        dist.reduce_scatter_tensor(temp_shard_out, q_grad, op=dist.ReduceOp.SUM, group=pg)
    else:
        chunks = list(q_grad.chunk(world_size, dim=0))
        dist.reduce_scatter(temp_shard_out, chunks, op=dist.ReduceOp.SUM, group=pg)

    deq_sum = temp_shard_out.float() / scale
    deq_avg = (deq_sum / float(world_size)).to(full_flat_grad.dtype)
    shard_out.copy_(deq_avg)

    if state.error_feedback:
        _apply_error_feedback(state, full_flat_grad, shard_out, sparse, pg)


__all__ = ["HybridTopKInt8State", "fsdp_hybrid_topk_int8_comm_hook"]


