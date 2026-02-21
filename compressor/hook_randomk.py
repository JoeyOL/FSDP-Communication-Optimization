import torch
import torch.distributed as dist

from .common import _apply_error_feedback, _ensure_residual, _sparse_all_gather_and_merge


class RandomKState:
    def __init__(
        self,
        k: int = 0,
        ratio: float = 0.01,
        error_feedback: bool = True,
        sparse_comm: bool = True,
        ef_local: bool = False,
    ) -> None:
        self.k = k
        self.ratio = ratio
        self.error_feedback = error_feedback
        self._ef_local = ef_local
        self.sparse_comm = sparse_comm


def fsdp_randomk_comm_hook(
    state: RandomKState,
    full_flat_grad: torch.Tensor,
    shard_out: torch.Tensor,
) -> None:
    """Random-k: 随机保留 k 个坐标并乘 d/k 无偏；sparse_comm 时只传 (indices, values)。"""
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
    scale = numel / k
    perm = torch.randperm(numel, device=g.device)
    indices = perm[:k]
    values = (g[indices] * scale).to(g.dtype)

    if state.sparse_comm:
        full_sum = _sparse_all_gather_and_merge(indices, values, numel, pg)
        shard_size = numel // world_size
        rank = dist.get_rank(pg)
        shard_start = rank * shard_size
        shard_end = shard_start + shard_size
        deq_avg = (full_sum[shard_start:shard_end] / float(world_size)).to(full_flat_grad.dtype)
        shard_out.copy_(deq_avg)
        if state.error_feedback:
            sparse_full = torch.zeros_like(g)
            sparse_full[indices] = values
            _apply_error_feedback(state, full_flat_grad, shard_out, sparse_full, pg)
        return

    sparse = torch.zeros_like(g)
    sparse[indices] = values
    shard_size = numel // world_size
    temp_shard = torch.empty(shard_size, device=g.device, dtype=g.dtype)
    if hasattr(dist, "reduce_scatter_tensor"):
        dist.reduce_scatter_tensor(temp_shard, sparse, op=dist.ReduceOp.SUM, group=pg)
    else:
        chunks = list(sparse.chunk(world_size, dim=0))
        dist.reduce_scatter(temp_shard, chunks, op=dist.ReduceOp.SUM, group=pg)

    deq_avg = (temp_shard / float(world_size)).to(full_flat_grad.dtype)
    shard_out.copy_(deq_avg)

    if state.error_feedback:
        _apply_error_feedback(state, full_flat_grad, shard_out, sparse, pg)


__all__ = ["RandomKState", "fsdp_randomk_comm_hook"]


