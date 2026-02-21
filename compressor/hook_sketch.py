import torch
import torch.distributed as dist

from .common import _apply_error_feedback, _ensure_residual


def _count_sketch_encode(g: torch.Tensor, width: int, depth: int) -> torch.Tensor:
    d = g.numel()
    sketch = torch.zeros(depth, width, device=g.device, dtype=g.dtype)
    for dep in range(depth):
        h = (torch.arange(d, device=g.device) * (dep + 1) + dep) % width
        s = 2 * (torch.arange(d, device=g.device) % 2) - 1
        sketch.index_add_(1, h, g * s)
    return sketch


def _count_sketch_decode_all(sketch: torch.Tensor, d: int, width: int, depth: int) -> torch.Tensor:
    est = torch.zeros(d, device=sketch.device, dtype=sketch.dtype)
    for dep in range(depth):
        h = (torch.arange(d, device=sketch.device) * (dep + 1) + dep) % width
        s = 2 * (torch.arange(d, device=sketch.device) % 2) - 1
        est += sketch[dep, h] * s
    est /= depth
    return est


def _heavymix_topk_indices(sketch: torch.Tensor, d: int, width: int, depth: int, k: int) -> torch.Tensor:
    """HEAVYMIX (Algorithm 1): from merged sketch get Topk indices. H = {i : hat_g_i^2 >= hat_l2^2/k}, Topk = H cup rand_l(NH)."""
    est = _count_sketch_decode_all(sketch, d, width, depth)
    est_sq = est * est
    hat_l2_sq = est_sq.sum().clamp(min=1e-12)
    thresh = hat_l2_sq / max(1, k)
    H_mask = est_sq >= thresh
    H_indices = torch.nonzero(H_mask, as_tuple=False).squeeze(-1)
    if H_indices.dim() == 0:
        H_indices = H_indices.unsqueeze(0)
    n_H = H_indices.numel()
    NH_mask = ~H_mask
    NH_indices = torch.nonzero(NH_mask, as_tuple=False).squeeze(-1)
    if NH_indices.dim() == 0:
        NH_indices = NH_indices.unsqueeze(0)
    l = min(k - n_H, NH_indices.numel())
    if l <= 0:
        return H_indices[:k]
    perm = torch.randperm(NH_indices.numel(), device=NH_indices.device)[:l]
    rand_NH = NH_indices[perm]
    topk = torch.cat([H_indices, rand_NH], dim=0)
    if topk.numel() > k:
        topk = topk[:k]
    return topk


class SketchState:
    def __init__(self, k: int = 0, ratio: float = 0.01, error_feedback: bool = True, two_round: bool = False, ef_local: bool = False) -> None:
        self.k = k
        self.ratio = ratio
        self.error_feedback = error_feedback
        self._ef_local = ef_local
        self.two_round = two_round
        self._sketch_depth = 3
        self._sketch_width = 0

    def _sketch_size(self, d: int) -> int:
        k = self.k if self.k > 0 else max(1, int(d * self.ratio))
        return max(k * 8, 256)

    def _count_sketch(self, g: torch.Tensor, width: int, depth: int) -> torch.Tensor:
        return _count_sketch_encode(g, width, depth)

    def _recover_heavy(self, g: torch.Tensor, sketch: torch.Tensor, width: int, depth: int, k: int) -> torch.Tensor:
        d = g.numel()
        est = _count_sketch_decode_all(sketch, d, width, depth)
        _, indices = est.abs().topk(k, largest=True, sorted=False)
        out = torch.zeros_like(g)
        out[indices] = g[indices]
        return out


def fsdp_sketch_comm_hook(
    state: SketchState,
    full_flat_grad: torch.Tensor,
    shard_out: torch.Tensor,
) -> None:
    """Sketched-SGD: two_round=True => round1 send sketch, merge, HEAVYMIX top-k indices; round2 send exact values at Topk."""
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

    d = g.numel()
    k = state.k if state.k > 0 else max(1, int(d * state.ratio))
    k = min(k, d)
    width = state._sketch_size(d)
    depth = state._sketch_depth

    if getattr(state, "two_round", False):
        sketch = state._count_sketch(g, width, depth)
        sketch_flat = sketch.view(-1)
        if hasattr(dist, "all_reduce"):
            dist.all_reduce(sketch_flat, op=dist.ReduceOp.SUM, group=pg)
        merged_sketch = sketch_flat.view(depth, width) / float(world_size)
        topk_indices = _heavymix_topk_indices(merged_sketch, d, width, depth, k)
        values_at_topk = g[topk_indices]
        if hasattr(dist, "all_gather_into_tensor"):
            all_vals = torch.empty(world_size * k, device=g.device, dtype=g.dtype)
            dist.all_gather_into_tensor(all_vals, values_at_topk.contiguous(), group=pg)
        else:
            all_vals = torch.cat(dist.all_gather(values_at_topk.contiguous(), group=pg), dim=0)
        full_sum = torch.zeros(d, device=g.device, dtype=g.dtype)
        full_sum.scatter_add_(0, topk_indices, all_vals.view(world_size, k).sum(dim=0))
        shard_size = d // world_size
        rank = dist.get_rank(pg)
        shard_start = rank * shard_size
        shard_end = shard_start + shard_size
        deq_avg = (full_sum[shard_start:shard_end] / float(world_size)).to(full_flat_grad.dtype)
        shard_out.copy_(deq_avg)
        if state.error_feedback:
            full_reconstructed = torch.zeros_like(g)
            full_reconstructed[topk_indices] = full_sum[topk_indices] / float(world_size)
            residual, start, end = _ensure_residual(state, full_flat_grad, pg)
            diff = g - full_reconstructed
            if start is None:
                residual.copy_(diff)
            else:
                residual.copy_(diff[start:end])
        return

    sketch = state._count_sketch(g, width, depth)
    sparse = state._recover_heavy(g, sketch, width, depth, k)
    shard_size = d // world_size
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


__all__ = ["SketchState", "fsdp_sketch_comm_hook"]


