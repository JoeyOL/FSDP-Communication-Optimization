from typing import Tuple

import torch
import torch.distributed as dist

from .common import _apply_error_feedback, _ensure_residual, _sparse_all_gather_and_merge


class ThresholdVState:
    def __init__(
        self,
        v: float = 0.0,
        v_neg: float | None = None,
        ratio: float = 0.01,
        error_feedback: bool = True,
        sparse_comm: bool = True,
    ) -> None:
        self.v = v
        # v_neg: negative threshold; if None or <0, use v for both (symmetric)
        self.v_neg = v_neg if v_neg is not None and v_neg >= 0 else v
        self.ratio = ratio
        self.error_feedback = error_feedback
        self.sparse_comm = sparse_comm

    @property
    def v_pos(self) -> float:
        return self.v

    @property
    def v_neg_val(self) -> float:
        return self.v_neg


def _thresholdv_to_fixed_k_indices_values(
    g: torch.Tensor,
    v_pos: float,
    v_neg: float,
    k: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """保留 g_i >= v_pos 或 g_i <= -v_neg 的坐标；若超过 k 个取绝对值最大的 k 个，不足则用 (0,0) 填充。"""
    mask = (g >= v_pos) | (g <= -v_neg)
    cand_indices = torch.nonzero(mask, as_tuple=False).squeeze(-1)
    if cand_indices.dim() == 0:
        cand_indices = cand_indices.unsqueeze(0)
    cand_values = g[cand_indices]
    n = cand_indices.numel()
    if n >= k:
        _, order = cand_values.abs().topk(k, largest=True, sorted=False)
        indices = cand_indices[order]
        values = cand_values[order]
    else:
        indices = torch.zeros(k, device=g.device, dtype=torch.long)
        values = torch.zeros(k, device=g.device, dtype=g.dtype)
        indices[:n] = cand_indices
        values[:n] = cand_values
    return indices, values


def fsdp_thresholdv_comm_hook(
    state: ThresholdVState,
    full_flat_grad: torch.Tensor,
    shard_out: torch.Tensor,
) -> None:
    """Threshold-v: 只传 |g_i| > v 的维度；sparse_comm 时只传 (indices, values)，固定 k 长。"""
    pg = dist.group.WORLD
    world_size = dist.get_world_size(pg)

    if world_size == 1:
        shard_out.copy_(full_flat_grad)
        return

    g = full_flat_grad.contiguous().view(-1)
    if state.error_feedback:
        residual = _ensure_residual(state, g)
        g = (g + residual).to(g.dtype)

    numel = g.numel()
    k = max(1, int(numel * state.ratio))

    if state.v > 0 or state.v_neg > 0:
        v_pos = state.v_pos
        v_neg = state.v_neg_val
    else:
        abs_g = g.abs()
        idx = max(1, numel - min(k, numel))
        v_pos = v_neg = torch.kthvalue(abs_g, idx).values.item()

    if state.sparse_comm:
        indices, values = _thresholdv_to_fixed_k_indices_values(g, v_pos, v_neg, k)
        full_sum = _sparse_all_gather_and_merge(g.new_tensor(indices, dtype=torch.long), values, numel, pg)
        shard_size = numel // world_size
        rank = dist.get_rank(pg)
        shard_start = rank * shard_size
        shard_end = shard_start + shard_size
        deq_avg = (full_sum[shard_start:shard_end] / float(world_size)).to(full_flat_grad.dtype)
        shard_out.copy_(deq_avg)
        if state.error_feedback:
            sparse_full = torch.where((g >= v_pos) | (g <= -v_neg), g, torch.zeros_like(g))
            _apply_error_feedback(state, full_flat_grad, shard_out, sparse_full, pg)
        return

    sparse = torch.where((g >= v_pos) | (g <= -v_neg), g, torch.zeros_like(g))
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


__all__ = ["ThresholdVState", "fsdp_thresholdv_comm_hook"]


