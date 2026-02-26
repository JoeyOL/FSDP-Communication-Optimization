from typing import Tuple

import torch
import torch.distributed as dist

from .common import _apply_error_feedback, _ensure_residual, _sparse_all_gather_and_merge, logger
from perf.comm_stats import add_bytes as _comm_add_bytes


# 仅在前几次调用、且 rank0 上打印关键步骤，帮助定位耗时环节
_THRESHOLDV_LOG_CAP = 50
_thresholdv_log_count = 0


class ThresholdVState:
    def __init__(
        self,
        v: float = 0.0,
        v_neg: float | None = None,
        ratio: float = 0.01,
        error_feedback: bool = True,
        sparse_comm: bool = True,
        ef_local: bool = False,
    ) -> None:
        self.v = v
        # v_neg: negative threshold; if None or <0, use v for both (symmetric)
        self.v_neg = v_neg if v_neg is not None and v_neg >= 0 else v
        self.ratio = ratio
        self.error_feedback = error_feedback
        self._ef_local = ef_local
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
    rank = dist.get_rank(pg)

    global _thresholdv_log_count
    _thresholdv_log_count += 1
    do_log = rank == 0 and _thresholdv_log_count <= _THRESHOLDV_LOG_CAP

    if world_size == 1:
        shard_out.copy_(full_flat_grad)
        return

    g = full_flat_grad.contiguous().view(-1)
    numel = g.numel()
    if do_log:
        logger.info(
            "[threshold_v] hook_start numel=%s ratio=%.4f sparse_comm=%s ef=%s (call#%s)",
            numel,
            state.ratio,
            state.sparse_comm,
            state.error_feedback,
            _thresholdv_log_count,
        )
        if g.device.type == "cuda":
            ev_start = torch.cuda.Event(enable_timing=True)
            ev_after_ef = torch.cuda.Event(enable_timing=True)
            ev_after_thresh = torch.cuda.Event(enable_timing=True)
            ev_after_select = torch.cuda.Event(enable_timing=True)
            ev_after_allgather = torch.cuda.Event(enable_timing=True)
            ev_after_copy = torch.cuda.Event(enable_timing=True)
            ev_end = torch.cuda.Event(enable_timing=True)
            ev_start.record()
    if state.error_feedback:
        residual, start, end = _ensure_residual(state, g, pg)
        if start is None:
            if do_log:
                logger.info("[threshold_v] apply_error_feedback full residual")
            g = (g + residual).to(g.dtype)
        else:
            if do_log:
                logger.info("[threshold_v] apply_error_feedback local residual slice [%s:%s]", start, end)
            g[start:end] += residual

    if do_log and g.device.type == "cuda":
        ev_after_ef.record()

    k_total = max(1, int(numel * state.ratio))

    if state.v > 0 or state.v_neg > 0:
        # 用户显式给定阈值，严格按 v/v_neg 执行
        v_pos = state.v_pos
        v_neg = state.v_neg_val
    else:
        # 自动双阈值：以目标稀疏率 ratio 为总预算，在正、负两侧分别估计分位数阈值 v_pos / v_neg。
        # 为降低开销，大向量上仍采用抽样近似。
        if do_log:
            logger.info("[threshold_v] auto dual-threshold start (k_total=%s, ratio=%.4f)", k_total, state.ratio)

        g_pos = g[g > 0]
        g_neg = (-g[g < 0])  # 负侧用绝对值
        n_pos = int(g_pos.numel())
        n_neg = int(g_neg.numel())

        if n_pos + n_neg == 0:
            # 退化情况：梯度全为 0
            v_pos = v_neg = 0.0
        else:
            # 按正负两侧占比分配总的“保留预算”
            k_pos = int(round(k_total * (n_pos / (n_pos + n_neg)))) if n_pos > 0 else 0
            k_neg = k_total - k_pos
            k_pos = max(0, min(n_pos, k_pos))
            k_neg = max(0, min(n_neg, k_neg))

            sample_cap = 200_000

            def _estimate_quantile_side(abs_side: torch.Tensor, n_side: int, k_side: int, label: str) -> float:
                if n_side == 0 or k_side <= 0:
                    return 0.0
                if n_side <= sample_cap:
                    idx = max(1, min(n_side, n_side - k_side + 1))
                    v_t = torch.kthvalue(abs_side, idx).values
                    if do_log:
                        logger.info(
                            "[threshold_v] auto %s-threshold exact: n=%s idx=%s v=%.6e",
                            label,
                            n_side,
                            idx,
                            float(v_t.item()),
                        )
                    return float(v_t.item())
                # 抽样近似
                stride = max(1, n_side // sample_cap)
                sample = abs_side[::stride]
                m = int(sample.numel())
                if m == 0:
                    return 0.0
                # 在样本上使用相同比例的分位数
                # 目标在该侧的比例约为 k_side / n_side
                k_sample = max(1, min(m, m - int(m * (k_side / max(1, n_side))) + 1))
                v_t = torch.kthvalue(sample, k_sample).values
                if do_log:
                    logger.info(
                        "[threshold_v] auto %s-threshold approx: n=%s m=%s k_sample=%s v=%.6e",
                        label,
                        n_side,
                        m,
                        k_sample,
                        float(v_t.item()),
                    )
                return float(v_t.item())

            v_pos = _estimate_quantile_side(g_pos.abs(), n_pos, k_pos, "pos")
            v_neg = _estimate_quantile_side(g_neg.abs(), n_neg, k_neg, "neg")

        if do_log:
            logger.info("[threshold_v] auto dual-threshold done v_pos=%.6e v_neg=%.6e", v_pos, v_neg)

    if do_log and g.device.type == "cuda":
        ev_after_thresh.record()

    if state.sparse_comm:
        if do_log:
            logger.info("[threshold_v] sparse path: select indices/values")
        # 稀疏通信路径下，按照目标总预算 k_total 选择不超过 k_total 个坐标。
        indices, values = _thresholdv_to_fixed_k_indices_values(g, v_pos, v_neg, k_total)
        if do_log:
            nnz = int((values != 0).sum().item())
            logger.info("[threshold_v] sparse path: selected nnz=%s (k_total=%s)", nnz, k_total)
            logger.info("[threshold_v] sparse path: all_gather_and_merge start")

        if do_log and g.device.type == "cuda":
            ev_after_select.record()

        # 打开 common._sparse_all_gather_and_merge 的内部计时（仅本次调用）
        if do_log:
            setattr(state, "_debug_sparse_allgather_timing", True)
        try:
            full_sum = _sparse_all_gather_and_merge(state, g.new_tensor(indices, dtype=torch.long), values, numel, pg)
        finally:
            if do_log:
                setattr(state, "_debug_sparse_allgather_timing", False)
        if do_log:
            logger.info("[threshold_v] sparse path: all_gather_and_merge done")

        if do_log and g.device.type == "cuda":
            ev_after_allgather.record()
        shard_size = numel // world_size
        shard_start = rank * shard_size
        shard_end = shard_start + shard_size
        deq_avg = (full_sum[shard_start:shard_end] / float(world_size)).to(full_flat_grad.dtype)
        shard_out.copy_(deq_avg)

        if do_log and g.device.type == "cuda":
            ev_after_copy.record()
        if state.error_feedback:
            if getattr(state, "_ef_local", False):
                # 本地 EF：直接在 shard 维度上更新 residual，避免构造 dense full 向量和额外 all_gather。
                idx = getattr(state, "_ef_index", 1) - 1
                res_list = getattr(state, "_ef_residual_list", None)
                if (
                    idx >= 0
                    and res_list is not None
                    and idx < len(res_list)
                    and res_list[idx] is not None
                    and res_list[idx].numel() == shard_size
                ):
                    approx_shard = (full_sum[shard_start:shard_end] / float(world_size)).to(g.dtype)
                    res_list[idx].copy_(g[shard_start:shard_end] - approx_shard)
            else:
                # 非本地 EF 保持原有 dense 方式
                sparse_full = torch.where((g >= v_pos) | (g <= -v_neg), g, torch.zeros_like(g))
                _apply_error_feedback(state, full_flat_grad, shard_out, sparse_full, pg)
        if do_log:
            logger.info("[threshold_v] sparse path: apply_error_feedback done")   

        if do_log and g.device.type == "cuda":
            ev_end.record()
            torch.cuda.synchronize(g.device)
            ms_ef = float(ev_start.elapsed_time(ev_after_ef))
            ms_thresh = float(ev_after_ef.elapsed_time(ev_after_thresh))
            ms_select = float(ev_after_thresh.elapsed_time(ev_after_select))
            ms_allg = float(ev_after_select.elapsed_time(ev_after_allgather))
            ms_copy = float(ev_after_allgather.elapsed_time(ev_after_copy))
            ms_tail = float(ev_after_copy.elapsed_time(ev_end))
            logger.info(
                "[threshold_v][timing] ef=%.3fms thresh=%.3fms select=%.3fms allgather+merge=%.3fms copy=%.3fms ef_update=%.3fms",
                ms_ef,
                ms_thresh,
                ms_select,
                ms_allg,
                ms_copy,
                ms_tail,
            )
        return

    if do_log:
        logger.info("[threshold_v] dense fallback path: build sparse mask")
    sparse = torch.where((g >= v_pos) | (g <= -v_neg), g, torch.zeros_like(g))
    shard_size = numel // world_size
    temp_shard = torch.empty(shard_size, device=g.device, dtype=g.dtype)
    if do_log:
        logger.info("[threshold_v] dense fallback path: reduce_scatter start")
    if hasattr(dist, "reduce_scatter_tensor"):
        dist.reduce_scatter_tensor(temp_shard, sparse, op=dist.ReduceOp.SUM, group=pg)
    else:
        chunks = list(sparse.chunk(world_size, dim=0))
        dist.reduce_scatter(temp_shard, chunks, op=dist.ReduceOp.SUM, group=pg)
    if do_log:
        logger.info("[threshold_v] dense fallback path: reduce_scatter done")

    # 近似统计：按输入 sparse 的元素数估算 reduce_scatter 负载
    try:
        _comm_add_bytes(state, sparse.numel() * sparse.element_size())
    except Exception:
        pass

    deq_avg = (temp_shard / float(world_size)).to(full_flat_grad.dtype)
    shard_out.copy_(deq_avg)

    if state.error_feedback:
        _apply_error_feedback(state, full_flat_grad, shard_out, sparse, pg)


__all__ = ["ThresholdVState", "fsdp_thresholdv_comm_hook"]


