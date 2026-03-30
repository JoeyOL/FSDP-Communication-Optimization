import torch
import torch.distributed as dist

from .common import _apply_error_feedback, _ensure_residual, logger
from perf.comm_stats import add_bytes as _comm_add_bytes


# 仅在前几次调用、且 rank0 上打印关键步骤，帮助定位耗时环节
_SKETCH_LOG_CAP = 50
_sketch_log_count = 0


def _count_sketch_encode(g: torch.Tensor, width: int, depth: int) -> torch.Tensor:
    d = g.numel()
    sketch = torch.zeros(depth, width, device=g.device, dtype=g.dtype)
    # 预先构建索引和符号，避免在每一层重复创建大向量
    idx = torch.arange(d, device=g.device)
    s = 2 * (idx % 2) - 1
    for dep in range(depth):
        h = (idx * (dep + 1) + dep) % width
        # 使用每一行的 1D 向量做 index_add_，避免在 dim=1 上对 1D source 进行索引导致越界。
        sketch[dep].index_add_(0, h, g * s)
    return sketch


def _count_sketch_decode_all(sketch: torch.Tensor, d: int, width: int, depth: int) -> torch.Tensor:
    est = torch.zeros(d, device=sketch.device, dtype=sketch.dtype)
    # 同样复用 idx 和符号，减少大向量重复构造开销
    idx = torch.arange(d, device=sketch.device)
    s = 2 * (idx % 2) - 1
    for dep in range(depth):
        h = (idx * (dep + 1) + dep) % width
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
    # Deterministic seed so all ranks select the same random fill (merged_sketch is identical after all_reduce)
    gen = torch.Generator(device=NH_indices.device)
    gen.manual_seed(int(est_sq.sum().item() * 1e6) % (2**31))
    perm = torch.randperm(NH_indices.numel(), device=NH_indices.device, generator=gen)[:l]
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
    rank = dist.get_rank(pg)

    global _sketch_log_count
    _sketch_log_count += 1
    do_log = rank == 0 and _sketch_log_count <= _SKETCH_LOG_CAP

    if world_size == 1:
        shard_out.copy_(full_flat_grad)
        return

    g = full_flat_grad.contiguous().view(-1)
    d = g.numel()
    k = state.k if state.k > 0 else max(1, int(d * state.ratio))
    k = min(k, d)
    width = state._sketch_size(d)
    depth = state._sketch_depth

    if do_log:
        logger.info(
            "[sketch] hook_start d=%s k=%s ratio=%.4f two_round=%s ef=%s (call#%s)",
            d,
            k,
            state.ratio,
            getattr(state, "two_round", False),
            state.error_feedback,
            _sketch_log_count,
        )
        if g.device.type == "cuda":
            ev_start = torch.cuda.Event(enable_timing=True)
            ev_after_ef = torch.cuda.Event(enable_timing=True)
            ev_after_encode = torch.cuda.Event(enable_timing=True)
            ev_after_allreduce = torch.cuda.Event(enable_timing=True)
            ev_after_heavy = torch.cuda.Event(enable_timing=True)
            ev_after_round2_gather = torch.cuda.Event(enable_timing=True)
            ev_after_reduce_scatter = torch.cuda.Event(enable_timing=True)
            ev_end = torch.cuda.Event(enable_timing=True)
            ev_start.record()

    if state.error_feedback:
        residual, start, end = _ensure_residual(state, g, pg)
        if start is None:
            if do_log:
                logger.info("[sketch] apply_error_feedback full residual")
            g = (g + residual).to(g.dtype)
        else:
            if do_log:
                logger.info("[sketch] apply_error_feedback local residual slice [%s:%s]", start, end)
            g[start:end] += residual

    if do_log and g.device.type == "cuda":
        ev_after_ef.record()

    if getattr(state, "two_round", False):
        if do_log:
            logger.info("[sketch] two_round: encode sketch")
        sketch = state._count_sketch(g, width, depth)
        if do_log and g.device.type == "cuda":
            ev_after_encode.record()

        sketch_flat = sketch.view(-1)
        if hasattr(dist, "all_reduce"):
            if do_log:
                logger.info("[sketch] two_round: all_reduce sketch start")
            dist.all_reduce(sketch_flat, op=dist.ReduceOp.SUM, group=pg)
        merged_sketch = sketch_flat.view(depth, width) / float(world_size)
        if do_log and g.device.type == "cuda":
            ev_after_allreduce.record()

        if do_log:
            logger.info("[sketch] two_round: HEAVYMIX topk")
        topk_indices = _heavymix_topk_indices(merged_sketch, d, width, depth, k)
        values_at_topk = g[topk_indices]
        if do_log and g.device.type == "cuda":
            ev_after_heavy.record()

        if hasattr(dist, "all_gather_into_tensor"):
            if do_log:
                logger.info("[sketch] two_round: all_gather values start")
            all_vals = torch.empty(world_size * k, device=g.device, dtype=g.dtype)
            dist.all_gather_into_tensor(all_vals, values_at_topk.contiguous(), group=pg)
        else:
            if do_log:
                logger.info("[sketch] two_round: all_gather values (list) start")
            all_vals_list = [torch.empty_like(values_at_topk) for _ in range(world_size)]
            dist.all_gather(all_vals_list, values_at_topk.contiguous(), group=pg)
            all_vals = torch.cat(all_vals_list, dim=0)
        if do_log and g.device.type == "cuda":
            ev_after_round2_gather.record()

        full_sum = torch.zeros(d, device=g.device, dtype=g.dtype)
        full_sum.scatter_add_(0, topk_indices, all_vals.view(world_size, k).sum(dim=0))
        shard_size = d // world_size
        shard_start = rank * shard_size
        shard_end = shard_start + shard_size
        deq_avg = (full_sum[shard_start:shard_end] / float(world_size)).to(full_flat_grad.dtype)
        shard_out.copy_(deq_avg)

        if state.error_feedback:
            # 不要再次调用 _ensure_residual：本参数已在上面调用过一次，再调会多占槽位导致后续参数错位
            full_reconstructed = torch.zeros_like(g)
            full_reconstructed[topk_indices] = full_sum[topk_indices] / float(world_size)
            diff = g - full_reconstructed
            idx = getattr(state, "_ef_index", 1) - 1
            if idx >= 0 and getattr(state, "_ef_residual_list", None) and idx < len(state._ef_residual_list):
                residual = state._ef_residual_list[idx]
                if getattr(state, "_ef_local", False):
                    start = getattr(state, "_ef_shard_start", 0)
                    end = getattr(state, "_ef_shard_end", 0)
                    if residual.numel() == (end - start) and end <= diff.numel():
                        residual.copy_(diff[start:end])
                elif residual.numel() == diff.numel():
                    residual.copy_(diff)

        # 通信字节统计（两轮）：
        # Round1: all_reduce(sketch_flat)，payload 大小约 depth * width * sizeof(float)
        # Round2: all_gather(values_at_topk)，payload 大小约 k * sizeof(float)
        try:
            elem_size = g.element_size()
            round1_bytes = depth * width * elem_size
            round2_bytes = k * elem_size
            _comm_add_bytes(state, round1_bytes + round2_bytes)
        except Exception:
            pass

        if do_log and g.device.type == "cuda":
            ev_end.record()
            torch.cuda.synchronize(g.device)
            ms_ef = float(ev_start.elapsed_time(ev_after_ef))
            ms_encode = float(ev_after_ef.elapsed_time(ev_after_encode))
            ms_allreduce = float(ev_after_encode.elapsed_time(ev_after_allreduce))
            ms_heavy = float(ev_after_allreduce.elapsed_time(ev_after_heavy))
            ms_round2 = float(ev_after_heavy.elapsed_time(ev_after_round2_gather))
            ms_tail = float(ev_after_round2_gather.elapsed_time(ev_end))
            logger.info(
                "[sketch][timing two_round] ef=%.3fms encode=%.3fms allreduce=%.3fms heavy=%.3fms round2_gather=%.3fms tail(decode+copy+ef)=%.3fms",
                ms_ef,
                ms_encode,
                ms_allreduce,
                ms_heavy,
                ms_round2,
                ms_tail,
            )
        return

    if do_log:
        logger.info("[sketch] one_round: encode sketch")
    sketch = state._count_sketch(g, width, depth)
    if do_log and g.device.type == "cuda":
        ev_after_encode.record()

    if do_log:
        logger.info("[sketch] one_round: recover_heavy")
    sparse = state._recover_heavy(g, sketch, width, depth, k)

    shard_size = d // world_size
    temp_shard = torch.empty(shard_size, device=g.device, dtype=g.dtype)
    if do_log:
        logger.info("[sketch] one_round: reduce_scatter start")
    if hasattr(dist, "reduce_scatter_tensor"):
        dist.reduce_scatter_tensor(temp_shard, sparse, op=dist.ReduceOp.SUM, group=pg)
    else:
        chunks = list(sparse.chunk(world_size, dim=0))
        dist.reduce_scatter(temp_shard, chunks, op=dist.ReduceOp.SUM, group=pg)
    # 近似统计：按 sparse 元素数估算 reduce_scatter 负载
    try:
        _comm_add_bytes(state, sparse.numel() * sparse.element_size())
    except Exception:
        pass
    if do_log and g.device.type == "cuda":
        ev_after_reduce_scatter.record()

    deq_avg = (temp_shard / float(world_size)).to(full_flat_grad.dtype)
    shard_out.copy_(deq_avg)
    if state.error_feedback:
        _apply_error_feedback(state, full_flat_grad, shard_out, sparse, pg)

    if do_log and g.device.type == "cuda":
        ev_end.record()
        torch.cuda.synchronize(g.device)
        ms_ef = float(ev_start.elapsed_time(ev_after_ef))
        ms_encode = float(ev_after_ef.elapsed_time(ev_after_encode))
        ms_reduce_scatter = float(ev_after_encode.elapsed_time(ev_after_reduce_scatter))
        ms_tail = float(ev_after_reduce_scatter.elapsed_time(ev_end))
        logger.info(
            "[sketch][timing one_round] ef=%.3fms encode=%.3fms reduce_scatter=%.3fms tail(recover+copy+ef)=%.3fms",
            ms_ef,
            ms_encode,
            ms_reduce_scatter,
            ms_tail,
        )


__all__ = ["SketchState", "fsdp_sketch_comm_hook"]


