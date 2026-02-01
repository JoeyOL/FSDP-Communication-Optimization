from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional


_COMM_NAME_KEYWORDS = (
    "nccl",
    "allreduce",
    "all_reduce",
    "reduce_scatter",
    "reducescatter",
    "allgather",
    "all_gather",
    "broadcast",
    "c10d",
)

_EXCLUDE_COMPUTE_KEYWORDS = (
    "memcpy",
    "memset",
    "cuda memcpy",
)


# 更严格的 compute 关键词（用于“口径A”）：尽量只把重计算 kernel 计入 compute。
# 注意：这是启发式规则，可能随 CUDA/cuBLAS/FlashAttention/Triton 版本变化。
_STRICT_COMPUTE_KEYWORDS = (
    "gemm",
    "sgemm",
    "hgemm",
    "igemm",
    "cublas",
    "cutlass",
    "cudnn",
    "attention",
    "flash",
    "sdpa",
    "matmul",
    "bmm",
    "fmha",
    "conv",
    "layernorm",
    "rmsnorm",
    "softmax",
    "triton",
    "fused",
)


@dataclass(frozen=True)
class Interval:
    start_us: float
    end_us: float


def _merge_intervals(intervals: list[Interval]) -> list[Interval]:
    if not intervals:
        return []
    intervals = sorted(intervals, key=lambda x: (x.start_us, x.end_us))
    merged: list[Interval] = [intervals[0]]
    for it in intervals[1:]:
        last = merged[-1]
        if it.start_us <= last.end_us:
            merged[-1] = Interval(last.start_us, max(last.end_us, it.end_us))
        else:
            merged.append(it)
    return merged


def _interval_total_us(intervals: Iterable[Interval]) -> float:
    return float(sum(max(0.0, it.end_us - it.start_us) for it in intervals))


def _intersection_total_us(a: list[Interval], b: list[Interval]) -> float:
    # a, b must be merged & sorted
    i = 0
    j = 0
    total = 0.0
    while i < len(a) and j < len(b):
        x = a[i]
        y = b[j]
        start = max(x.start_us, y.start_us)
        end = min(x.end_us, y.end_us)
        if end > start:
            total += end - start
        if x.end_us < y.end_us:
            i += 1
        else:
            j += 1
    return float(total)


def find_latest_trace_file(profiler_dir: Path) -> Optional[Path]:
    if not profiler_dir.exists():
        return None

    # tensorboard_trace_handler 通常会生成 *.pt.trace.json
    candidates = []
    for p in profiler_dir.glob("*.json"):
        name = p.name.lower()
        if name in ("summary_rank0.json",):
            continue
        if name.endswith(".pt.trace.json") or "trace" in name:
            candidates.append(p)

    if not candidates:
        return None

    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _load_trace_events(trace_path: Path) -> list[dict[str, Any]]:
    # Chrome trace 是一个 JSON object，常见为 {"traceEvents": [...]} 或直接是数组
    data = json.loads(trace_path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "traceEvents" in data and isinstance(data["traceEvents"], list):
        return data["traceEvents"]
    if isinstance(data, list):
        return data
    return []


def _is_comm_event(name: str) -> bool:
    n = (name or "").lower()
    return any(k in n for k in _COMM_NAME_KEYWORDS)


def _is_compute_event_loose(name: str) -> bool:
    n = (name or "").lower()
    if any(k in n for k in _EXCLUDE_COMPUTE_KEYWORDS):
        return False
    return True


def _is_compute_event_strict(name: str) -> bool:
    n = (name or "").lower()
    if any(k in n for k in _EXCLUDE_COMPUTE_KEYWORDS):
        return False
    return any(k in n for k in _STRICT_COMPUTE_KEYWORDS)


def _extract_step_windows(events: list[dict[str, Any]]) -> list[Interval]:
    # 优先使用 profiler 自动插入的 ProfilerStep#X 区间
    # 注意：trace 可能在多个 thread/track 上重复产生 ProfilerStep，需对齐到单一 thread。
    windows_by_tid: dict[str, list[Interval]] = {}
    for e in events:
        if e.get("ph") != "X":
            continue
        name = str(e.get("name", ""))
        if not name.startswith("ProfilerStep#"):
            continue
        ts = e.get("ts")
        dur = e.get("dur")
        if ts is None or dur is None:
            continue
        try:
            ts_us = float(ts)
            dur_us = float(dur)
        except Exception:
            continue
        if dur_us <= 0:
            continue
        tid = str(e.get("tid", "0"))
        windows_by_tid.setdefault(tid, []).append(Interval(ts_us, ts_us + dur_us))

    if not windows_by_tid:
        return []

    # 选择 ProfilerStep 数量最多的 thread 作为主线
    best_tid = max(windows_by_tid.items(), key=lambda kv: len(kv[1]))[0]
    windows = windows_by_tid[best_tid]
    return sorted(windows, key=lambda w: w.start_us)


def compute_comm_compute_overlap_from_trace(trace_path: Path) -> dict[str, Any]:
    """从 profiler 的 Chrome trace 估算计算-通信重叠率。

    输出两套口径：

    - "loose"：compute = 非通信 GPU kernel（排除 memcpy/memset），容易高估“重叠”。
    - "strict"（口径A）：compute = 仅匹配少量“重计算 kernel”的启发式关键词（如 gemm/attention/triton）。

    统一定义：
      comm_covered_ratio_* = overlap_ms_* / comm_total_ms
    """

    events = _load_trace_events(trace_path)

    # 只取真正的 GPU kernel 事件（Chrome trace 里通常 cat == "kernel"；避免把 cuda_runtime/python range 误当成 GPU 时间线）
    gpu_events: list[dict[str, Any]] = []
    for e in events:
        if e.get("ph") != "X":
            continue
        cat = str(e.get("cat", "")).lower()
        if cat != "kernel":
            continue
        if e.get("ts") is None or e.get("dur") is None:
            continue
        gpu_events.append(e)

    step_windows = _extract_step_windows(events)

    def intervals_in_window(
        window: Optional[Interval],
    ) -> tuple[list[Interval], list[Interval], list[Interval]]:
        comm: list[Interval] = []
        compute_loose: list[Interval] = []
        compute_strict: list[Interval] = []
        for e in gpu_events:
            name = str(e.get("name", ""))
            try:
                ts_us = float(e.get("ts"))
                dur_us = float(e.get("dur"))
            except Exception:
                continue
            if dur_us <= 0:
                continue
            start = ts_us
            end = ts_us + dur_us
            if window is not None:
                if end <= window.start_us or start >= window.end_us:
                    continue
                start = max(start, window.start_us)
                end = min(end, window.end_us)
            it = Interval(start, end)
            if _is_comm_event(name):
                comm.append(it)
            else:
                if _is_compute_event_loose(name):
                    compute_loose.append(it)
                if _is_compute_event_strict(name):
                    compute_strict.append(it)

        return _merge_intervals(comm), _merge_intervals(compute_loose), _merge_intervals(compute_strict)

    per_step: list[dict[str, Any]] = []
    if step_windows:
        for idx, w in enumerate(step_windows):
            comm_m, compute_loose_m, compute_strict_m = intervals_in_window(w)
            comm_us = _interval_total_us(comm_m)

            compute_loose_us = _interval_total_us(compute_loose_m)
            overlap_loose_us = _intersection_total_us(comm_m, compute_loose_m)
            comm_covered_loose = (overlap_loose_us / comm_us) if comm_us > 0 else 0.0

            compute_strict_us = _interval_total_us(compute_strict_m)
            overlap_strict_us = _intersection_total_us(comm_m, compute_strict_m)
            comm_covered_strict = (overlap_strict_us / comm_us) if comm_us > 0 else 0.0

            per_step.append(
                {
                    "step_idx": int(idx),
                    "comm_total_ms": comm_us / 1000.0,
                    "compute_total_ms_loose": compute_loose_us / 1000.0,
                    "overlap_ms_loose": overlap_loose_us / 1000.0,
                    "comm_covered_ratio_loose": comm_covered_loose,
                    "comm_exposed_ratio_loose": 1.0 - comm_covered_loose,
                    "compute_total_ms_strict": compute_strict_us / 1000.0,
                    "overlap_ms_strict": overlap_strict_us / 1000.0,
                    "comm_covered_ratio_strict": comm_covered_strict,
                    "comm_exposed_ratio_strict": 1.0 - comm_covered_strict,
                }
            )

    # overall（不分 step）
    comm_all, compute_loose_all, compute_strict_all = intervals_in_window(None)
    comm_us_all = _interval_total_us(comm_all)

    compute_loose_us_all = _interval_total_us(compute_loose_all)
    overlap_loose_us_all = _intersection_total_us(comm_all, compute_loose_all)
    comm_covered_loose_all = (overlap_loose_us_all / comm_us_all) if comm_us_all > 0 else 0.0

    compute_strict_us_all = _interval_total_us(compute_strict_all)
    overlap_strict_us_all = _intersection_total_us(comm_all, compute_strict_all)
    comm_covered_strict_all = (overlap_strict_us_all / comm_us_all) if comm_us_all > 0 else 0.0

    return {
        "trace_file": str(trace_path.name),
        "overall": {
            "comm_total_ms": comm_us_all / 1000.0,
            "compute_total_ms_loose": compute_loose_us_all / 1000.0,
            "overlap_ms_loose": overlap_loose_us_all / 1000.0,
            "comm_covered_ratio_loose": comm_covered_loose_all,
            "comm_exposed_ratio_loose": 1.0 - comm_covered_loose_all,
            "compute_total_ms_strict": compute_strict_us_all / 1000.0,
            "overlap_ms_strict": overlap_strict_us_all / 1000.0,
            "comm_covered_ratio_strict": comm_covered_strict_all,
            "comm_exposed_ratio_strict": 1.0 - comm_covered_strict_all,
        },
        "per_step": per_step,
        "step_count": int(len(step_windows)),
        "note": "overlap 指标基于 trace 的 GPU kernel 名称关键词启发式分类：loose 口径 compute=非通信且非 memcpy/memset 的 kernel，strict(口径A) 口径 compute=少量重计算关键词。适合趋势对比（1卡vs2卡），非严格算子级归因。",
    }
