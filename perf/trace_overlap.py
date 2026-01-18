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


def _is_compute_event(name: str) -> bool:
    n = (name or "").lower()
    if any(k in n for k in _EXCLUDE_COMPUTE_KEYWORDS):
        return False
    return True


def _extract_step_windows(events: list[dict[str, Any]]) -> list[Interval]:
    # 优先使用 profiler 自动插入的 ProfilerStep#X 区间
    windows: list[Interval] = []
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
        windows.append(Interval(ts_us, ts_us + dur_us))

    return sorted(windows, key=lambda w: w.start_us)


def compute_comm_compute_overlap_from_trace(trace_path: Path) -> dict[str, Any]:
    """从 profiler 的 Chrome trace 估算计算-通信重叠率。

    口径（GPU 时间线近似）：
    - comm intervals：名字包含 NCCL / all_reduce / reduce_scatter / all_gather 等关键字的 GPU kernel 事件。
    - compute intervals：其它 GPU kernel 事件（排除 memcpy/memset）。

    overlap_ratio（通信覆盖比）定义：
      overlap_ms / comm_total_ms
    """

    events = _load_trace_events(trace_path)

    # 只取 GPU kernel（cat 常见包含 "Kernel"，也可能是 "gpu"）
    gpu_events: list[dict[str, Any]] = []
    for e in events:
        if e.get("ph") != "X":
            continue
        cat = str(e.get("cat", ""))
        if ("Kernel" not in cat) and ("gpu" not in cat.lower()) and ("cuda" not in cat.lower()):
            continue
        if e.get("ts") is None or e.get("dur") is None:
            continue
        gpu_events.append(e)

    step_windows = _extract_step_windows(events)

    def intervals_in_window(window: Optional[Interval]) -> tuple[list[Interval], list[Interval]]:
        comm: list[Interval] = []
        compute: list[Interval] = []
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
            elif _is_compute_event(name):
                compute.append(it)

        return _merge_intervals(comm), _merge_intervals(compute)

    per_step: list[dict[str, Any]] = []
    if step_windows:
        for idx, w in enumerate(step_windows):
            comm_m, compute_m = intervals_in_window(w)
            comm_us = _interval_total_us(comm_m)
            compute_us = _interval_total_us(compute_m)
            overlap_us = _intersection_total_us(comm_m, compute_m)
            per_step.append(
                {
                    "step_idx": int(idx),
                    "comm_total_ms": comm_us / 1000.0,
                    "compute_total_ms": compute_us / 1000.0,
                    "overlap_ms": overlap_us / 1000.0,
                    "comm_covered_ratio": (overlap_us / comm_us) if comm_us > 0 else 0.0,
                }
            )

    # overall（不分 step）
    comm_all, compute_all = intervals_in_window(None)
    comm_us_all = _interval_total_us(comm_all)
    compute_us_all = _interval_total_us(compute_all)
    overlap_us_all = _intersection_total_us(comm_all, compute_all)

    return {
        "trace_file": str(trace_path.name),
        "overall": {
            "comm_total_ms": comm_us_all / 1000.0,
            "compute_total_ms": compute_us_all / 1000.0,
            "overlap_ms": overlap_us_all / 1000.0,
            "comm_covered_ratio": (overlap_us_all / comm_us_all) if comm_us_all > 0 else 0.0,
            "comm_exposed_ratio": 1.0 - ((overlap_us_all / comm_us_all) if comm_us_all > 0 else 0.0),
        },
        "per_step": per_step,
        "step_count": int(len(step_windows)),
        "note": "该指标基于 trace 的 GPU kernel 名称关键词近似分类，适合趋势对比（1卡vs2卡），非严格算子级归因。",
    }
