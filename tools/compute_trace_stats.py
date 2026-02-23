from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

# Allow running this script directly: make repo root importable
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from perf.trace_overlap import compute_comm_compute_overlap_from_trace


def _compute_comm_total_bytes_per_step_per_rank(
    param_numel: int,
    world_size: int,
    comm_hook: str,
    error_feedback: bool,
    ef_local: bool,
    nc_bit_packing: bool = True,
    sparse_comm: bool = True,
    ratio: float = 0.01,
) -> Optional[int]:
    """Estimate total communication bytes sent per rank per step (one backward).

    Uses same conventions as comm_usage.md: reduce_scatter + all_gather (if EF and not ef_local).
    Returns None if hook type is not handled.
    """
    if world_size < 1:
        world_size = 1
    n = param_numel
    w = world_size
    ef_ag = error_feedback and not ef_local  # extra all_gather for full EF

    hook = (comm_hook or "none").lower().strip()
    if hook == "none":
        rs = n * 4  # float32
        ag = (n * 4) if ef_ag else 0
        return rs + ag
    if hook == "fp16":
        rs = n * 2
        ag = (n * 2) if ef_ag else 0
        return rs + ag
    if hook == "int8":
        rs = n * 1
        ag = (n * 1) if ef_ag else 0
        return rs + ag
    if hook in ("qsgd", "signsgd", "onebit"):
        # reduce_scatter in float16 or low-bit; all_gather same if EF
        rs = n * 2
        ag = (n * 2) if ef_ag else 0
        return rs + ag
    if hook == "nc":
        if nc_bit_packing:
            # all_to_all: each rank sends W shards of ceil(shard_el/8)*9 bytes
            shard_el = n // w
            packed_per_shard = math.ceil(shard_el / 8) * 9
            rs = w * packed_per_shard
        else:
            rs = n * 4  # float32 reduce_scatter
        ag = (n * 4) if ef_ag else 0
        return rs + ag
    if hook in ("topk", "randomk", "thresholdv"):
        # 稀疏通信：仅发送 (indices, values)，每 rank 发送量 ~ k*(sizeof(index)+sizeof(value))
        if sparse_comm:
            k = max(1, int(n * max(ratio, 0.0)))
            # indices:int64=8B, values:fp32=4B
            main = k * (8 + 4)
            ag = (n * 4) if ef_ag else 0  # EF 仍然需要一次 full all_gather(shard_out)
            return main + ag
        # 稠密通信：补零后走 reduce_scatter，与 baseline 相同
        rs = n * 4
        ag = (n * 4) if ef_ag else 0
        return rs + ag
    if hook in ("sketch", "hybrid_topk_int8", "onebit_seide"):
        # 复杂/多轮稀疏通信：暂用 baseline 作为上界近似
        rs = n * 4
        ag = (n * 4) if ef_ag else 0
        return rs + ag
    return None


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
    # some traces use these wrapper names
    "_all_gather_base",
    "_reduce_scatter_base",
)

_SCOPE_NAMES_DEFAULT = (
    "forward_pass",
    "backward_pass",
    "optimizer_step",
)


def _load_trace_events(trace_path: Path) -> list[dict[str, Any]]:
    data = json.loads(trace_path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "traceEvents" in data and isinstance(data["traceEvents"], list):
        return data["traceEvents"]
    if isinstance(data, list):
        return data
    return []


def _is_comm_name(name: str) -> bool:
    n = (name or "").lower()
    return any(k in n for k in _COMM_NAME_KEYWORDS)


def _summarize_kernels(trace_path: Path, *, comm_only: bool) -> dict[str, Any]:
    """Summarize GPU kernels from trace.

    - Only counts Chrome trace events with ph=="X" and cat=="kernel".
    - Aggregates by exact kernel name.
    - When comm_only=True, only keeps kernels matching _COMM_NAME_KEYWORDS.
    """

    events = _load_trace_events(trace_path)

    rows: dict[str, dict[str, float]] = defaultdict(lambda: {"count": 0.0, "dur_us": 0.0, "is_comm": 0.0})
    total_kernel_dur_us = 0.0
    comm_kernel_dur_us = 0.0

    for e in events:
        if e.get("ph") != "X":
            continue
        cat = str(e.get("cat", "")).lower()
        if cat != "kernel":
            continue
        dur = e.get("dur")
        if dur is None:
            continue
        try:
            dur_us = float(dur)
        except Exception:
            continue
        if dur_us <= 0:
            continue

        name = str(e.get("name", ""))
        total_kernel_dur_us += dur_us

        is_comm = _is_comm_name(name)
        if is_comm:
            comm_kernel_dur_us += dur_us

        if comm_only and not is_comm:
            continue

        rows[name]["count"] += 1.0
        rows[name]["dur_us"] += dur_us
        rows[name]["is_comm"] = 1.0 if is_comm else 0.0

    kernel_rows = [
        {
            "name": name,
            "count": int(v["count"]),
            "cuda_time_total_ms": v["dur_us"] / 1000.0,
            "is_comm": bool(v.get("is_comm", 0.0)),
        }
        for name, v in rows.items()
    ]
    kernel_rows.sort(key=lambda r: r["cuda_time_total_ms"], reverse=True)

    comm_ratio_kernel_sum = (comm_kernel_dur_us / total_kernel_dur_us) if total_kernel_dur_us > 0 else 0.0

    return {
        "total_kernel_time_ms_sum": total_kernel_dur_us / 1000.0,
        "comm_kernel_time_ms_sum": comm_kernel_dur_us / 1000.0,
        "comm_ratio_kernel_sum": comm_ratio_kernel_sum,
        "kernels": kernel_rows,
    }


def _percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    if q <= 0:
        return float(sorted_values[0])
    if q >= 1:
        return float(sorted_values[-1])
    idx = int(round(q * (len(sorted_values) - 1)))
    idx = max(0, min(len(sorted_values) - 1, idx))
    return float(sorted_values[idx])


def _stats_ms(values_ms: list[float]) -> dict[str, float]:
    if not values_ms:
        return {}
    values = sorted(float(x) for x in values_ms)
    mean = sum(values) / len(values)
    return {
        "count": float(len(values)),
        "mean_ms": float(mean),
        "p50_ms": _percentile(values, 0.50),
        "p90_ms": _percentile(values, 0.90),
        "p95_ms": _percentile(values, 0.95),
        "min_ms": float(values[0]),
        "max_ms": float(values[-1]),
    }


class _Interval:
    __slots__ = ("start_us", "end_us")

    def __init__(self, start_us: float, end_us: float) -> None:
        self.start_us = float(start_us)
        self.end_us = float(end_us)


def _extract_step_windows(events: list[dict[str, Any]]) -> list[_Interval]:
    windows_by_tid: dict[str, list[_Interval]] = {}
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
        windows_by_tid.setdefault(tid, []).append(_Interval(ts_us, ts_us + dur_us))

    if not windows_by_tid:
        return []

    best_tid = max(windows_by_tid.items(), key=lambda kv: len(kv[1]))[0]
    windows = windows_by_tid[best_tid]
    windows.sort(key=lambda w: w.start_us)
    return windows


def _overlap_us(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    start = max(a_start, b_start)
    end = min(a_end, b_end)
    return max(0.0, end - start)


def _summarize_scopes(trace_path: Path, *, scope_names: tuple[str, ...] = _SCOPE_NAMES_DEFAULT) -> dict[str, Any]:
    events = _load_trace_events(trace_path)
    step_windows = _extract_step_windows(events)

    # Gather scope intervals (CPU ranges, usually cat=user_annotation)
    scope_intervals: dict[str, list[_Interval]] = {name: [] for name in scope_names}
    for e in events:
        if e.get("ph") != "X":
            continue
        name = str(e.get("name", ""))
        if name not in scope_intervals:
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
        scope_intervals[name].append(_Interval(ts_us, ts_us + dur_us))

    # Per-step sums
    per_step: list[dict[str, Any]] = []
    for idx, w in enumerate(step_windows):
        row: dict[str, Any] = {"step_idx": int(idx)}
        total_ms = 0.0
        for name in scope_names:
            s = 0.0
            for it in scope_intervals[name]:
                s += _overlap_us(it.start_us, it.end_us, w.start_us, w.end_us)
            ms = s / 1000.0
            row[f"{name}_ms"] = ms
            total_ms += ms
        row["sum_scopes_ms"] = total_ms
        per_step.append(row)

    overall: dict[str, Any] = {}
    for name in scope_names:
        values = [float(r.get(f"{name}_ms", 0.0)) for r in per_step]
        overall[name] = {"total_ms": float(sum(values)), "stats": _stats_ms(values)}

    overall_values = [float(r.get("sum_scopes_ms", 0.0)) for r in per_step]
    overall["sum_scopes"] = {"total_ms": float(sum(overall_values)), "stats": _stats_ms(overall_values)}

    return {
        "scope_names": list(scope_names),
        "step_count": int(len(step_windows)),
        "overall": overall,
        "per_step": per_step,
        "note": "record_function(name) 产生的是 CPU scope 事件（常见 cat=user_annotation）。这里按 ProfilerStep#X 作为 step 窗口，对每个 scope 做区间重叠求和。",
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Compute Step1 statistics from a single PyTorch profiler Chrome trace (*.pt.trace.json). "
            "This runs fully offline and does not require training to compute summary files."
        )
    )
    ap.add_argument("trace_path", type=str, help="Path to *.pt.trace.json")
    ap.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory for generated files (default: same directory as trace).",
    )
    ap.add_argument(
        "--json_name",
        type=str,
        default="summary_from_trace.json",
        help="Output JSON filename.",
    )
    ap.add_argument(
        "--csv_name",
        type=str,
        default="comm_kernel_summary.csv",
        help="Output CSV filename (comm kernels only).",
    )
    ap.add_argument(
        "--csv_all_kernels",
        action="store_true",
        help=(
            "Write ALL GPU kernel ops into the CSV (aggregated by kernel name). "
            "When not set, CSV contains only comm-related kernels."
        ),
    )
    ap.add_argument(
        "--csv_topk",
        type=int,
        default=50,
        help="If >0, keep only top-K rows by cuda_time_total_ms.",
    )

    ap.add_argument(
        "--compat_rank0_names",
        action="store_true",
        help="Write legacy names: summary_rank0.json + comm_op_summary_rank0.csv (trace-derived).",
    )
    ap.add_argument(
        "--param_numel",
        type=int,
        default=None,
        help="Total model param count (for comm_total_bytes when param_numel.txt missing, e.g. old runs).",
    )
    args = ap.parse_args()

    trace_path = Path(args.trace_path)
    if not trace_path.exists():
        raise SystemExit(f"Trace file not found: {trace_path}")

    out_dir = Path(args.out_dir) if args.out_dir else trace_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    overlap = compute_comm_compute_overlap_from_trace(trace_path)
    kernel_summary = _summarize_kernels(trace_path, comm_only=not args.csv_all_kernels)
    scopes = _summarize_scopes(trace_path)

    # 通信总字节数（每 step 每 rank 发送量）：从 log_dir 的 config.json + param_numel.txt（或 --param_numel）推算。
    # 注意：我们先算好这些 summary 字段，下面按顺序构造 payload，保证它们出现在 JSON 的最前面。
    comm_bytes: Optional[int] = None
    comm_bytes_note: Optional[str] = None

    log_dir = out_dir.parent

    # 1) 优先使用运行时轻量通信统计器写出的 comm_stats_rank0.json
    comm_stats_path = log_dir / "comm_stats_rank0.json"
    if comm_stats_path.exists():
        try:
            cs = json.loads(comm_stats_path.read_text(encoding="utf-8"))
            b = cs.get("bytes_per_step_per_rank", None)
            if isinstance(b, (int, float)) and b > 0:
                comm_bytes = int(b)
                comm_bytes_note = (
                    "from runtime perf.comm_stats (sum over all comm hooks); "
                    "per step, per rank send volume."
                )
        except Exception:
            pass

    # 2) 若没有 runtime 统计，则回退到基于 config+param_numel 的离线估算
    if comm_bytes is None:
        config_path = log_dir / "config.json"
        param_path = log_dir / "param_numel.txt"
        param_numel_val: Optional[int] = None
        if args.param_numel is not None and args.param_numel > 0:
            param_numel_val = args.param_numel
        elif param_path.exists():
            try:
                param_numel_val = int(param_path.read_text(encoding="utf-8").strip())
            except Exception:
                pass
        if config_path.exists() and param_numel_val is not None:
            try:
                config = json.loads(config_path.read_text(encoding="utf-8"))
                world_size = int(config.get("nproc", 1)) or 1
                comm_hook = config.get("comm_hook", "none") or "none"
                error_feedback = config.get("comm_error_feedback", False) is True
                ef_local = config.get("comm_ef_local", True) is True
                nc_bit_packing = config.get("comm_nc_bit_packing", True) is True
                sparse_comm = config.get("comm_sparse_comm", True) is True
                ratio_val = float(config.get("comm_topk_ratio", 0.01) or 0.01)
                total_bytes = _compute_comm_total_bytes_per_step_per_rank(
                    param_numel_val,
                    world_size,
                    comm_hook,
                    error_feedback,
                    ef_local,
                    nc_bit_packing,
                    sparse_comm,
                    ratio_val,
                )
                if total_bytes is not None:
                    comm_bytes = total_bytes
                    comm_bytes_note = (
                        "estimated from param_numel + config (comm_hook, error_feedback, ef_local); "
                        "per step, per rank send volume."
                    )
            except Exception:
                pass

    # 构造最终 payload，按“summary 在前、细节在后”的顺序插 key，保证 JSON 顶部优先展示整体指标。
    payload: dict[str, Any] = {}
    if comm_bytes is not None:
        payload["comm_total_bytes_per_step_per_rank"] = comm_bytes
        if comm_bytes_note is not None:
            payload["comm_total_bytes_note"] = comm_bytes_note

    # overlap/scopes 的 overall 也单独提前一份，便于快速查看整体情况
    payload["overlap_overall"] = overlap.get("overall", {})
    payload["scopes_overall"] = scopes.get("overall", {})

    # 其余详细信息随后
    payload["trace_file"] = str(trace_path.name)
    payload["overlap"] = overlap
    payload["comm_from_trace"] = {
        "comm_ratio_kernel_sum": kernel_summary["comm_ratio_kernel_sum"],
        "comm_kernel_time_ms_sum": kernel_summary["comm_kernel_time_ms_sum"],
        "total_kernel_time_ms_sum": kernel_summary["total_kernel_time_ms_sum"],
    }
    payload["scopes"] = scopes

    if args.compat_rank0_names:
        json_name = "summary_rank0.json"
        csv_name = "comm_op_summary_rank0.csv"
    else:
        json_name = args.json_name
        csv_name = args.csv_name

    out_json = out_dir / json_name
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    out_csv = out_dir / csv_name
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "count", "cuda_time_total_ms", "is_comm"])
        writer.writeheader()
        rows = list(kernel_summary["kernels"])
        if args.csv_topk and args.csv_topk > 0:
            rows = rows[: int(args.csv_topk)]
        for r in rows:
            writer.writerow(
                {
                    "name": r["name"],
                    "count": r["count"],
                    "cuda_time_total_ms": f"{float(r['cuda_time_total_ms']):.6f}",
                    "is_comm": str(bool(r.get("is_comm", False))),
                }
            )

    print(f"Wrote: {out_json}")
    print(f"Wrote: {out_csv}")


if __name__ == "__main__":
    main()
