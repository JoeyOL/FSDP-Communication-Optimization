from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch
from torch.profiler import ProfilerActivity
from torch.utils.tensorboard import SummaryWriter

from .trace_overlap import (
    compute_comm_compute_overlap_from_trace,
    find_latest_trace_file,
)


_COMM_KEYWORDS = (
    "reduce_scatter",
    "all_gather",
    "all_reduce",
    "broadcast",
    "nccl",
    "c10d",
)


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


def _step_time_stats(step_times_ms: list[float]) -> dict[str, float]:
    if not step_times_ms:
        return {}
    values = sorted(float(x) for x in step_times_ms)
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


def _summarize_profiler_comm_ops(prof: torch.profiler.profile) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []

    total_cuda_us = 0.0
    total_cpu_us = 0.0
    comm_cuda_us = 0.0
    comm_cpu_us = 0.0

    # key_averages() 会聚合相同 key 的事件
    for evt in prof.key_averages():
        key = (getattr(evt, "key", "") or "").lower()

        # cpu/cuda 时间字段在不同版本上都较稳定（单位 us）
        cuda_us = float(getattr(evt, "cuda_time_total", 0.0) or 0.0)
        cpu_us = float(getattr(evt, "cpu_time_total", 0.0) or 0.0)

        total_cuda_us += cuda_us
        total_cpu_us += cpu_us

        is_comm = any(k in key for k in _COMM_KEYWORDS)
        if is_comm:
            comm_cuda_us += cuda_us
            comm_cpu_us += cpu_us

        rows.append(
            {
                "name": getattr(evt, "key", ""),
                "count": int(getattr(evt, "count", 0) or 0),
                "cpu_time_total_ms": cpu_us / 1000.0,
                "cuda_time_total_ms": cuda_us / 1000.0,
                "is_comm": bool(is_comm),
            }
        )

    # 只导出 comm 相关行（按 cuda_time_total_ms 排序）
    comm_rows = sorted(
        (r for r in rows if r["is_comm"]),
        key=lambda r: (r["cuda_time_total_ms"], r["cpu_time_total_ms"]),
        reverse=True,
    )

    comm_ratio_cuda = (comm_cuda_us / total_cuda_us) if total_cuda_us > 0 else 0.0
    comm_ratio_cpu = (comm_cpu_us / total_cpu_us) if total_cpu_us > 0 else 0.0

    return {
        "total_cuda_time_ms": total_cuda_us / 1000.0,
        "total_cpu_time_ms": total_cpu_us / 1000.0,
        "comm_cuda_time_ms": comm_cuda_us / 1000.0,
        "comm_cpu_time_ms": comm_cpu_us / 1000.0,
        "comm_ratio_cuda": comm_ratio_cuda,
        "comm_ratio_cpu": comm_ratio_cpu,
        "comm_ops": comm_rows,
    }


def _bool_arg(args: Any, name: str, default: bool) -> bool:
    if args is None:
        return default
    return bool(getattr(args, name, default))


def _int_arg(args: Any, name: str, default: int) -> int:
    if args is None:
        return default
    try:
        return int(getattr(args, name, default))
    except Exception:
        return default


@dataclass
class MonitoringContext:
    rank: int
    enabled: bool
    log_dir: Optional[Path] = None
    tb_log_dir: Optional[Path] = None
    profiler_log_dir: Optional[Path] = None
    tb_writer: Optional[SummaryWriter] = None
    prof: Optional[torch.profiler.profile] = None
    step_times_ms: list[float] = field(default_factory=list)


def init_monitoring(args: Any, rank: int) -> MonitoringContext:
    """初始化 TensorBoard + torch.profiler（只在 rank0 启用）。

    约定：
    - `args.profile` 默认为 True（如果训练脚本未提供该参数，则按 True 处理，保持原行为）。
    - `args.profile_step_time` 控制是否采集 step wall time。
    """

    if rank != 0:
        return MonitoringContext(rank=rank, enabled=False)

    enabled = _bool_arg(args, "profile", True)
    if not enabled:
        return MonitoringContext(rank=rank, enabled=False)

    output_dir = Path(getattr(args, "output_dir", "."))
    run_name = getattr(args, "run_name", "run")
    log_dir = output_dir / "logs" / run_name
    tb_log_dir = log_dir / "tensorboard"
    profiler_log_dir = log_dir / "profiler"
    tb_log_dir.mkdir(parents=True, exist_ok=True)
    profiler_log_dir.mkdir(parents=True, exist_ok=True)

    tb_writer = SummaryWriter(log_dir=str(tb_log_dir))

    schedule = torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1)
    prof = torch.profiler.profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=schedule,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(str(profiler_log_dir)),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True,
        with_modules=True,
    )
    prof.start()

    return MonitoringContext(
        rank=rank,
        enabled=True,
        log_dir=log_dir,
        tb_log_dir=tb_log_dir,
        profiler_log_dir=profiler_log_dir,
        tb_writer=tb_writer,
        prof=prof,
    )


def step_begin(ctx: MonitoringContext, args: Any) -> Optional[float]:
    if not ctx.enabled:
        return None
    if not _bool_arg(args, "profile_step_time", False):
        return None
    return time.perf_counter()


def step_end(ctx: MonitoringContext, args: Any, start_t: Optional[float]) -> None:
    if not ctx.enabled:
        return

    if start_t is not None:
        elapsed_ms = (time.perf_counter() - start_t) * 1000.0
        ctx.step_times_ms.append(float(elapsed_ms))

    if ctx.prof is not None:
        # 让 profiler 的 schedule 前进；否则 trace 很可能不会落盘
        ctx.prof.step()


def should_stop_early(args: Any, global_step: int) -> bool:
    max_steps = _int_arg(args, "max_steps", 0)
    return bool(max_steps and (global_step + 1) >= max_steps)


def finalize_monitoring(
    ctx: MonitoringContext,
    *,
    args: Any,
    epoch: int,
    total_loss: float,
    num_batches: int,
) -> None:
    if not ctx.enabled:
        return

    # 1) 停止 profiler
    if ctx.prof is not None:
        ctx.prof.stop()

        # 2) 导出通信算子摘要（rank0）
        try:
            profiler_dir = ctx.profiler_log_dir or (Path(".") / "profiler")
            summary = _summarize_profiler_comm_ops(ctx.prof)

            # 写 JSON
            out_json = profiler_dir / "summary_rank0.json"

            overlap = None
            try:
                trace_path = find_latest_trace_file(profiler_dir)
                if trace_path is not None:
                    overlap = compute_comm_compute_overlap_from_trace(trace_path)
            except Exception:
                overlap = None

            payload = {
                "epoch": int(epoch),
                "num_batches": int(num_batches),
                "total_loss": float(total_loss),
                "avg_loss": float(total_loss / max(1, num_batches)),
                "step_time": _step_time_stats(ctx.step_times_ms),
                "profiler": {
                    "comm_ratio_cuda": summary.get("comm_ratio_cuda", 0.0),
                    "comm_ratio_cpu": summary.get("comm_ratio_cpu", 0.0),
                    "comm_cuda_time_ms": summary.get("comm_cuda_time_ms", 0.0),
                    "comm_cpu_time_ms": summary.get("comm_cpu_time_ms", 0.0),
                    "total_cuda_time_ms": summary.get("total_cuda_time_ms", 0.0),
                    "total_cpu_time_ms": summary.get("total_cpu_time_ms", 0.0),
                },
                "overlap": overlap,
            }
            out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

            # 写 CSV（仅 comm ops）
            out_csv = profiler_dir / "comm_op_summary_rank0.csv"
            with out_csv.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "name",
                        "count",
                        "cpu_time_total_ms",
                        "cuda_time_total_ms",
                    ],
                )
                writer.writeheader()
                for row in summary.get("comm_ops", []):
                    writer.writerow(
                        {
                            "name": row.get("name", ""),
                            "count": row.get("count", 0),
                            "cpu_time_total_ms": f"{float(row.get('cpu_time_total_ms', 0.0)):.6f}",
                            "cuda_time_total_ms": f"{float(row.get('cuda_time_total_ms', 0.0)):.6f}",
                        }
                    )
        except Exception:
            # 监控/摘要失败不应影响训练流程
            pass

    # 3) 写 epoch loss，并关闭 TB
    if ctx.tb_writer is not None:
        avg_epoch_loss = total_loss / max(1, num_batches)
        ctx.tb_writer.add_scalar("Loss/epoch", avg_epoch_loss, epoch)
        ctx.tb_writer.close()
