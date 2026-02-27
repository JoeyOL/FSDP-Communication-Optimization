#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

# Use non-interactive backend for servers
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def load_metrics(log_dir: Path) -> Dict[str, Any]:
    """Load training_metrics.json from a single log directory."""
    metrics_path = log_dir / "training_metrics.json"
    if not metrics_path.is_file():
        raise FileNotFoundError(f"training_metrics.json not found in {log_dir}")
    with open(metrics_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _pretty_run_name(raw: str) -> str:
    """
    Convert run_name like:
      step2_profile.sh-sketch_two_round-20260227-050041
    into a human-friendly legend:
      sketch two round

    Rules:
    - Drop leading script name (前缀到第一个 '-' 为止)
    - Drop trailing时间戳部分（形如 -YYYYMMDD-HHMMSS）
    - Replace '_' with space
    """
    name = raw
    # 去掉前缀脚本名: 保留第一个 '-' 之后的部分
    if "-" in name:
        parts = name.split("-")
        if len(parts) >= 2:
            name = "-".join(parts[1:])

    # 去掉末尾时间戳: 形如 -20260227-050041
    m = re.search(r"(.*)-\d{8}-\d{6}$", name)
    if m:
        name = m.group(1)

    # 下划线替换为空格
    name = name.replace("_", " ")
    return name


def _extract_curve(entries: List[Dict[str, Any]]) -> tuple[List[int], List[float]]:
    """Extract (steps, values) from a list of {step, value, ...} dicts."""
    steps: List[int] = []
    values: List[float] = []
    for item in entries:
        if "step" not in item or "value" not in item:
            continue
        steps.append(int(item["step"]))
        values.append(float(item["value"]))
    return steps, values


def plot_metric(
    runs: List[Dict[str, Any]],
    metric_key: str,
    title: str,
    ylabel: str,
    out_path: Path,
) -> None:
    """Plot one metric (e.g. loss_curve, memory_allocated_gb) for all runs."""
    plt.figure(figsize=(8, 5))

    has_data = False
    for run in runs:
        metrics = run["metrics"]
        name = run["name"]
        curve = metrics.get(metric_key, [])
        if not curve:
            continue
        steps, values = _extract_curve(curve)
        if not steps:
            continue
        has_data = True
        plt.plot(steps, values, label=name)

    if not has_data:
        plt.close()
        return

    plt.title(title)
    plt.xlabel("step")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_memory_bar(
    runs: List[Dict[str, Any]],
    metric_key: str,
    title: str,
    ylabel: str,
    out_path: Path,
) -> None:
    """
    Plot a bar chart for memory metrics:
    - y 轴：每个方法在该 metric（allocated/reserved）上的所有 step 平均值
    - x 轴：方法名（run name）
    """
    names: List[str] = []
    means: List[float] = []

    for run in runs:
        metrics = run["metrics"]
        name = run["name"]
        curve = metrics.get(metric_key, [])
        if not curve:
            continue
        _, values = _extract_curve(curve)
        if not values:
            continue
        m = sum(values) / float(len(values))
        names.append(name)
        means.append(m)

    if not names:
        return

    plt.figure(figsize=(8, 5))
    x = range(len(names))
    # 为每个方法分配不同颜色，提升可区分性
    cmap = plt.get_cmap("tab20")
    colors = [cmap(i % cmap.N) for i in x]
    plt.bar(x, means, color=colors)
    plt.xticks(x, names, rotation=30, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Given a list of log directories (each containing training_metrics.json), "
            "generate comparison plots for loss_curve, memory_allocated_gb, "
            "and memory_reserved_gb."
        )
    )
    
    parser.add_argument(
        "--out-dir",
        type=str,
        default=".",
        help="Directory to save output PNG figures (default: current directory).",
    )

    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    runs: List[Dict[str, Any]] = []
    
    log_dirs = [
        "fsdp_output/logs/step2_profile.sh-baseline-20260226-133107",
        "fsdp_output/logs/step2_profile.sh-int8_linear-20260226-133816",
        "fsdp_output/logs/step2_profile.sh-nc-20260227-064238",
        "fsdp_output/logs/step2_profile.sh-onebit_seide-20260227-071239",
        "fsdp_output/logs/step2_profile.sh-qsgd-20260227-044543",
        "fsdp_output/logs/step2_profile.sh-threshold_v-20260227-045317",
        "fsdp_output/logs/step2_profile.sh-sketch-20260227-050958",
        "fsdp_output/logs/step2_profile.sh-randomk-20260226-141728",
        "fsdp_output/logs/step2_profile.sh-topk-20260226-141013",
    ]

    for log_dir_str in log_dirs:
        log_dir = Path(log_dir_str).resolve()
        metrics = load_metrics(log_dir)
        # Prefer run_name from metrics; fall back to directory name
        raw_name = metrics.get("run_name") or log_dir.name
        name = _pretty_run_name(str(raw_name))
        runs.append({"name": name, "metrics": metrics})

    # Loss curve
    plot_metric(
        runs,
        metric_key="loss_curve",
        title="Training Loss Curve",
        ylabel="loss",
        out_path=out_dir / "loss_curve.png",
    )

    # Memory allocated (GB) - 柱状图：各方法的 step 平均值
    plot_memory_bar(
        runs,
        metric_key="memory_allocated_gb",
        title="GPU Memory Allocated (GB)",
        ylabel="allocated (GB)",
        out_path=out_dir / "memory_allocated_gb.png",
    )

    # Memory reserved (GB) - 柱状图：各方法的 step 平均值
    plot_memory_bar(
        runs,
        metric_key="memory_reserved_gb",
        title="GPU Memory Reserved (GB)",
        ylabel="reserved (GB)",
        out_path=out_dir / "memory_reserved_gb.png",
    )


if __name__ == "__main__":
    main()

