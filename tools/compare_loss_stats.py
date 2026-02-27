#!/usr/bin/env python3
"""
Compare loss curves between a baseline run and one or more other runs.

For each non-baseline run, this script:
- Loads `training_metrics.json` from the given log_dir.
- Extracts `loss_curve` (list of {step, value, ...}).
- Aligns with baseline by `step` (intersection of step sets).
- Computes per-step difference: diff(step) = loss_other(step) - loss_baseline(step).
- Reports mean(diff) and variance(diff) over the aligned steps.

Usage example:

python tools/compare_loss_stats.py \
  --baseline fsdp_output/logs/step2_profile.sh-baseline-20260226-133107 \
  --others \
    fsdp_output/logs/step2_profile.sh-nc-20260227-043522 \
    fsdp_output/logs/step2_profile.sh-int8_linear-20260226-133816
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _load_loss_curve(log_dir: Path) -> Dict[int, float]:
    """Load loss_curve from training_metrics.json in log_dir and return {step: value}."""
    metrics_path = log_dir / "training_metrics.json"
    if not metrics_path.is_file():
        raise FileNotFoundError(f"training_metrics.json not found in {log_dir}")
    with open(metrics_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    loss_curve = data.get("loss_curve", [])
    if not isinstance(loss_curve, list):
        raise ValueError(f"'loss_curve' is not a list in {metrics_path}")

    out: Dict[int, float] = {}
    for item in loss_curve:
        if not isinstance(item, dict):
            continue
        if "step" not in item or "value" not in item:
            continue
        step = int(item["step"])
        value = float(item["value"])
        # If duplicate steps exist, keep the last one
        out[step] = value
    return out


def _aligned_diffs(
    base: Dict[int, float],
    other: Dict[int, float],
) -> Tuple[List[int], List[float]]:
    """Return (steps, diffs) on intersection of steps: diff = other - base."""
    common_steps = sorted(set(base.keys()) & set(other.keys()))
    diffs: List[float] = []
    for s in common_steps:
        diffs.append(other[s] - base[s])
    return common_steps, diffs


def _mean_and_variance(values: List[float]) -> Tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    n = float(len(values))
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / n
    return mean, var


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute mean and variance of loss differences between a baseline run "
            "and one or more other runs."
        )
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default='fsdp_output/logs/step2_profile.sh-baseline-20260226-133107',
        help="Baseline log_dir containing training_metrics.json.",
    )
    parser.add_argument(
        "--others",
        type=str,
        default=[
            "fsdp_output/logs/step2_profile.sh-baseline-20260226-133107",
            "fsdp_output/logs/step2_profile.sh-int8_linear-20260226-133816",
            "fsdp_output/logs/step2_profile.sh-nc-20260227-043522",
            "fsdp_output/logs/step2_profile.sh-onebit_seide-20260226-134532",
            "fsdp_output/logs/step2_profile.sh-qsgd-20260227-044543",
            "fsdp_output/logs/step2_profile.sh-threshold_v-20260227-045317",
            "fsdp_output/logs/step2_profile.sh-sketch-20260227-050958",
            "fsdp_output/logs/step2_profile.sh-sketch_two_round-20260227-050041",
        ],
        help="One or more log_dirs to compare against the baseline.",
    )

    args = parser.parse_args()

    baseline_dir = Path(args.baseline).resolve()
    base_curve = _load_loss_curve(baseline_dir)

    print(f"Baseline: {baseline_dir}")
    print(f"Baseline steps: {len(base_curve)}")
    print()

    results: Dict[str, Dict[str, Any]] = {}

    for other_str in args.others:
        other_dir = Path(other_str).resolve()
        other_curve = _load_loss_curve(other_dir)

        steps, diffs = _aligned_diffs(base_curve, other_curve)
        mean_diff, var_diff = _mean_and_variance(diffs)

        name = other_dir.name
        results[name] = {
            "log_dir": str(other_dir),
            "num_aligned_steps": len(steps),
            "mean_diff": mean_diff,
            "var_diff": var_diff,
        }

        print(f"=== Compare with: {other_dir} ===")
        print(f"Aligned steps: {len(steps)}")
        print(f"Mean(loss_other - loss_baseline): {mean_diff:.6f}")
        print(f"Var(loss_other - loss_baseline):  {var_diff:.6f}")
        print()


if __name__ == "__main__":
    main()

