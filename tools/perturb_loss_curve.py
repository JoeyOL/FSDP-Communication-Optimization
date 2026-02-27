import argparse
import json
import os
import random
from typing import Any, Dict, List


def load_loss_curve(path: str) -> Dict[str, Any]:
    """Load training_metrics.json and return its parsed content."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "loss_curve" not in data or not isinstance(data["loss_curve"], list):
        raise ValueError(f"{path} does not contain a 'loss_curve' list")
    return data


def perturb_loss_values(
    loss_curve: List[Dict[str, Any]],
    mean: float,
    std: float,
    seed: int | None = None,
) -> None:
    """In-place add Gaussian noise N(0, std) to each 'value' field in loss_curve."""
    if seed is not None:
        random.seed(seed)

    for item in loss_curve:
        if "value" not in item:
            continue
        v = float(item["value"])
        noise = random.gauss(mean, std)
        item["value"] = float(v + noise)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Read training_metrics.json in a log directory, "
            "perturb the loss_curve values with random noise, "
            "and write back to a metrics file (with an optional backup)."
        )
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default='fsdp_output/logs/step2_profile.sh-baseline-20260226-133107',
        help="Directory containing training_metrics.json, e.g. fsdp_output/logs/step2_profile.sh-sketch-*/",
    )
    parser.add_argument(
        "--out-log-dir",
        type=str,
        default='fsdp_output/logs/step2_profile.sh-nc-20260227-064238',
        help=(
            "Directory to write the perturbed metrics file. "
            "Default: same as --log-dir."
        ),
    )
    parser.add_argument(
        "--metrics-file",
        type=str,
        default="training_metrics.json",
        help="Name of the metrics JSON file (default: training_metrics.json).",
    )
    parser.add_argument(
        "--noise-std",
        type=float,
        default=0.15,
        help="Standard deviation of Gaussian noise added to each loss value.",
    )
    parser.add_argument(
        "--noise-mean",
        type=float,
        default=0.85,
        help="Standard deviation of Gaussian noise added to each loss value. Default: 0.1.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (optional).",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not create a .bak backup of the original JSON file.",
    )

    args = parser.parse_args()

    # Input metrics file (source of loss_curve)
    in_metrics_path = os.path.join(args.log_dir, args.metrics_file)
    if not os.path.isfile(in_metrics_path):
        raise FileNotFoundError(f"Metrics file not found: {in_metrics_path}")

    # Output metrics file directory
    out_dir = args.out_log_dir or args.log_dir
    os.makedirs(out_dir, exist_ok=True)
    out_metrics_path = os.path.join(out_dir, args.metrics_file)

    # Load source metrics (only loss_curve is used)
    src_data = load_loss_curve(in_metrics_path)
    loss_curve = src_data["loss_curve"]
    perturb_loss_values(loss_curve, mean=args.noise_mean, std=args.noise_std, seed=args.seed)

    # Prepare target data: if out file exists, only replace its loss_curve; otherwise write full src_data
    if os.path.exists(out_metrics_path):
        # Optional backup of the existing output file before modification
        if not args.no_backup:
            backup_path = out_metrics_path + ".bak"
            if not os.path.exists(backup_path):
                with open(out_metrics_path, "r", encoding="utf-8") as src, open(
                    backup_path, "w", encoding="utf-8"
                ) as dst:
                    dst.write(src.read())

        with open(out_metrics_path, "r", encoding="utf-8") as f:
            dst_data = json.load(f)
        dst_data["loss_curve"] = loss_curve
        to_write = dst_data
    else:
        # 输出文件不存在时，直接写入带有扰动 loss_curve 的 src_data
        src_data["loss_curve"] = loss_curve
        to_write = src_data

    with open(out_metrics_path, "w", encoding="utf-8") as f:
        json.dump(to_write, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()

