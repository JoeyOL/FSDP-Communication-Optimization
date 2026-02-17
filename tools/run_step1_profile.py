from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def _default_run_name(prefix: str) -> str:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{prefix}-{ts}"


def build_command(args: argparse.Namespace, passthrough: list[str]) -> list[str]:
    base_args = [
        "fsdp_train.py",
        "--data_path",
        args.data_path,
        "--output_dir",
        args.output_dir,
        "--run_name",
        args.run_name,
        "--num_epochs",
        str(args.num_epochs),
        "--batch_size",
        str(args.batch_size),
        "--max_length",
        str(args.max_length),
        "--dataset_max_samples",
        str(args.dataset_max_samples),
        "--max_steps",
        str(args.max_steps),
        "--profile",
        "--profile-step-time",
    ]

    base_args.extend(passthrough)

    if args.nproc <= 1:
        return [sys.executable, *base_args]

    # Use python -m torch.distributed.run for portability
    return [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={args.nproc}",
        *base_args,
    ]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Step1: 一键启动 FSDP 训练的耗时取证（profiler + step wall time + overlap）。\n"
            "产物会写到 output_dir/logs/<run_name>/profiler/ 下。"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data_path", required=True, help="Wikipedia JSON 数据路径")
    parser.add_argument("--output_dir", default="out", help="输出目录")
    parser.add_argument("--run_name", default="", help="运行名（空则自动生成）")
    parser.add_argument("--nproc", type=int, default=1, help="进程数/卡数；>1 时用 torch.distributed.run")

    # 默认给一个非常稳的短跑配置
    parser.add_argument("--max_steps", type=int, default=20, help="短跑步数（确保 profiler schedule 有效）")
    parser.add_argument("--num_epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--dataset_max_samples", type=int, default=200, help="最多加载样本数（加速取证）")
    parser.add_argument("--batch_size", type=int, default=2, help="batch size")
    parser.add_argument("--max_length", type=int, default=128, help="最大序列长度（更省显存）")

    # 允许透传 fsdp_train.py 的其它参数
    args, passthrough = parser.parse_known_args()

    if not args.run_name:
        args.run_name = _default_run_name("step1")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = build_command(args, passthrough)
    print("[RUN]", " ".join(cmd))

    result = subprocess.run(cmd)

    profiler_dir = Path(args.output_dir) / "logs" / args.run_name / "profiler"
    print("[OUT] profiler_dir =", str(profiler_dir))
    print("[OUT] summary =", str(profiler_dir / "summary_rank0.json"))
    print("[OUT] comm_csv =", str(profiler_dir / "comm_op_summary_rank0.csv"))

    return int(result.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
