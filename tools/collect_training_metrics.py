#!/usr/bin/env python3
"""
从一次训练的 log_dir 收集指标：
- 训练 loss 曲线、GPU 显存曲线（从 TensorBoard events 读取）
- 吞吐（tokens/s，由步数、config 与 events 时间戳计算）
- 可选：训练结束后计算验证集 PPL
  - 支持显式提供验证集 JSON
  - 或基于同源 Wikipedia 数据自动划分 5% held-out 子集做验证（仅依赖 data_path）

用法:
  python tools/collect_training_metrics.py --log_dir /path/to/fsdp_output/logs/run-xxx
  # 只收集曲线与吞吐

  python tools/collect_training_metrics.py --log_dir ... --val_data /path/to/val.json
  # 使用显式验证集 JSON 计算 PPL

  python tools/collect_training_metrics.py --log_dir ... --eval_wiki_5pct
  # 使用同源 Wikipedia（config.data_path）自动划分 5% held-out 子集计算 PPL（无需单独 val.json）
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# 项目根目录
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def load_config(log_dir: Path) -> dict:
    config_path = log_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"config.json 不存在: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def collect_tb_scalars(tb_dir: Path) -> dict:
    """从 log_dir/tensorboard 读取 events，返回 Loss/step、Memory 等曲线。"""
    if not tb_dir.is_dir():
        return {}
    event_files = list(tb_dir.glob("events.out.tfevents.*"))
    if not event_files:
        return {}

    acc = EventAccumulator(str(tb_dir))
    acc.Reload()

    out = {}
    for tag in ("Loss/step", "LearningRate/step", "Memory/Allocated_GB", "Memory/Reserved_GB"):
        try:
            events = acc.Scalars(tag)
        except KeyError:
            continue
        # 每个元素: (step, value, wall_time)
        out[tag] = [{"step": e.step, "value": e.value, "wall_time": e.wall_time} for e in events]
    return out


def compute_throughput(scalars: dict, config: dict) -> dict | None:
    """根据 Loss/step 的 step 与 wall_time 以及 config 计算 tokens/s。"""
    loss_curve = scalars.get("Loss/step")
    if not loss_curve:
        return None
    steps = [x["step"] for x in loss_curve]
    times = [x["wall_time"] for x in loss_curve]
    if len(steps) < 2:
        return None
    num_steps = max(steps) - min(steps) + 1
    total_time = max(times) - min(times)
    if total_time <= 0:
        return None

    batch_size = int(config.get("batch_size", 1))
    max_length = int(config.get("max_length", 512))
    nproc = int(config.get("nproc", 1))
    # 每步全局 token 数（所有 rank）
    tokens_per_step = batch_size * max_length * nproc
    total_tokens = num_steps * tokens_per_step
    tokens_per_sec = total_tokens / total_time

    return {
        "num_steps": num_steps,
        "wall_time_sec": round(total_time, 2),
        "total_tokens": total_tokens,
        "tokens_per_sec": round(tokens_per_sec, 2),
    }


def _eval_ppl_on_dataset(model, tokenizer, dataset) -> dict:
    """在给定 dataset 上计算 token-level PPL。"""
    import torch
    from torch.utils.data import DataLoader
    from transformers import DataCollatorForLanguageModeling

    device = next(model.parameters()).device
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=collator)

    total_nll = 0.0
    total_tokens = 0
    criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            labels = batch.get("labels")
            if labels is None:
                labels = input_ids.clone()
                labels[labels == tokenizer.pad_token_id] = -100
            else:
                labels = labels.to(device)

            logits = model(input_ids).logits
            # (B, L, V) -> (B*L, V), labels (B, L)
            B, L, V = logits.shape
            logits_flat = logits.view(-1, V)
            labels_flat = labels.view(-1)
            mask = labels_flat != -100
            if mask.sum() == 0:
                continue
            nll = criterion(logits_flat[mask], labels_flat[mask])
            total_nll += nll.item()
            total_tokens += mask.sum().item()

    if total_tokens == 0:
        return {"val_ppl": None, "val_mean_nll": None, "val_tokens": 0, "error": "no valid tokens"}

    mean_nll = total_nll / total_tokens
    ppl = float(torch.exp(torch.tensor(mean_nll)).item())
    return {
        "val_ppl": round(ppl, 4),
        "val_mean_nll": round(mean_nll, 4),
        "val_tokens": total_tokens,
    }


def compute_val_ppl(
    checkpoint_dir: Path,
    val_data_path: Path,
    config: dict,
    max_samples: int = 0,
) -> dict:
    """加载 checkpoint 与显式验证集，计算 token 级平均 NLL，PPL = exp(mean NLL)。"""
    import torch
    from create_model import load_model, load_tokenizer
    from data_base import WikipediaDataset

    model_path = checkpoint_dir / "pytorch_model.bin"
    if not model_path.exists():
        raise FileNotFoundError(f"checkpoint 权重不存在: {model_path}")

    tokenizer = load_tokenizer()
    model_size = config.get("model_size", "small")
    model = load_model(tokenizer, model_size=model_size)
    state = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    max_length = int(config.get("max_length", 512))
    dataset = WikipediaDataset(
        str(val_data_path),
        tokenizer,
        max_length=max_length,
        max_samples=max_samples,
    )
    return _eval_ppl_on_dataset(model, tokenizer, dataset)


def compute_val_ppl_same_wiki_5pct(
    checkpoint_dir: Path,
    config: dict,
) -> dict:
    """基于 config.data_path 的同源 Wikipedia 数据，自动划分 5% held-out 子集做验证。"""
    import torch
    from torch.utils.data import Subset
    from create_model import load_model, load_tokenizer
    from data_base import WikipediaDataset

    model_path = checkpoint_dir / "pytorch_model.bin"
    if not model_path.exists():
        raise FileNotFoundError(f"checkpoint 权重不存在: {model_path}")

    data_path = config.get("data_path")
    if not data_path:
        return {"val_ppl": None, "val_mean_nll": None, "val_tokens": 0, "error": "config.data_path 缺失，无法构建同源验证集"}

    tokenizer = load_tokenizer()
    model_size = config.get("model_size", "small")
    model = load_model(tokenizer, model_size=model_size)
    state = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    max_length = int(config.get("max_length", 512))
    dataset = WikipediaDataset(
        str(data_path),
        tokenizer,
        max_length=max_length,
        max_samples=0,
    )
    n = len(dataset)
    if n == 0:
        return {"val_ppl": None, "val_mean_nll": None, "val_tokens": 0, "error": "同源 Wikipedia 数据集大小为 0"}

    k = max(1, int(n * 0.05))
    start = max(0, n - k)
    indices = list(range(start, n))
    heldout = Subset(dataset, indices)

    return _eval_ppl_on_dataset(model, tokenizer, heldout)


def main() -> None:
    parser = argparse.ArgumentParser(description="收集 step1 训练曲线、吞吐与可选验证集 PPL")
    parser.add_argument("--log_dir", type=str, required=True, help="单次运行的 log 目录，如 fsdp_output/logs/step1-xxx")
    parser.add_argument("--val_data", type=str, default=None, help="验证集 JSON 路径；若提供则在训练结束后计算验证集 PPL")
    parser.add_argument(
        "--eval_wiki_5pct",
        action="store_true",
        help="在与训练同源的 Wikipedia 数据（config.data_path）上自动划分 5% held-out 子集计算验证 PPL",
    )
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="checkpoint 目录（含 pytorch_model.bin）；默认用 output_dir/final_model")
    parser.add_argument("--output", type=str, default=None, help="输出 JSON 路径；默认 log_dir/training_metrics.json")
    parser.add_argument("--val_max_samples", type=int, default=0, help="验证集最多用多少条样本（0=全部）")
    args = parser.parse_args()

    log_dir = Path(args.log_dir).resolve()
    if not log_dir.is_dir():
        print(f"错误: log_dir 不存在或不是目录: {log_dir}", file=sys.stderr)
        sys.exit(1)

    config = load_config(log_dir)
    tb_dir = log_dir / "tensorboard"
    scalars = collect_tb_scalars(tb_dir)

    metrics = {
        "log_dir": str(log_dir),
        "run_name": config.get("run_name", ""),
        "loss_curve": scalars.get("Loss/step", []),
        "memory_allocated_gb": scalars.get("Memory/Allocated_GB", []),
        "memory_reserved_gb": scalars.get("Memory/Reserved_GB", []),
        "learning_rate_curve": scalars.get("LearningRate/step", []),
    }

    throughput = compute_throughput(scalars, config)
    if throughput:
        metrics["throughput"] = throughput

    if args.eval_wiki_5pct:
        output_dir = config.get("output_dir", ".")
        ckpt_dir = Path(output_dir) / "final_model" if not args.checkpoint_dir else Path(args.checkpoint_dir).resolve()
        try:
            metrics["validation"] = compute_val_ppl_same_wiki_5pct(ckpt_dir, config)
        except Exception as e:
            metrics["validation"] = {"error": str(e)}
    elif args.val_data:
        val_path = Path(args.val_data).resolve()
        if not val_path.exists():
            print(f"警告: 验证集不存在，跳过 PPL: {val_path}", file=sys.stderr)
        else:
            ckpt_dir = args.checkpoint_dir
            if not ckpt_dir:
                output_dir = config.get("output_dir", ".")
                ckpt_dir = Path(output_dir) / "final_model"
            else:
                ckpt_dir = Path(ckpt_dir).resolve()
            try:
                metrics["validation"] = compute_val_ppl(
                    ckpt_dir, val_path, config, max_samples=args.val_max_samples
                )
            except Exception as e:
                metrics["validation"] = {"error": str(e)}

    out_path = Path(args.output).resolve() if args.output else log_dir / "training_metrics.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"已写入: {out_path}")


if __name__ == "__main__":
    main()
