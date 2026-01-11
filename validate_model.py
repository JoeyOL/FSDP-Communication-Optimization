import argparse
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from create_model import load_model
from data_base import WikipediaDataset
from logger import logger


@torch.no_grad()
def evaluate(model, dataloader, device: torch.device) -> tuple[float, float]:
    """返回 (avg_loss, ppl)"""
    model.eval()
    total_loss = 0.0
    total_batches = 0

    for batch in dataloader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        total_loss += float(loss.item())
        total_batches += 1

    avg_loss = total_loss / max(1, total_batches)
    ppl = math.exp(avg_loss) if avg_loss < 20 else float("inf")
    return avg_loss, ppl


def main() -> None:
    parser = argparse.ArgumentParser(description="加载训练保存的模型并做验证")
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="训练输出的 final_model 目录（包含 pytorch_model.bin 和 tokenizer files）",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="/root/llama-7b/datasets/wikipedia_en_10mb.json",
        help="验证用数据集路径（Wikipedia JSON）",
    )
    parser.add_argument("--max_length", type=int, default=128, help="序列最大长度")
    parser.add_argument("--batch_size", type=int, default=2, help="验证 batch size")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    weights_path = model_dir / "pytorch_model.bin"
    if not weights_path.exists():
        raise FileNotFoundError(f"未找到权重文件: {weights_path}")

    # tokenizer：训练时 save_pretrained(final_dir) 已写入。
    # 但本项目的 create_model.load_tokenizer() 目前从 ./models/gpt2 读取，
    # 为了保证“加载保存产物”闭环，这里优先从 model_dir 加载。
    from transformers import GPT2Tokenizer

    tokenizer = GPT2Tokenizer.from_pretrained(str(model_dir))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")

    model = load_model(tokenizer)
    state_dict = torch.load(weights_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        logger.warning(f"missing keys: {len(missing)} (示例: {missing[:5]})")
    if unexpected:
        logger.warning(f"unexpected keys: {len(unexpected)} (示例: {unexpected[:5]})")

    model.to(device)

    dataset = WikipediaDataset(args.data_path, tokenizer, args.max_length)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    avg_loss, ppl = evaluate(model, dataloader, device)
    logger.info(f"验证完成: avg_loss={avg_loss:.4f}, ppl={ppl:.2f}")


if __name__ == "__main__":
    main()
