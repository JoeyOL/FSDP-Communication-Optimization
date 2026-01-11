import argparse
import json
import os
from pathlib import Path

from tqdm import tqdm


def bytes_to_mb(n: int) -> float:
    return n / 1024 / 1024


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "下载英文 Wikipedia 语料并导出为 JSON 数组格式（与本项目 datasets/*.json 兼容）"
        )
    )
    parser.add_argument(
        "--config",
        type=str,
        default="20220301.en",
        help=(
            "HuggingFace datasets 的 wikipedia 配置名，例如 20220301.en；"
            "不同版本可能可用配置略有差异"
        ),
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="数据集 split，通常为 train",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/root/llama-7b/datasets/wikipedia_en_custom.json",
        help="输出 JSON 文件路径（JSON 数组）",
    )
    parser.add_argument(
        "--max_mb",
        type=int,
        default=0,
        help="输出文件最大大小（MB，0 表示不限制；近似控制）",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help="最多导出多少条样本（0 表示不限制）",
    )
    parser.add_argument(
        "--min_chars",
        type=int,
        default=200,
        help="过滤过短样本：text 字段至少多少字符",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="保留参数位（目前按顺序导出，不做随机采样）",
    )

    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 延迟导入，避免没装 datasets 时影响其他脚本
    from datasets import load_dataset

    # 在一些离线/受限环境中，会无法访问 HuggingFace Hub。
    # 这里做更友好的错误提示，并允许你后续改用本地缓存/镜像。
    try:
        ds = load_dataset("wikipedia", args.config, split=args.split)
    except Exception as e:
        raise RuntimeError(
            "无法从 HuggingFace Hub 下载 'wikipedia' 数据集。\n"
            "可能原因：当前环境无法联网 / 没有配置 HF 缓存 / 需要代理。\n\n"
            "可选解决方案：\n"
            "1) 确认网络可访问，并设置环境变量 HF_ENDPOINT/HF_HOME（如需镜像/自定义缓存）。\n"
            "2) 在可联网机器上提前下载到本地缓存目录，再拷贝到训练机。\n"
            "3) 如果你已经有 wikipedia_en_*.json（本项目 datasets/ 目录），可以跳过下载脚本。\n\n"
            f"原始错误: {type(e).__name__}: {e}"
        ) from e

    max_bytes = args.max_mb * 1024 * 1024 if args.max_mb and args.max_mb > 0 else None
    max_samples = args.max_samples if args.max_samples and args.max_samples > 0 else None

    written = 0

    # 写成 JSON 数组：[
    #   {"id":...,"title":...,"text":...},
    #   ...
    # ]
    # 为避免内存爆炸，采用流式写出，而非把所有样本 accumulate 再 json.dump。
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("[\n")

        pbar = tqdm(total=max_samples, desc="导出 Wikipedia")
        first = True

        for item in ds:
            # wikipedia dataset 通常字段：id/title/text
            text = item.get("text", "") or ""
            if len(text) < args.min_chars:
                continue

            record = {
                "id": str(item.get("id", "")),
                "title": item.get("title", "") or "",
                "text": text,
            }

            s = json.dumps(record, ensure_ascii=False)
            if not first:
                f.write(",\n")
            f.write("  " + s)
            first = False

            written += 1

            # 近似大小控制：用文件当前位置判断
            if max_bytes is not None and f.tell() >= max_bytes:
                break
            if max_samples is not None and written >= max_samples:
                break

            if max_samples is not None:
                pbar.update(1)
            else:
                # 没有上限时也给点进度
                if written % 1000 == 0:
                    pbar.set_postfix({"written": written, "size_mb": f"{bytes_to_mb(f.tell()):.1f}"})

        pbar.close()
        f.write("\n]\n")

    size_mb = bytes_to_mb(os.path.getsize(out_path))
    print(f"写入完成: {out_path} samples={written} size={size_mb:.1f}MB")


if __name__ == "__main__":
    main()
