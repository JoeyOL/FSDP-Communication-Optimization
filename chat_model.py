import argparse
from pathlib import Path

import torch

from create_model import load_model
from logger import logger


def build_prompt(user_text: str) -> str:
    """极简对话格式：你可以后续替换成更严谨的 chat template。"""
    return f"用户：{user_text}\n助手："


@torch.no_grad()
def generate_reply(
    model,
    tokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int = 128,
    temperature: float = 0.8,
    top_p: float = 0.95,
    do_sample: bool = True,
) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    output_ids = model.generate(**inputs, **gen_kwargs)
    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # 只截取新生成的部分（尽量稳健）
    if text.startswith(prompt):
        return text[len(prompt) :].strip()
    return text.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="从训练保存的模型目录加载并进行交互式对话")
    parser.add_argument(
        "--model_dir",
        type=str,
        help="训练输出的 final_model 目录；需要包含 pytorch_model.bin 和 tokenizer 文件",
        default="./fsdp_output/final_model",
    )
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument(
        "--no_sample",
        action="store_true",
        help="关闭采样，使用贪心解码（输出更稳定但更死板）",
    )
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    weights_path = model_dir / "pytorch_model.bin"
    if not weights_path.exists():
        raise FileNotFoundError(f"未找到权重文件: {weights_path}")

    # 从保存目录加载 tokenizer（final_model 里是 GPT2 tokenizer 的文件集合）
    from transformers import GPT2Tokenizer

    tokenizer = GPT2Tokenizer.from_pretrained(str(model_dir))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")

    model = load_model(tokenizer)
    state_dict = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    logger.info("进入对话模式：输入内容后回车；输入 /exit 退出。")

    while True:
        try:
            user_text = input("你> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_text:
            continue
        if user_text in {"/exit", "/quit"}:
            break

        prompt = user_text
        reply = generate_reply(
            model,
            tokenizer,
            prompt,
            device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=not args.no_sample,
        )
        print(f"助手> {reply}")


if __name__ == "__main__":
    main()
