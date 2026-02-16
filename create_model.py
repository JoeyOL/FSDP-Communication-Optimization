from transformers import (
    GPT2Config, GPT2LMHeadModel, GPT2Tokenizer
)
from transformers import LlamaTokenizer, LlamaForCausalLM
import torch

SAVE_PATH = "./models/gpt2"
LLAMA_PATH = "./models/llama-7b"
def load_tokenizer():
    tokenizer = GPT2Tokenizer.from_pretrained(SAVE_PATH)
    tokenizer.add_special_tokens({"eos_token": "</s>", "bos_token": "<s>",
                                "unk_token": "<unk>", "pad_token": "<pad>", "mask_token": "<mask>"})
    return tokenizer

GPT2_MODEL_SIZES = {
    "small": {"n_layer": 12, "n_embd": 768, "n_head": 12},
    "medium": {"n_layer": 24, "n_embd": 1024, "n_head": 16},
    "large": {"n_layer": 36, "n_embd": 1280, "n_head": 20},
    "xl": {"n_layer": 48, "n_embd": 1600, "n_head": 25},
}

def load_model(tokenizer, model_size="small"):
    vocab_size = len(tokenizer)
    
    if model_size not in GPT2_MODEL_SIZES:
        raise ValueError(f"Invalid model_size: {model_size}. Must be one of {list(GPT2_MODEL_SIZES.keys())}")
    
    size_config = GPT2_MODEL_SIZES[model_size]
    config = GPT2Config(
        vocab_size=vocab_size,
        n_layer=size_config["n_layer"],
        n_embd=size_config["n_embd"],
        n_head=size_config["n_head"],
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        torch_dtype='float32'
    )
    # 随机初始化的模型不是预训练的，所以不需要加载预训练的权重
    model = GPT2LMHeadModel(config)

    # Ensure embedding table matches tokenizer size.
    if model.config.vocab_size != vocab_size:
        model.resize_token_embeddings(vocab_size)

    print(f"模型参数的数据类型是: {next(model.parameters()).dtype}")
    return model

# def load_tokenizer():
#     # 从本地加载 LLaMA tokenizer
#     tokenizer = LlamaTokenizer.from_pretrained(LLAMA_PATH)
#     # LLaMA 已有这些特殊 token，可能不需要添加
#     if tokenizer.pad_token is None:
#         tokenizer.pad_token = tokenizer.eos_token
#         tokenizer.bos_token = tokenizer.eos_token
#         tokenizer.unk_token = tokenizer.eos_token
#     return tokenizer

# def load_model(tokenizer):
#     # 从本地加载 LLaMA-7B 模型，使用更激进的内存优化
#     model = LlamaForCausalLM.from_pretrained(
#         LLAMA_PATH,
#         torch_dtype=torch.bfloat16,  # 使用 bfloat16 替代 float16 更稳定
#         low_cpu_mem_usage=True,      # 降低 CPU 内存使用
#         use_cache=False         # 关闭缓存，节省内存
#     )
    
#     # 如果需要调整词汇表大小
#     if len(tokenizer) != model.config.vocab_size:
#         model.resize_token_embeddings(len(tokenizer))
    
#     return model