import os
import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.utils.data import DataLoader, DistributedSampler
from transformers import (
    get_linear_schedule_with_warmup,
    DataCollatorForLanguageModeling
)
from pathlib import Path
import argparse
import functools
import torch.distributed as dist
import random
import numpy as np
from transformers import (
    DataCollatorForLanguageModeling
)
from transformers.models.gpt2.modeling_gpt2 import GPT2Block
from create_model import load_model, load_tokenizer
from data_base import WikipediaDataset
from train_func import train_epoch_with_monitoring
from logger import logger
from comm_compress import list_methods, make_comm_hook
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_state_dict,
)


def set_seed(seed: int) -> None:
    """尽量保证可复现（注意：多 GPU/FSDP 仍可能存在非确定性算子）。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_distributed():
    """设置分布式训练环境"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        rank = 0
        world_size = 1
        local_rank = 0
    
    # 总是初始化进程组，即使是单GPU也需要（FSDP要求）
    if world_size > 1:
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
    else:
        # 单GPU环境下也需要初始化进程组
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        # 使用单机模式初始化进程组
        port = 12356
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = str(port)
        dist.init_process_group(backend='gloo', 
                              rank=0, world_size=1)
    
    return rank, world_size, local_rank

def main():
    parser = argparse.ArgumentParser(description='LLaMA-7B FSDP 训练')
    parser.add_argument('--model_path', type=str, default='/root/llama-7b', help='模型路径')
    parser.add_argument('--data_path', type=str, default='/root/llama-7b/datasets/wikipedia_en_10mb.json', help='数据集路径')
    parser.add_argument('--output_dir', type=str, default='/root/llama-7b/fsdp_output', help='输出目录')
    parser.add_argument('--batch_size', type=int, default=2, help='批量大小')
    parser.add_argument('--learning_rate', type=float, default=1e-6, help='学习率')
    parser.add_argument('--num_epochs', type=int, default=3, help='训练轮数')
    parser.add_argument('--max_length', type=int, default=512, help='最大序列长度')
    parser.add_argument('--warmup_steps', type=int, default=100, help='预热步数')
    parser.add_argument('--save_steps', type=int, default=500, help='保存间隔')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='梯度累积步数')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='权重衰减')
    parser.add_argument('--log_interval', type=int, default=10, help='日志输出间隔')
    parser.add_argument('--eval_steps', type=int, default=None, help='评估间隔步数')
    parser.add_argument('--dataloader_num_workers', type=int, default=2, help='数据加载器worker数量')
    parser.add_argument('--run_name', type=str, default='llama7b-fsdp-wiki', help='运行名称')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--dataset_shard_size', type=int, default=2000, help='预分词缓存分片大小（条数），用于大 JSON 文件')
    parser.add_argument('--dataset_max_samples', type=int, default=0, help='最多加载/预分词多少条样本（0表示全量），用于快速自检')

    # --- Step2：通信压缩模块化（FSDP comm hook） ---
    parser.add_argument(
        '--comm_compress',
        type=str,
        default='none',
        help='通信压缩方法：none/int8/...（可用方法见 comm_compress.list_methods()）',
    )
    parser.add_argument(
        '--comm_config_json',
        type=str,
        default='',
        help='压缩方法配置（JSON 对象字符串），例如：{"num_bits":8}',
    )
    
    args = parser.parse_args()
    
    # 设置分布式训练
    rank, world_size, local_rank = setup_distributed()

    # 可复现性（在初始化进程组后调用，保证各 rank 都设置）
    set_seed(args.seed + rank)
    
    logger.info(f"🎯 Rank {rank} 开始加载模型...")
    logger.info(f"模型路径: {args.model_path}")
    logger.info(f"当前设备: cuda:{local_rank}")
    
    tokenizer = load_tokenizer()
    
    model = load_model(tokenizer)
    model = model.to(f'cuda:{local_rank}')
    
    # 优化的 FSDP 配置 - 更激进的内存优化
    logger.info("创建FSDP包装...")
    # 优化的 FSDP 配置
    model = FSDP(model,
        device_id=local_rank,
        auto_wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={
                GPT2Block,
            }
        )
    )

    # --- Step2：按参数注册通信压缩 hook ---
    method = (args.comm_compress or 'none').strip().lower()
    if method not in ('none', 'off', 'disable', 'disabled'):
        if world_size <= 1:
            if rank == 0:
                logger.warning(f"已指定 --comm_compress={method}，但当前 world_size=1，不会注册 comm hook。")
        else:
            if rank == 0:
                logger.info(f"🔧 注册通信压缩 hook: {method}")
                logger.info(f"可用方法: {list(list_methods().keys())}")
            state, hook = make_comm_hook(method, args.comm_config_json or None)
            if hook is not None:
                model.register_comm_hook(state, hook)
                if rank == 0:
                    logger.info("✅ 通信压缩 hook 注册成功")
    
    logger.info(f"✅ Rank {rank} 模型加载完成，参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 创建输出目录
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # 加载数据集
    logger.info("加载数据集...")
    dataset = WikipediaDataset(
        args.data_path,
        tokenizer,
        args.max_length,
        shard_size=args.dataset_shard_size,
        max_samples=args.dataset_max_samples,
    )
    
    # 创建数据加载器 - 减少内存使用
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank) if world_size > 1 else None
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        collate_fn=data_collator,
        num_workers=args.dataloader_num_workers,
        pin_memory=True
    )


    # 创建优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    # 创建带预热的学习率调度器 (关键！)
    total_steps = (len(dataloader) // args.gradient_accumulation_steps) * args.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps
    )
    logger.info(f"总训练步数: {total_steps}, 预热步数: {args.warmup_steps}")
    logger.info(f"总训练步数: {total_steps}")
    logger.info(f"每个epoch步数: {len(dataloader)}")
        
    # 训练循环
    for epoch in range(args.num_epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        
        # 训练一个epoch
        avg_loss = train_epoch_with_monitoring(
            model, dataloader, optimizer, scheduler, epoch + 1, rank, world_size, args, 
        )
        
        if rank == 0:
            logger.info(f"Epoch {epoch + 1}/{args.num_epochs}, 平均损失: {avg_loss:.4f}")
    
    logger.info(f"Rank {rank} 正在参与收集状态字典...")
    # torch==2.5.1 的 StateDictOptions 不支持 rank0_only。
    # 用 broadcast_from_rank0：先由 rank0 收集完整 state_dict，再广播到其他 rank，
    # 同时启用 cpu_offload 将 state_dict 放到 CPU，降低 GPU 峰值显存。
    # 注意：get_state_dict 内部包含集体通信，必须所有 rank 都执行到这里。
    options = StateDictOptions(
        full_state_dict=True,
        cpu_offload=True,
        broadcast_from_rank0=True,
    )
    full_state_dict = get_state_dict(model, optimizer, options=options)
    
    if rank == 0:
        logger.info("训练完成! Rank 0 开始保存模型...")
        
        # 保存最终模型
        final_dir = Path(args.output_dir) / "final_model"
        final_dir.mkdir(parents=True, exist_ok=True)
        
        # 从返回值中提取模型状态（不同 torch 版本返回结构可能不同）
        # - 可能是 dict: {"model": ..., "optimizer": ...}
        # - 也可能是 tuple: (model_state_dict, optim_state_dict)
        if isinstance(full_state_dict, dict):
            model_state_dict = full_state_dict["model"]
        elif isinstance(full_state_dict, tuple) and len(full_state_dict) >= 1:
            model_state_dict = full_state_dict[0]
        else:
            raise TypeError(
                f"get_state_dict 返回了不支持的类型: {type(full_state_dict)}"
            )
        logger.info("状态字典在 Rank 0 上收集完成。")
        
        # 保存模型权重
        torch.save(model_state_dict, final_dir / "pytorch_model.bin")
        tokenizer.save_pretrained(final_dir)

    # 防止 rank0 保存时间较长导致其他 rank 提前退出，引发后续通信/销毁阶段异常
    if dist.is_initialized():
        dist.barrier()
    
    dist.barrier()
    
    # 清理分布式训练
    # 清理分布式训练
    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
