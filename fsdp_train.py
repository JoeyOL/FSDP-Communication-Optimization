import os
import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType
from torch.distributed.fsdp import ShardingStrategy
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
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_state_dict,
)
from compressor.comm_hooks import build_comm_hook


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
    parser.add_argument('--learning_rate', type=float, default=6e-5, help='学习率')
    parser.add_argument('--num_epochs', type=int, default=3, help='训练轮数')
    parser.add_argument('--max_length', type=int, default=512, help='最大序列长度')
    parser.add_argument('--warmup_steps', type=int, default=100, help='预热步数')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='梯度累积步数')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='权重衰减')
    parser.add_argument('--eval_steps', type=int, default=None, help='评估间隔步数')
    parser.add_argument('--dataloader_num_workers', type=int, default=2, help='数据加载器worker数量')
    parser.add_argument('--run_name', type=str, default='llama7b-fsdp-wiki', help='运行名称')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--model_size', type=str, default='small', choices=['small', 'medium', 'large', 'xl'], help='模型大小')
    parser.add_argument('--dataset_shard_size', type=int, default=2000, help='预分词缓存分片大小（条数），用于大 JSON 文件')
    parser.add_argument('--dataset_max_samples', type=int, default=0, help='最多加载/预分词多少条样本（0表示全量），用于快速自检')
    parser.add_argument(
        '--comm-hook',
        type=str,
        default='none',
        help='FSDP 通信压缩 hook: none, int8, fp16, qsgd, signsgd, onebit, onebit_seide, nc, topk, randomk, thresholdv, sketch, hybrid_topk_int8',
    )
    parser.add_argument('--comm-onebit-col-size', type=int, default=256, help='1-bit Seide 按列重建时的列大小')
    parser.add_argument('--comm-int8-variant', type=str, default='linear', choices=['linear', 'dynamic_tree'], help='INT8 变体: linear 或 dynamic_tree')
    parser.add_argument('--comm-signsgd-use-delta', action='store_true', help='SignSGD 使用 δ（学习率）作为步长，仅传方向')
    parser.add_argument(
        '--comm-error-feedback',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='是否启用误差反馈（与量化/稀疏化结合）',
    )
    parser.add_argument(
        '--comm-sparse-comm',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='稀疏算法(topk/randomk/thresholdv)是否用稀疏通信(只传 indices+values)，默认开启',
    )
    parser.add_argument('--comm-qsgd-s', type=int, default=4, help='QSGD 量化水平数 s')
    parser.add_argument('--comm-qsgd-bucket-size', type=int, default=0, help='QSGD 桶大小（0=整向量一桶，如512按桶独立scale）')
    parser.add_argument('--comm-topk-ratio', type=float, default=0.01, help='Top-k/Random-k/Threshold 稀疏比例（k=ratio*numel）')
    parser.add_argument('--comm-threshold-v', type=float, default=0.0, help='Threshold-v 正侧阈值 v_pos（0 则用 ratio 分位数）')
    parser.add_argument('--comm-threshold-v-neg', type=float, default=None, help='Threshold-v 负侧阈值 v_neg（缺省与 v 对称）')
    parser.add_argument('--comm-sketch-two-round', action='store_true', help='Sketched-SGD 两轮通信 + HEAVYMIX')

    # --- Step1/取证：耗时 profiling 与短跑 ---
    parser.add_argument(
        '--profile',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='是否启用 torch.profiler（默认启用；可用 --no-profile 关闭）',
    )
    parser.add_argument(
        '--profile_step_time',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='是否额外统计每 step 的 wall time（默认关闭）',
    )
    parser.add_argument(
        '--max_steps',
        type=int,
        default=0,
        help='最多训练多少个 step（跨 epoch 计数；0 表示不限制）',
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
    
    model = load_model(tokenizer, model_size=args.model_size)
    model = model.to(f'cuda:{local_rank}')
    
    # 优化的 FSDP 配置 - 更激进的内存优化
    logger.info("创建FSDP包装...")
    # 优化的 FSDP 配置
    model = FSDP(model,
        device_id=local_rank,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        auto_wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={
                GPT2Block,
            }
        )
    )
    comm_state, comm_hook = build_comm_hook(
        args.comm_hook,
        error_feedback=getattr(args, 'comm_error_feedback', False),
        int8_variant=getattr(args, 'comm_int8_variant', 'linear'),
        sparse_comm=getattr(args, 'comm_sparse_comm', True),
        qsgd_s=getattr(args, 'comm_qsgd_s', 4),
        qsgd_bucket_size=getattr(args, 'comm_qsgd_bucket_size', 0),
        topk_ratio=getattr(args, 'comm_topk_ratio', 0.01),
        randomk_ratio=getattr(args, 'comm_topk_ratio', 0.01),
        threshold_ratio=getattr(args, 'comm_topk_ratio', 0.01),
        threshold_v=getattr(args, 'comm_threshold_v', 0.0),
        threshold_v_neg=getattr(args, 'comm_threshold_v_neg', None),
        sketch_ratio=getattr(args, 'comm_topk_ratio', 0.01),
        sketch_two_round=getattr(args, 'comm_sketch_two_round', False),
        onebit_col_size=getattr(args, 'comm_onebit_col_size', 256),
        signsgd_use_delta=getattr(args, 'comm_signsgd_use_delta', False),
    )
    if comm_hook is not None and world_size > 1:
        logger.info(f"🔧 注册通信压缩 hook: {args.comm_hook}")
        model.register_comm_hook(comm_state, comm_hook)
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
