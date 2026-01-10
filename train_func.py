import torch
from logger import logger
from tqdm import tqdm
from pathlib import Path
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from torch.profiler import profile, record_function, ProfilerActivity
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist


# --- 新增带监控的训练函数 ---
def train_epoch_with_monitoring(model, dataloader, optimizer, scheduler, epoch, rank, world_size, args):
    """训练一个epoch，并使用Profiler和TensorBoard进行监控"""
    model.train()
    scaler = ShardedGradScaler()
    total_loss = 0.0
    num_batches = len(dataloader)
    
    # --- TensorBoard 和 Profiler 设置 (仅在rank 0上执行) ---
    tb_writer = None
    prof = None
    if rank == 0:
        # 定义日志目录
        log_dir = Path(args.output_dir) / "logs" / args.run_name
        tb_log_dir = log_dir / "tensorboard"
        profiler_log_dir = log_dir / "profiler"
        tb_log_dir.mkdir(parents=True, exist_ok=True)
        profiler_log_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化 TensorBoard Writer
        tb_writer = SummaryWriter(log_dir=str(tb_log_dir))
        
        # 配置 Profiler
        prof = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            on_trace_ready=torch.profiler.tensorboard_trace_handler(str(profiler_log_dir)),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            with_flops=True,
            with_modules=True
        )
        prof.start()
        logger.info(f"📊 TensorBoard 日志已启动，目录: {tb_log_dir}")
        logger.info(f"⏱️ Profiler 已启动，追踪文件将保存至: {profiler_log_dir}")
    
    dist.barrier()  # 确保所有进程都完成初始化

    optimizer.zero_grad()
    
    # 使用 disable 参数，更简洁地控制进度条只在 rank 0 上显示
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}", disable=(rank != 0))
    
    for batch_idx, batch in enumerate(progress_bar):
        global_step = epoch * num_batches + batch_idx
        try:
            # 将数据移动到GPU
            batch = {k: v.to(f'cuda:{rank}', non_blocking=True) for k, v in batch.items()}
            
            with record_function("forward_pass"): # Profiler 记录
                outputs = model(**batch)
                loss = outputs.loss
                loss = loss / args.gradient_accumulation_steps
            
            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(f"检测到无效损失 (NaN/Inf)，跳过此步。")
                continue
            
            # 2. 反向传播
            with record_function("backward_pass"): # Profiler 记录
                scaler.scale(loss).backward()
            
            total_loss += loss.item() * args.gradient_accumulation_steps

            # 3. 梯度累积和更新
            if (batch_idx + 1) % args.gradient_accumulation_steps == 0 or (batch_idx + 1) == num_batches:
                with record_function("optimizer_step"): # Profiler 记录
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    scheduler.step()

            # --- TensorBoard 日志记录 (仅在rank 0) ---
            if rank == 0 and tb_writer:
                current_loss = loss.item() * args.gradient_accumulation_steps
                tb_writer.add_scalar('Loss/step', current_loss, global_step)
                tb_writer.add_scalar('LearningRate/step', scheduler.get_last_lr()[0], global_step)
                if batch_idx % 20 == 0: # 每20步记录一次内存
                    mem_alloc = torch.cuda.memory_allocated(rank) / 1024**3
                    mem_res = torch.cuda.memory_reserved(rank) / 1024**3
                    tb_writer.add_scalar('Memory/Allocated_GB', mem_alloc, global_step)
                    tb_writer.add_scalar('Memory/Reserved_GB', mem_res, global_step)

        except torch.cuda.OutOfMemoryError:
            logger.error(f"步骤 {batch_idx} 发生 CUDA OOM！")
            optimizer.zero_grad()
            torch.cuda.empty_cache()
            continue
        except Exception as e:
            logger.error(f"训练步骤 {batch_idx} 发生未知错误: {e}")
            continue
            
        if rank == 0:
            progress_bar.set_postfix({
            'loss': f'{loss.item() * args.gradient_accumulation_steps:.4f}',
            'avg_loss': f'{total_loss/(batch_idx+1):.4f}',
            'lr': f'{scheduler.get_last_lr()[0]:.2e}',
            'gpu_mem': f'{torch.cuda.memory_allocated(rank)/1024**3:.1f}GB'
            })

    # --- 训练结束后清理 ---
    if rank == 0:
        if prof:
            prof.stop()
            logger.info("⏱️ Profiler 已停止。")
        if tb_writer:
            avg_epoch_loss = total_loss / num_batches
            tb_writer.add_scalar('Loss/epoch', avg_epoch_loss, epoch)
            tb_writer.close()
            logger.info("📊 TensorBoard writer 已关闭。")
    
    dist.barrier()  # 确保所有进程都完成

    avg_loss = total_loss / num_batches
    return avg_loss
    

def train_epoch(model, dataloader, optimizer, scheduler, epoch, rank, world_size, args, save_checkpoint_fn=None):
    """训练一个epoch"""
    model.train()
    scaler = ShardedGradScaler()  # FSDP-compatible GradScaler
    total_loss = 0.0
    num_batches = len(dataloader)
    optimizer.zero_grad()  # 初始化梯度
    
    if rank == 0:
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
    else:
        progress_bar = dataloader
    
    for batch_idx, batch in enumerate(dataloader):
        try:
            # 将数据移动到GPU
            batch = {k: v.to(f'cuda:{rank}', non_blocking=True) for k, v in batch.items()}

            # 1. 前向传播 (在autocast下)
            with torch.cuda.amp.autocast(dtype=torch.bfloat16): # 推荐使用 bfloat16
                outputs = model(**batch)
                loss = outputs.loss
                # 对累积的损失进行缩放
                loss = loss / args.gradient_accumulation_steps
            
            # 检查损失是否有效
            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(f"检测到无效损失 (NaN/Inf) 在步骤 {batch_idx}，跳过此批次。")
                optimizer.zero_grad() # 清理掉可能存在的坏梯度
                continue
            
            # 2. 反向传播 (计算缩放后的梯度)
            scaler.scale(loss).backward()
            
            total_loss += loss.item() * args.gradient_accumulation_steps # 记录未缩放的损失

            # 3. 梯度累积和更新
            if (batch_idx + 1) % args.gradient_accumulation_steps == 0 or (batch_idx + 1) == num_batches:
                # 3.1 (可选但推荐) 梯度裁剪，在优化器步骤之前
                # 首先 unscale 梯度
                scaler.unscale_(optimizer)
                # 然后在原始梯度上进行裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                
        except torch.cuda.OutOfMemoryError:
            # 捕获OOM错误
            logger.error(f"步骤 {batch_idx} 发生 CUDA Out-of-Memory 错误！")
            
            # 1. 释放不再需要的变量
            # 在Python中，离开try块后，outputs和loss等变量会自动被回收，
            # 但显式删除可以更清晰地表达意图。
            try:
                del outputs
                del loss
            except NameError:
                # 如果在创建这些变量之前就OOM了，它们可能不存在
                pass

            # 2. 清理梯度和优化器状态
            optimizer.zero_grad()

            # 3. 强制PyTorch释放未使用的缓存显存 (关键步骤)
            torch.cuda.empty_cache()
            
            logger.warning("已释放显存缓存并跳过此批次。")
            continue # 继续下一个批次的训练

        except Exception as e:
            logger.error(f"训练步骤 {batch_idx} 发生未知错误: {e}")
            continue
            # 更新进度条
        if rank == 0:
            progress_bar.set_postfix({
                'loss': f'{loss.item() * args.gradient_accumulation_steps:.4f}',
                'avg_loss': f'{total_loss/(batch_idx+1):.4f}',
                'lr': f'{scheduler.get_last_lr()[0]:.2e}',
                'gpu_mem': f'{torch.cuda.memory_allocated(rank)/1024**3:.1f}GB'
            })
            progress_bar.update(1)
    
    avg_loss = total_loss / num_batches
    return avg_loss