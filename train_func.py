import json
from pathlib import Path

import torch
from logger import logger
from tqdm import tqdm
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from torch.profiler import record_function
import torch.distributed as dist

from perf.comm_profiler import (
    finalize_monitoring,
    init_monitoring,
    should_stop_early,
    step_begin,
    step_end,
)
from perf.comm_stats import snapshot as comm_snapshot, total_bytes as comm_total_bytes
from perf.grad_error_stats import snapshot as grad_error_snapshot


# ---------------------------------------------------------------------------
# P2 fix / documentation: ShardedGradScaler + communication hook interaction
# ---------------------------------------------------------------------------
# PyTorch FSDP communication hooks (registered via
# model.register_comm_hook(state, hook)) run INSIDE the backward pass.
# When ShardedGradScaler is used, it calls scaler.scale(loss).backward(),
# which means gradients seen by the hook are already multiplied by the
# scaler's current scale factor (typically a large power-of-2).
#
# Impact on compression algorithms:
#   - INT8 / FP16 / SignSGD / 1-bit Seide: These normalise or quantise
#     relative to max/norm, so uniform scaling does NOT change behaviour.
#   - QSGD / NC: Bucket norms and stochastic rounding are scale-invariant.
#   - TopK / RandomK / ThresholdV: Absolute threshold comparisons (v_pos,
#     v_neg) would be affected if set as raw values. However, ratio-based
#     selection (ratio=0.01) is unaffected since it operates on relative
#     ordering.
#   - Sketch: Count-sketch + heavy hitter detection operates on actual
#     magnitudes, but since all elements are uniformly scaled, top-k
#     selection is still correct.
#
# Conclusion: For ratio-based or relative compression, GradScaler is safe.
# If users specify absolute thresholds (e.g., ThresholdV with v > 0), they
# should be aware that the effective threshold in gradient space is
# threshold / scaler.get_scale().
#
# For users who want to disable mixed-precision scaling entirely (e.g., when
# using bfloat16 which doesn't need loss scaling), set use_grad_scaler=False
# via args.
# ---------------------------------------------------------------------------


def _create_grad_scaler(args) -> ShardedGradScaler | None:
    """Create ShardedGradScaler based on configuration.

    Returns None if gradient scaling is disabled (e.g., bfloat16 training
    where loss scaling is unnecessary and could interfere with absolute-
    threshold compression algorithms).
    """
    use_scaler = getattr(args, "use_grad_scaler", True)
    if not use_scaler:
        logger.info("[train] GradScaler disabled via args.use_grad_scaler=False")
        return None
    return ShardedGradScaler()


# --- 新增带监控的训练函数 ---
def train_epoch_with_monitoring(model, dataloader, optimizer, scheduler, epoch, rank, world_size, args):
    """训练一个epoch，并使用Profiler和TensorBoard进行监控"""
    model.train()
    scaler = _create_grad_scaler(args)
    total_loss = 0.0
    num_batches = len(dataloader)

    # --- TensorBoard 和 Profiler 设置 (仅在 rank 0 上执行；实现细节在 perf/ 下) ---
    monitor = init_monitoring(args, rank, num_batches)
    if rank == 0 and monitor.enabled:
        logger.info(f"📊 TensorBoard 日志已启动，目录: {monitor.tb_log_dir}")
        logger.info(f"⏱️ Profiler 已启动，追踪文件将保存至: {monitor.profiler_log_dir}")
    
    dist.barrier()  # 确保所有进程都完成初始化

    optimizer.zero_grad()
    
    # 使用 disable 参数，确保只有 rank0 打印进度条
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}", disable=(rank != 0), dynamic_ncols=True)
    
    step_count = 0

    for batch_idx, batch in enumerate(progress_bar):
        # epoch 传入为 1-based，这里换算为 0-based 以保证 max_steps 计数准确
        global_step = (epoch - 1) * num_batches + batch_idx
        step_t0 = step_begin(monitor, args)
        try:
            # 将数据移动到GPU
            # 默认约定：cuda 设备已在外部通过 torch.cuda.set_device(local_rank) 设置
            batch = {k: v.cuda(non_blocking=True) for k, v in batch.items()}
            
            with record_function("forward_pass"): # Profiler 记录
                outputs = model(**batch)
                loss = outputs.loss
                loss = loss / args.gradient_accumulation_steps
            
            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(f"检测到无效损失 (NaN/Inf)，跳过此步。")
                continue
            
            # 2. 反向传播
            with record_function("backward_pass"): # Profiler 记录
                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
            
            total_loss += loss.item() * args.gradient_accumulation_steps

            # 3. 梯度累积和更新
            if (batch_idx + 1) % args.gradient_accumulation_steps == 0 or (batch_idx + 1) == num_batches:
                with record_function("optimizer_step"): # Profiler 记录
                    if scaler is not None:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()

            # --- TensorBoard 日志记录 (仅在rank 0) ---
            if rank == 0 and monitor.tb_writer:
                current_loss = loss.item() * args.gradient_accumulation_steps
                monitor.tb_writer.add_scalar('Loss/step', current_loss, global_step)
                monitor.tb_writer.add_scalar('LearningRate/step', scheduler.get_last_lr()[0], global_step)
                if batch_idx % 20 == 0: # 每20步记录一次内存
                    mem_alloc = torch.cuda.memory_allocated(rank) / 1024**3
                    mem_res = torch.cuda.memory_reserved(rank) / 1024**3
                    monitor.tb_writer.add_scalar('Memory/Allocated_GB', mem_alloc, global_step)
                    monitor.tb_writer.add_scalar('Memory/Reserved_GB', mem_res, global_step)

        except torch.cuda.OutOfMemoryError:
            logger.error(f"步骤 {batch_idx} 发生 CUDA OOM！")
            optimizer.zero_grad()
            torch.cuda.empty_cache()
            continue
        except Exception as e:
            # 对于明确的数值错误（例如 NC 路径检测到的非有限值），直接中止本次训练，
            # 避免在后续 step 中重复触发相同错误。
            msg = str(e)
            if "[nc] detected non-finite" in msg:
                logger.error(f"训练步骤 {batch_idx} 检测到 NC 数值错误，终止当前训练: {e}")
                raise
            logger.error(f"训练步骤 {batch_idx} 发生未知错误: {e}")
            continue

        # 让 profiler schedule 前进 + （可选）采集 step wall time
        step_end(monitor, args, step_t0)
        step_count += 1

        # --- 可选：短跑，用于耗时取证 ---
        if should_stop_early(args, global_step):
            if rank == 0:
                logger.info(f"达到 max_steps={getattr(args, 'max_steps', 0)}，提前结束本轮训练。")
            break
            
        if rank == 0:
            progress_bar.set_postfix({
            'loss': f'{loss.item() * args.gradient_accumulation_steps:.4f}',
            'avg_loss': f'{total_loss/(batch_idx+1):.4f}',
            'lr': f'{scheduler.get_last_lr()[0]:.2e}',
            'gpu_mem': f'{torch.cuda.memory_allocated()/1024**3:.1f}GB'
            })

    # --- 训练结束后清理 ---
    if rank == 0:
        finalize_monitoring(monitor, args=args, epoch=epoch, total_loss=total_loss, num_batches=num_batches)
        if monitor.enabled:
            logger.info("⏱️ Profiler 已停止，并已写出摘要文件。")
            logger.info("📊 TensorBoard writer 已关闭。")
    
    dist.barrier()  # 确保所有进程都完成

    avg_loss = total_loss / num_batches

    # 在 rank0 上将轻量通信统计与梯度误差统计写入 log_dir，供离线分析使用。
    if rank == 0:
        try:
            hooks = comm_snapshot()
            total_bytes = int(comm_total_bytes())
            effective_steps = max(1, step_count)
            bytes_per_step = int(total_bytes / effective_steps)
            log_dir = Path(getattr(args, "output_dir", "/root/llama-7b/fsdp_output")) / "logs" / getattr(
                args, "run_name", "run"
            )
            log_dir.mkdir(parents=True, exist_ok=True)
            out = {
                "step_count": effective_steps,
                "total_bytes_per_rank": total_bytes,
                "bytes_per_step_per_rank": bytes_per_step,
                "by_hook": hooks,
            }
            (log_dir / "comm_stats_rank0.json").write_text(
                json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            logger.info(
                "[comm_stats] wrote runtime comm stats to %s (bytes_per_step_per_rank=%s)",
                log_dir / "comm_stats_rank0.json",
                bytes_per_step,
            )
        except Exception as e:
            logger.warning(f"[comm_stats] failed to write runtime comm stats: {e}")

        # 梯度压缩误差统计：当前仅在启用误差反馈的压缩 hook 中生效，
        # 通过 perf.grad_error_stats 在各个 hook 内部累积相对 L2 误差。
        try:
            grad_stats = grad_error_snapshot()
            if grad_stats:
                (log_dir / "grad_error_stats_rank0.json").write_text(
                    json.dumps({"by_hook": grad_stats}, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                logger.info(
                    "[grad_error_stats] wrote runtime grad error stats to %s",
                    log_dir / "grad_error_stats_rank0.json",
                )
        except Exception as e:
            logger.warning(f"[grad_error_stats] failed to write runtime grad error stats: {e}")

    return avg_loss
    

def train_epoch(model, dataloader, optimizer, scheduler, epoch, rank, world_size, args, save_checkpoint_fn=None):
    """训练一个epoch"""
    model.train()
    scaler = _create_grad_scaler(args)
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
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            total_loss += loss.item() * args.gradient_accumulation_steps # 记录未缩放的损失

            # 3. 梯度累积和更新
            if (batch_idx + 1) % args.gradient_accumulation_steps == 0 or (batch_idx + 1) == num_batches:
                if scaler is not None:
                    # 3.1 (可选但推荐) 梯度裁剪，在优化器步骤之前
                    # 首先 unscale 梯度
                    scaler.unscale_(optimizer)
                    # 然后在原始梯度上进行裁剪
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
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
