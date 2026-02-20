"""
Common utilities for FSDP communication hooks: logger, error feedback, sparse all_gather helpers.
这些工具函数在不同压缩算法之间共享，避免循环依赖。
"""

from typing import Any, Optional

import torch
import torch.distributed as dist

try:
    from logger import logger  # 项目里的统一 logger
except ImportError:  # 直接跑单文件时的兜底
    import logging

    logger = logging.getLogger(__name__)


def _ensure_residual(state: Any, full_flat_grad: torch.Tensor) -> torch.Tensor:
    if getattr(state, "residual", None) is None:
        state.residual = torch.zeros_like(
            full_flat_grad, device=full_flat_grad.device, dtype=full_flat_grad.dtype
        )
    return state.residual


def _all_gather_shard(shard: torch.Tensor, group: Optional[dist.ProcessGroup] = None) -> torch.Tensor:
    """All-gather shard from all ranks into full tensor."""
    if group is None:
        group = dist.group.WORLD
    world_size = dist.get_world_size(group)
    full_size = list(shard.shape)
    full_size[0] *= world_size
    full = torch.empty(full_size, device=shard.device, dtype=shard.dtype)
    if hasattr(dist, "all_gather_into_tensor"):
        dist.all_gather_into_tensor(full, shard, group=group)
    else:
        chunks = list(full.chunk(world_size, dim=0))
        dist.all_gather(chunks, shard, group=group)
    return full


def _apply_error_feedback(
    state: Any,
    full_flat_grad: torch.Tensor,
    shard_out: torch.Tensor,
    to_compress: torch.Tensor,
    group: Optional[dist.ProcessGroup] = None,
) -> None:
    """在通信后重建 full，再更新 EF 残差: residual = to_compress - full."""
    full_reconstructed = _all_gather_shard(shard_out, group=group)
    residual = _ensure_residual(state, full_flat_grad)
    residual.copy_(to_compress - full_reconstructed)


def _sparse_all_gather_and_merge(
    indices: torch.Tensor,
    values: torch.Tensor,
    numel: int,
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    """
    各 rank 发送 (indices, values)，all_gather 后合并为完整梯度向量 sum_r sparse_r。
    indices/values 形状均为 (k,)；返回 full 形状 (numel,)，dtype/device 与 values 一致。
    """
    if group is None:
        group = dist.group.WORLD
    world_size = dist.get_world_size(group)
    k = indices.numel()
    device = values.device
    dtype = values.dtype

    indices_flat = indices.contiguous().view(-1)
    values_flat = values.contiguous().view(-1)

    if hasattr(dist, "all_gather_into_tensor"):
        indices_buf = torch.empty(world_size * k, device=device, dtype=indices.dtype)
        values_buf = torch.empty(world_size * k, device=device, dtype=dtype)
        dist.all_gather_into_tensor(indices_buf, indices_flat, group=group)
        dist.all_gather_into_tensor(values_buf, values_flat, group=group)
        indices_all = indices_buf.view(world_size, k)
        values_all = values_buf.view(world_size, k)
    else:
        indices_list = [torch.empty(k, device=device, dtype=indices.dtype) for _ in range(world_size)]
        values_list = [torch.empty(k, device=device, dtype=dtype) for _ in range(world_size)]
        dist.all_gather(indices_list, indices_flat, group=group)
        dist.all_gather(values_list, values_flat, group=group)
        indices_all = torch.stack(indices_list, dim=0)
        values_all = torch.stack(values_list, dim=0)

    full = torch.zeros(numel, device=device, dtype=dtype)
    for r in range(world_size):
        idx = indices_all[r].long()
        val = values_all[r]
        full.index_add_(0, idx, val)

    return full


__all__ = [
    "logger",
    "_ensure_residual",
    "_all_gather_shard",
    "_apply_error_feedback",
    "_sparse_all_gather_and_merge",
]

