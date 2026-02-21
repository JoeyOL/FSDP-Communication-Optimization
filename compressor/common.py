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


def _ensure_residual(
    state: Any,
    full_flat_grad: torch.Tensor,
    group: Optional[dist.ProcessGroup] = None,
) -> tuple[torch.Tensor, Optional[int], Optional[int]]:
    """Return (residual, start, end) for current parameter. When ef_local: residual is shard-sized,
    (start, end) is this rank's slice; then hook should do g[start:end] += residual (no all_gather).
    When full EF: residual is full-sized, (start, end) is (None, None); hook does g += residual.
    """
    group = group or dist.group.WORLD
    numel = full_flat_grad.numel()
    world_size = dist.get_world_size(group)
    rank = dist.get_rank(group)
    shard_size = numel // world_size
    ef_local = getattr(state, "_ef_local", False)

    first_numel = getattr(state, "_ef_first_numel", None)
    if first_numel is None:
        state._ef_first_numel = numel
        state._ef_index = 0
        state._ef_residual_list = []
    elif numel == first_numel and getattr(state, "_ef_index", 0) > 0:
        state._ef_index = 0  # new backward, same first param
    idx = state._ef_index
    while idx >= len(state._ef_residual_list):
        state._ef_residual_list.append(None)

    if ef_local and shard_size > 0:
        # Local EF: residual is shard-sized; no all_gather later.
        res_size = shard_size
        start = rank * shard_size
        end = start + shard_size
        state._ef_shard_start = start
        state._ef_shard_end = end
    else:
        res_size = numel
        state._ef_shard_start = state._ef_shard_end = None

    r = state._ef_residual_list[idx]
    if r is None or r.numel() != res_size:
        state._ef_residual_list[idx] = torch.zeros(
            res_size, device=full_flat_grad.device, dtype=full_flat_grad.dtype
        )
        r = state._ef_residual_list[idx]
    state._ef_index = idx + 1

    if ef_local and shard_size > 0:
        return r, start, end
    return r, None, None


_all_gather_shard_log_count = 0

def _all_gather_shard(shard: torch.Tensor, group: Optional[dist.ProcessGroup] = None) -> torch.Tensor:
    """All-gather shard from all ranks into full tensor."""
    global _all_gather_shard_log_count
    if group is None:
        group = dist.group.WORLD
    world_size = dist.get_world_size(group)
    r = dist.get_rank(group)
    _all_gather_shard_log_count += 1
    do_log = _all_gather_shard_log_count <= 2
    if do_log:
        logger.info("[common] rank=%s _all_gather_shard before collective (call#%s)", r, _all_gather_shard_log_count)
    full_size = list(shard.shape)
    full_size[0] *= world_size
    full = torch.empty(full_size, device=shard.device, dtype=shard.dtype)
    if hasattr(dist, "all_gather_into_tensor"):
        dist.all_gather_into_tensor(full, shard, group=group)
    else:
        chunks = list(full.chunk(world_size, dim=0))
        dist.all_gather(chunks, shard, group=group)
    if do_log:
        logger.info("[common] rank=%s _all_gather_shard after collective", r)
    return full


def _apply_error_feedback(
    state: Any,
    full_flat_grad: torch.Tensor,
    shard_out: torch.Tensor,
    to_compress: torch.Tensor,
    group: Optional[dist.ProcessGroup] = None,
) -> None:
    """Update EF residual. When ef_local: residual = to_compress[my_shard] - shard_out (no all_gather).
    When full EF: full_reconstructed = all_gather(shard_out), residual = to_compress - full_reconstructed.
    """
    idx = getattr(state, "_ef_index", 1) - 1
    if idx < 0 or not getattr(state, "_ef_residual_list", None) or idx >= len(state._ef_residual_list):
        return
    residual = state._ef_residual_list[idx]
    if getattr(state, "_ef_local", False):
        start = getattr(state, "_ef_shard_start", 0)
        end = getattr(state, "_ef_shard_end", 0)
        if residual.numel() == shard_out.numel() and end <= to_compress.numel():
            residual.copy_(to_compress[start:end] - shard_out)
        return
    full_reconstructed = _all_gather_shard(shard_out, group=group)
    if residual.numel() == to_compress.numel():
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

