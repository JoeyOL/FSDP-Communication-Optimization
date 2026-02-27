from typing import Tuple

import torch
import torch.distributed as dist

from .common import _ensure_residual, logger
from perf.comm_stats import add_bytes as _comm_add_bytes

try:
    # C++ 实现的 1-bit Seide 辅助算子（如可用则优先使用）
    from .onebit_seide_ops import (
        onebit_seide_per_column as cpp_onebit_seide_per_column,
        onebit_seide_reconstruct_from_gathered as cpp_onebit_seide_reconstruct,
    )

    _HAS_ONEBIT_CPP = True
except Exception:
    _HAS_ONEBIT_CPP = False

# 运行时探测：如果加载/调用 C++ 扩展失败，则退回 Python 实现，且不再重试。
_ONEBIT_CPP_RUNTIME_AVAILABLE = True


def _pack_signs_to_bytes(signs: torch.Tensor) -> torch.Tensor:
    """Pack signs (1 = non-negative, 0 = negative) into uint8, 8 per byte, LSB first."""
    flat = (signs >= 0).to(torch.uint8).view(-1)
    n = flat.numel()
    n_pad = (8 - n % 8) % 8
    if n_pad:
        flat = torch.cat([flat, torch.zeros(n_pad, device=flat.device, dtype=torch.uint8)])
    powers = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128], device=flat.device, dtype=torch.uint8)
    return (flat.view(-1, 8) * powers).sum(dim=1).to(torch.uint8)


def _unpack_signs_from_bytes_all(packed_all: torch.Tensor, numel: int) -> torch.Tensor:
    """Vectorized unpack for all ranks at once.

    packed_all: (world_size, packed_bytes) uint8
    return: (world_size, numel) float32 in {+1, -1}
    """
    world_size, n_bytes = packed_all.shape
    n_bits = min(numel, n_bytes * 8)
    if n_bits == 0:
        return torch.zeros(world_size, numel, device=packed_all.device, dtype=torch.float32)

    packed_int = packed_all.to(torch.int32)  # (world_size, n_bytes)
    bit_pos = torch.arange(8, device=packed_all.device, dtype=torch.int32)  # (8,)
    # (world_size, n_bytes, 8)
    bits_expanded = (packed_int.unsqueeze(-1) >> bit_pos.unsqueeze(0).unsqueeze(0)) & 1
    bits_flat = bits_expanded.view(world_size, -1)[:, :n_bits]  # (world_size, n_bits)

    out = torch.zeros(world_size, numel, device=packed_all.device, dtype=torch.float32)
    out[:, :n_bits] = torch.where(bits_flat == 1, 1.0, -1.0)
    return out


def _onebit_seide_per_column(g: torch.Tensor, col_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """For each column: signs (1/-1), a=mean(positive), b=mean(negative)."""
    global _ONEBIT_CPP_RUNTIME_AVAILABLE
    # 如果有 C++ 实现且运行时可用，优先使用
    if _HAS_ONEBIT_CPP and _ONEBIT_CPP_RUNTIME_AVAILABLE:
        try:
            return cpp_onebit_seide_per_column(g, col_size)
        except Exception as e:
            logger.warning(
                f"[1bit_seide] C++ onebit_seide_per_column unavailable at runtime, "
                f"falling back to Python implementation: {e}"
            )
            _ONEBIT_CPP_RUNTIME_AVAILABLE = False
    numel = g.numel()
    num_cols = (numel + col_size - 1) // col_size
    device = g.device
    dtype = g.dtype
    signs = g.sign()
    signs_pos = (signs >= 0).to(dtype)
    signs_neg = 1.0 - signs_pos

    # 向量化计算每列的 a 和 b
    # 将 g 重塑为 (num_cols, col_size)，不足的用 0 填充
    n_pad = (col_size - numel % col_size) % col_size
    if n_pad > 0:
        g_padded = torch.cat([g, torch.zeros(n_pad, device=device, dtype=dtype)])
        signs_pos_padded = torch.cat([signs_pos, torch.zeros(n_pad, device=device, dtype=dtype)])
        signs_neg_padded = torch.cat([signs_neg, torch.zeros(n_pad, device=device, dtype=dtype)])
    else:
        g_padded = g
        signs_pos_padded = signs_pos
        signs_neg_padded = signs_neg

    g_cols = g_padded.view(num_cols, col_size)  # (num_cols, col_size)
    pos_mask = signs_pos_padded.view(num_cols, col_size)  # (num_cols, col_size)
    neg_mask = signs_neg_padded.view(num_cols, col_size)  # (num_cols, col_size)

    # 计算每列的正值均值和负值均值
    pos_sum = (g_cols * pos_mask).sum(dim=1)  # (num_cols,)
    pos_count = pos_mask.sum(dim=1)  # (num_cols,)
    neg_sum = (g_cols * neg_mask).sum(dim=1)  # (num_cols,)
    neg_count = neg_mask.sum(dim=1)  # (num_cols,)

    # 避免除零
    a_vec = torch.where(pos_count > 0, pos_sum / pos_count.clamp(min=1e-8), torch.zeros_like(pos_sum))
    b_vec = torch.where(neg_count > 0, neg_sum / neg_count.clamp(min=1e-8), torch.zeros_like(neg_sum))

    packed = _pack_signs_to_bytes(signs)
    return packed, a_vec, b_vec


def _onebit_seide_reconstruct_from_gathered(
    signs_all: torch.Tensor,
    a_all: torch.Tensor,
    b_all: torch.Tensor,
    numel: int,
    col_size: int,
    world_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """full_sum[i] = sum_r (a_r[c] if sign_r[i] >= 0 else b_r[c])."""
    global _ONEBIT_CPP_RUNTIME_AVAILABLE
    # 有 C++ 实现且运行时可用时优先使用
    if _HAS_ONEBIT_CPP and _ONEBIT_CPP_RUNTIME_AVAILABLE:
        try:
            return cpp_onebit_seide_reconstruct(signs_all, a_all, b_all, numel, col_size)
        except Exception as e:
            logger.warning(
                f"[1bit_seide] C++ onebit_seide_reconstruct unavailable at runtime, "
                f"falling back to Python implementation: {e}"
            )
            _ONEBIT_CPP_RUNTIME_AVAILABLE = False

    num_cols = a_all.shape[1]
    col_idx = torch.arange(numel, device=device, dtype=torch.long) // col_size
    col_idx = col_idx.clamp(max=num_cols - 1)

    # 批量解包所有 signs（一次性处理所有 rank）
    # signs_all: (world_size, packed_bytes) -> (world_size, numel)
    signs_unpacked = _unpack_signs_from_bytes_all(signs_all, numel)

    # 批量扩展 a 和 b: (world_size, num_cols) -> (world_size, numel)
    col_idx_expanded = col_idx.unsqueeze(0).expand(world_size, -1)  # (world_size, numel)
    col_idx_expanded = col_idx_expanded.clamp(min=0, max=num_cols - 1)

    a_expanded = torch.gather(a_all, dim=1, index=col_idx_expanded)  # (world_size, numel)
    b_expanded = torch.gather(b_all, dim=1, index=col_idx_expanded)  # (world_size, numel)

    recon_all = torch.where(signs_unpacked >= 0, a_expanded, b_expanded)  # (world_size, numel)
    return recon_all.sum(dim=0)  # (numel,)


class OneBitSeideState:
    def __init__(self, col_size: int = 256, error_feedback: bool = True, use_cpp: bool = True, ef_local: bool = False) -> None:
        self.col_size = col_size
        self.error_feedback = error_feedback
        self._ef_local = ef_local
        # 是否尝试使用 C++/CUDA 扩展实现 1-bit Seide（如果可用）
        self.use_cpp = use_cpp


def fsdp_onebit_seide_comm_hook(
    state: OneBitSeideState,
    full_flat_grad: torch.Tensor,
    shard_out: torch.Tensor,
) -> None:
    """1-bit Seide: per-column (a,b) minimize squared error, transmit signs + (a,b), EF."""
    global _ONEBIT_CPP_RUNTIME_AVAILABLE

    pg = dist.group.WORLD
    world_size = dist.get_world_size(pg)
    rank = dist.get_rank(pg)

    if world_size == 1:
        shard_out.copy_(full_flat_grad)
        return

    # 如果显式关闭 C++/CUDA 补丁，则禁用 C++ 路径（本进程内全局）
    if not getattr(state, "use_cpp", True):
        _ONEBIT_CPP_RUNTIME_AVAILABLE = False

    # 移除大部分日志，只保留关键信息（首次调用时）
    if not hasattr(state, "_logged_once"):
        logger.info(
            f"[1bit_seide R{rank}] Hook initialized: col_size={state.col_size}, error_feedback={state.error_feedback}"
        )
        state._logged_once = True

    # 记录原始形状用于调试
    original_shape = full_flat_grad.shape
    g = full_flat_grad.contiguous().view(-1)
    numel_original = g.numel()

    # 如果形状异常，记录警告
    if original_shape != (numel_original,):
        logger.debug(f"[1bit_seide R{rank}] full_flat_grad reshaped: {original_shape} -> {g.shape}")

    if state.error_feedback:
        # 确保 residual 的形状与当前 shard 匹配
        # FSDP 会为不同大小的参数组调用 hook，所以 residual 需要按 shard 大小管理
        if not hasattr(state, "residual") or state.residual.shape != g.shape:
            # 如果 residual 不存在或形状不匹配，重新创建
            state.residual = torch.zeros_like(g)
        residual = state.residual
        g = (g + residual).to(g.dtype)

    numel = g.numel()
    col_size = min(state.col_size, numel)
    if col_size < 1:
        col_size = 1
    num_cols = (numel + col_size - 1) // col_size

    packed, a_vec, b_vec = _onebit_seide_per_column(g, col_size)
    packed_bytes = packed.numel()
    ab_per_rank = num_cols * 2

    # 确保所有 rank 的维度一致（FSDP 应该保证，但添加保护）
    # 使用异步操作减少等待时间
    packed_bytes_tensor = torch.tensor(packed_bytes, device=g.device, dtype=torch.long)
    ab_per_rank_tensor = torch.tensor(ab_per_rank, device=g.device, dtype=torch.long)
    dist.all_reduce(packed_bytes_tensor, op=dist.ReduceOp.MAX, group=pg, async_op=False)
    dist.all_reduce(ab_per_rank_tensor, op=dist.ReduceOp.MAX, group=pg, async_op=False)
    packed_bytes = packed_bytes_tensor.item()
    ab_per_rank = ab_per_rank_tensor.item()

    # 如果当前 rank 的 packed 较小，需要填充
    if packed.numel() < packed_bytes:
        packed_padded = torch.zeros(packed_bytes, device=g.device, dtype=torch.uint8)
        packed_padded[: packed.numel()] = packed
        packed = packed_padded

    # 如果当前 rank 的 ab 较小，需要填充
    ab_vec = torch.cat([a_vec, b_vec])
    if ab_vec.numel() < ab_per_rank:
        ab_padded = torch.zeros(ab_per_rank, device=g.device, dtype=g.dtype)
        ab_padded[: ab_vec.numel()] = ab_vec
        ab_vec = ab_padded

    # 同步 num_cols（FSDP 应该保证相同，但添加保护）
    num_cols_tensor = torch.tensor(num_cols, device=g.device, dtype=torch.long)
    dist.all_reduce(num_cols_tensor, op=dist.ReduceOp.MAX, group=pg, async_op=False)
    num_cols = num_cols_tensor.item()

    # 记录本次 all-gather 的通信字节数（按每个 rank 发送的 payload 估算）
    per_rank_bytes = int(packed_bytes) + int(ab_per_rank) * g.element_size()
    _comm_add_bytes(state, per_rank_bytes)

    # All-gather（这是主要的通信开销）
    if hasattr(dist, "all_gather_into_tensor"):
        signs_buf = torch.empty(world_size * packed_bytes, device=g.device, dtype=torch.uint8)
        ab_buf = torch.empty(world_size * ab_per_rank, device=g.device, dtype=g.dtype)
        dist.all_gather_into_tensor(signs_buf, packed.contiguous(), group=pg)
        dist.all_gather_into_tensor(ab_buf, ab_vec.contiguous(), group=pg)
        signs_all = signs_buf.view(world_size, packed_bytes)
        ab_all = ab_buf.view(world_size, ab_per_rank)
        a_all = ab_all[:, :num_cols]
        b_all = ab_all[:, num_cols : num_cols * 2] if num_cols * 2 <= ab_per_rank else ab_all[:, num_cols:]
    else:
        signs_list = [torch.empty(packed_bytes, device=g.device, dtype=torch.uint8) for _ in range(world_size)]
        ab_list = [torch.empty(ab_per_rank, device=g.device, dtype=g.dtype) for _ in range(world_size)]
        dist.all_gather(signs_list, packed.contiguous(), group=pg)
        dist.all_gather(ab_list, ab_vec.contiguous(), group=pg)
        signs_all = torch.stack(signs_list, dim=0)
        ab_all = torch.stack(ab_list, dim=0)
        a_all = ab_all[:, :num_cols]
        b_all = ab_all[:, num_cols : num_cols * 2] if num_cols * 2 <= ab_per_rank else ab_all[:, num_cols:]

    # 重建（已优化为批量操作）
    full_sum = _onebit_seide_reconstruct_from_gathered(
        signs_all, a_all, b_all, numel, col_size, world_size, g.device, g.dtype
    )

    # 检查 full_sum 的形状
    if full_sum.shape[0] != numel:
        logger.error(
            f"[1bit_seide R{rank}] full_sum shape mismatch: full_sum.shape={full_sum.shape}, expected numel={numel}, g.shape={g.shape}, full_flat_grad.shape={full_flat_grad.shape}"
        )
        raise RuntimeError(f"full_sum shape mismatch: {full_sum.shape} vs {numel}")

    shard_size = numel // world_size
    shard_start = rank * shard_size
    shard_end = min(shard_start + shard_size, numel)

    deq_avg = (full_sum[shard_start:shard_end] / float(world_size)).to(full_flat_grad.dtype)
    shard_out.copy_(deq_avg)

    if state.error_feedback:
        full_reconstructed = full_sum / float(world_size)
        # 确保维度匹配
        if g.shape != full_reconstructed.shape:
            logger.error(
                f"[1bit_seide R{rank}] Shape mismatch: g.shape={g.shape}, full_reconstructed.shape={full_reconstructed.shape}, numel={numel}, full_sum.shape={full_sum.shape}, full_flat_grad.shape={full_flat_grad.shape}"
            )
            raise RuntimeError(
                f"Shape mismatch in error feedback: g.shape={g.shape}, full_reconstructed.shape={full_reconstructed.shape}"
            )
        # 确保 residual 的形状正确（应该已经在上面初始化了）
        if not hasattr(state, "residual") or state.residual.shape != g.shape:
            state.residual = torch.zeros_like(g)
        residual = state.residual
        residual.copy_(g - full_reconstructed)


__all__ = ["OneBitSeideState", "fsdp_onebit_seide_comm_hook"]


