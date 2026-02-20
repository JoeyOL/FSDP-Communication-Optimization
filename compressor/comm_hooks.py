"""
FSDP communication hooks for gradient compression (quantization, sparsification).
Supports optional Error Feedback (EF) for biased compressors.
"""
import torch
import torch.distributed as dist
import math
from typing import Optional, Tuple, Any

try:
    from logger import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Error Feedback: residual buffer and all_gather for full reconstructed grad
# ---------------------------------------------------------------------------

def _ensure_residual(state: Any, full_flat_grad: torch.Tensor) -> torch.Tensor:
    if getattr(state, "residual", None) is None:
        state.residual = torch.zeros_like(
            full_flat_grad, device=full_flat_grad.device, dtype=full_flat_grad.dtype
        )
    return state.residual


def _all_gather_shard(shard: torch.Tensor, group=None) -> torch.Tensor:
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
    group=None,
) -> None:
    """After shard_out is written: all_gather to get full reconstructed, then residual = to_compress - full."""
    full_reconstructed = _all_gather_shard(shard_out, group=group)
    residual = _ensure_residual(state, full_flat_grad)
    residual.copy_(to_compress - full_reconstructed)


# ---------------------------------------------------------------------------
# Sparse communication: all_gather (indices, values) then merge to full tensor
# ---------------------------------------------------------------------------

def _sparse_all_gather_and_merge(
    indices: torch.Tensor,
    values: torch.Tensor,
    numel: int,
    group=None,
) -> torch.Tensor:
    """
    各 rank 发送 (indices, values)，all_gather 后合并为完整梯度向量 sum_r sparse_r。
    indices/values 形状均为 (k,)；返回 full 形状 (numel,)，dtype/device 与 values 一致。
    """
    if group is None:
        group = dist.group.WORLD
    world_size = dist.get_world_size(group)
    rank = dist.get_rank(group)
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


# ---------------------------------------------------------------------------
# INT8 symmetric quantization (existing + optional EF)
# ---------------------------------------------------------------------------

class GradQuantState:
    def __init__(self, num_bits: int = 8, error_feedback: bool = False, variant: str = "linear") -> None:
        self.num_bits = num_bits
        self.error_feedback = error_feedback
        # variant: "linear" (symmetric scale=global_max) or "dynamic_tree" (normalize to [0,1], 7-bit stochastic)
        self.variant = (variant or "linear").lower().strip()


def _int8_linear_quantize(g: torch.Tensor, global_max: torch.Tensor, world_size: int) -> torch.Tensor:
    q = 127
    qr = max(1, q // world_size)
    scale = qr / torch.clamp(global_max, min=1e-8)
    return torch.clamp((g * scale).round(), -qr, qr).to(torch.int8)


def _int8_dynamic_tree_quantize(g: torch.Tensor, scale: float) -> torch.Tensor:
    """Normalize to [0,1] by max, 7-bit stochastic rounding; store as uint8: sign*128 + level (paper 3.1)."""
    if scale < 1e-12:
        return torch.zeros_like(g, dtype=torch.uint8)
    g_norm = (g.abs() / scale).clamp(0.0, 1.0)
    level_float = g_norm * 127.0
    lo = level_float.floor().clamp(0, 126)
    hi = level_float.ceil().clamp(1, 127)
    u = torch.empty_like(g).uniform_(0, 1)
    lev = torch.where(u < (level_float - lo), hi, lo)
    lev = lev.to(torch.int64)
    sign_bit = (g >= 0).to(torch.int64)
    return (sign_bit * 128 + lev).clamp(0, 255).to(torch.uint8)


def _int8_dynamic_tree_dequantize(q_grad: torch.Tensor, scale: float) -> torch.Tensor:
    """Dequant uint8 (sign*128 + 7-bit level) back to float."""
    q = q_grad.to(torch.int64)
    sign = torch.where(q >= 128, 1.0, -1.0)
    lev = (q & 127).float()
    return sign * (lev / 127.0) * scale


def fsdp_quantized_comm_hook(
    state: GradQuantState,
    full_flat_grad: torch.Tensor,
    shard_out: torch.Tensor,
) -> None:
    """
    FSDP communication hook for int8 quantization before reduce-scatter.
    variant=linear: symmetric scale=global_max; variant=dynamic_tree: normalize [0,1] + 7-bit stochastic (paper 3.1).
    With error_feedback: compresses (g + residual) and updates residual.
    """
    pg = dist.group.WORLD
    world_size = dist.get_world_size(pg)

    if world_size == 1:
        shard_out.copy_(full_flat_grad)
        return

    g = full_flat_grad.contiguous().view(-1)
    if state.error_feedback:
        residual = _ensure_residual(state, g)
        g = (g + residual).to(g.dtype)

    numel = g.numel()
    assert numel % world_size == 0, (
        f"flat grad numel {numel} must be divisible by world_size {world_size}"
    )

    local_max = g.abs().max().to(torch.float32)
    global_max = local_max.clone()
    dist.all_reduce(global_max, op=dist.ReduceOp.MAX, group=pg)
    scale_val = global_max.clamp(min=1e-8).item()

    if getattr(state, "variant", "linear") == "dynamic_tree":
        q_grad = _int8_dynamic_tree_quantize(g, scale_val)
        full_size = q_grad.numel()
        if hasattr(dist, "all_gather_into_tensor"):
            all_q = torch.empty(world_size * full_size, device=g.device, dtype=torch.uint8)
            dist.all_gather_into_tensor(all_q, q_grad.contiguous(), group=pg)
        else:
            all_q = torch.cat(dist.all_gather(q_grad.contiguous(), group=pg), dim=0)
        all_decoded = _int8_dynamic_tree_dequantize(all_q.view(world_size, full_size), scale_val)
        full_sum = all_decoded.sum(dim=0)
        shard_size = numel // world_size
        rank = dist.get_rank(pg)
        shard_start = rank * shard_size
        shard_end = shard_start + shard_size
        deq_avg = (full_sum[shard_start:shard_end] / float(world_size)).to(full_flat_grad.dtype)
        shard_out.copy_(deq_avg)
    else:
        qr = max(1, 127 // world_size)
        scale = qr / torch.clamp(global_max, min=1e-8)
        q_grad = _int8_linear_quantize(g, global_max, world_size)
        temp_shard_out = torch.empty_like(shard_out, dtype=torch.int8)
        if hasattr(dist, "reduce_scatter_tensor"):
            dist.reduce_scatter_tensor(temp_shard_out, q_grad, op=dist.ReduceOp.SUM, group=pg)
        else:
            chunks = list(q_grad.chunk(world_size, dim=0))
            dist.reduce_scatter(temp_shard_out, chunks, op=dist.ReduceOp.SUM, group=pg)
        deq_sum = temp_shard_out.float() / scale
        deq_avg = (deq_sum / float(world_size)).to(full_flat_grad.dtype)
        shard_out.copy_(deq_avg)

    if state.error_feedback:
        _apply_error_feedback(state, full_flat_grad, shard_out, g, pg)


# ---------------------------------------------------------------------------
# FP16: half-precision reduce-scatter
# ---------------------------------------------------------------------------

class FP16State:
    def __init__(self, error_feedback: bool = False) -> None:
        self.error_feedback = error_feedback


def fsdp_fp16_comm_hook(
    state: FP16State,
    full_flat_grad: torch.Tensor,
    shard_out: torch.Tensor,
) -> None:
    """Compress gradient to float16 for communication, then dequantize to full precision."""
    pg = dist.group.WORLD
    world_size = dist.get_world_size(pg)

    if world_size == 1:
        shard_out.copy_(full_flat_grad)
        return

    g = full_flat_grad.contiguous().view(-1)
    if state.error_feedback:
        residual = _ensure_residual(state, g)
        g = (g + residual).to(g.dtype)

    g_fp16 = g.half()
    shard_size = g.numel() // world_size
    temp_shard = torch.empty(shard_size, device=g.device, dtype=torch.float16)
    if hasattr(dist, "reduce_scatter_tensor"):
        dist.reduce_scatter_tensor(temp_shard, g_fp16, op=dist.ReduceOp.SUM, group=pg)
    else:
        chunks = list(g_fp16.chunk(world_size, dim=0))
        dist.reduce_scatter(temp_shard, chunks, op=dist.ReduceOp.SUM, group=pg)

    deq_avg = (temp_shard.float() / float(world_size)).to(full_flat_grad.dtype)
    shard_out.copy_(deq_avg)

    if state.error_feedback:
        _apply_error_feedback(state, full_flat_grad, shard_out, g, pg)


# ---------------------------------------------------------------------------
# QSGD: stochastic quantization to s levels (unbiased)
# ---------------------------------------------------------------------------

class QSGDState:
    def __init__(self, s: int = 4, error_feedback: bool = False, bucket_size: int = 0) -> None:
        self.s = max(2, int(s))
        self.error_feedback = error_feedback
        # bucket_size <= 0 or >= numel: whole vector one bucket; else per-bucket QSGD (paper Section 4)
        self.bucket_size = bucket_size


def _qsgd_quantize(v: torch.Tensor, s: int) -> torch.Tensor:
    """QSGD stochastic quantization: scale by norm, round to s levels stochastically."""
    norm = v.norm(p=2)
    if norm < 1e-12:
        return v
    v_n = v / norm
    # v_n in [-1,1]; quantize to {-1, -(s-1)/s, ..., (s-1)/s, 1}
    abs_v = v_n.abs()
    level_float = abs_v * s
    lo = level_float.floor().clamp(0, s - 1)
    hi = level_float.ceil().clamp(0, s)
    p = level_float - lo
    # stochastic rounding
    u = torch.empty_like(v).uniform_(0, 1)
    lev = torch.where(u < p, hi, lo)
    q_abs = lev.float() / s
    q_n = q_abs * v_n.sign()
    return q_n * norm


def _qsgd_quantize_bucketed(g: torch.Tensor, s: int, bucket_size: int) -> torch.Tensor:
    """QSGD per bucket: each bucket of size bucket_size gets independent L2 norm and s-level quantization."""
    numel = g.numel()
    if bucket_size <= 0 or bucket_size >= numel:
        return _qsgd_quantize(g, s)
    out = []
    for start in range(0, numel, bucket_size):
        end = min(start + bucket_size, numel)
        out.append(_qsgd_quantize(g[start:end], s))
    return torch.cat(out)


def fsdp_qsgd_comm_hook(
    state: QSGDState,
    full_flat_grad: torch.Tensor,
    shard_out: torch.Tensor,
) -> None:
    """QSGD: stochastic quantization with s levels (per-bucket if bucket_size set), then reduce-scatter in float."""
    pg = dist.group.WORLD
    world_size = dist.get_world_size(pg)

    if world_size == 1:
        shard_out.copy_(full_flat_grad)
        return

    g = full_flat_grad.contiguous().view(-1)
    if state.error_feedback:
        residual = _ensure_residual(state, g)
        g = (g + residual).to(g.dtype)

    q_g = _qsgd_quantize_bucketed(g, state.s, state.bucket_size)
    shard_size = g.numel() // world_size
    temp_shard = torch.empty(shard_size, device=g.device, dtype=g.dtype)
    if hasattr(dist, "reduce_scatter_tensor"):
        dist.reduce_scatter_tensor(temp_shard, q_g, op=dist.ReduceOp.SUM, group=pg)
    else:
        chunks = list(q_g.chunk(world_size, dim=0))
        dist.reduce_scatter(temp_shard, chunks, op=dist.ReduceOp.SUM, group=pg)

    deq_avg = (temp_shard / float(world_size)).to(full_flat_grad.dtype)
    shard_out.copy_(deq_avg)

    if state.error_feedback:
        _apply_error_feedback(state, full_flat_grad, shard_out, g, pg)


# ---------------------------------------------------------------------------
# 1-bit Seide: per-column two reconstruction values (minimize squared error) + EF
# ---------------------------------------------------------------------------

def _pack_signs_to_bytes(signs: torch.Tensor) -> torch.Tensor:
    """Pack signs (1 = non-negative, 0 = negative) into uint8, 8 per byte, LSB first."""
    flat = (signs >= 0).to(torch.uint8).view(-1)
    n = flat.numel()
    n_pad = (8 - n % 8) % 8
    if n_pad:
        flat = torch.cat([flat, torch.zeros(n_pad, device=flat.device, dtype=torch.uint8)])
    powers = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128], device=flat.device, dtype=torch.uint8)
    return (flat.view(-1, 8) * powers).sum(dim=1).to(torch.uint8)


def _unpack_signs_from_bytes(packed: torch.Tensor, numel: int) -> torch.Tensor:
    """Unpack bytes to signs: 1 for non-negative, -1 for negative. Vectorized."""
    n_bytes = packed.numel()
    n_bits = min(numel, n_bytes * 8)
    if n_bits == 0:
        return torch.zeros(numel, device=packed.device, dtype=torch.float32)
    
    # 向量化：将每个字节展开为 8 个 bit
    packed_int = packed.to(torch.int32)  # shape: (n_bytes,)
    # 创建 bit 位置索引: [0,1,2,3,4,5,6,7] 重复 n_bytes 次
    bit_pos = torch.arange(8, device=packed.device, dtype=torch.int32)  # shape: (8,)
    # 对每个字节，提取所有 8 个 bit: (packed_int.unsqueeze(-1) >> bit_pos) & 1
    bits_expanded = (packed_int.unsqueeze(-1) >> bit_pos.unsqueeze(0)) & 1  # shape: (n_bytes, 8)
    bits_flat = bits_expanded.view(-1)  # shape: (n_bytes * 8,)
    bits_flat = bits_flat[:n_bits]  # 只取需要的 bit
    
    # 转换为 signs: 1 -> 1.0, 0 -> -1.0
    out = torch.zeros(numel, device=packed.device, dtype=torch.float32)
    out[:n_bits] = torch.where(bits_flat == 1, 1.0, -1.0)
    return out


def _onebit_seide_per_column(g: torch.Tensor, col_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """For each column: signs (1/-1), a=mean(positive), b=mean(negative). Vectorized."""
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
    """full_sum[i] = sum over r of (a_r[c] if sign_r[i]>=0 else b_r[c]). Optimized vectorized."""
    num_cols = a_all.shape[1]
    col_idx = torch.arange(numel, device=device, dtype=torch.long) // col_size
    col_idx = col_idx.clamp(max=num_cols - 1)
    
    # 批量解包所有 signs（一次性处理所有 rank）
    signs_unpacked = torch.zeros(world_size, numel, device=device, dtype=torch.float32)
    for r in range(world_size):
        signs_unpacked[r] = _unpack_signs_from_bytes(signs_all[r], numel)
    
    # 批量扩展 a 和 b: (world_size, num_cols) -> (world_size, numel)
    # 使用 gather 而不是高级索引，避免维度问题
    col_idx_expanded = col_idx.unsqueeze(0).expand(world_size, -1)  # (world_size, numel)
    # 确保 col_idx_expanded 的值在有效范围内
    col_idx_expanded = col_idx_expanded.clamp(min=0, max=num_cols - 1)
    
    # 检查维度
    if col_idx_expanded.shape != (world_size, numel):
        raise RuntimeError(f"col_idx_expanded shape mismatch: {col_idx_expanded.shape} vs {(world_size, numel)}")
    if a_all.shape != (world_size, num_cols):
        raise RuntimeError(f"a_all shape mismatch: {a_all.shape} vs {(world_size, num_cols)}")
    
    a_expanded = torch.gather(a_all, dim=1, index=col_idx_expanded)  # (world_size, numel)
    b_expanded = torch.gather(b_all, dim=1, index=col_idx_expanded)  # (world_size, numel)
    
    # 确保维度匹配
    if signs_unpacked.shape != (world_size, numel) or a_expanded.shape != (world_size, numel) or b_expanded.shape != (world_size, numel):
        raise RuntimeError(f"Shape mismatch in reconstruct: signs_unpacked={signs_unpacked.shape}, a_expanded={a_expanded.shape}, b_expanded={b_expanded.shape}, expected={(world_size, numel)}")
    
    # 批量选择: where(signs >= 0, a, b) for each rank
    recon_all = torch.where(signs_unpacked >= 0, a_expanded, b_expanded)  # (world_size, numel)
    
    # 求和所有 rank
    full_sum = recon_all.sum(dim=0)  # (numel,)
    return full_sum


class OneBitSeideState:
    def __init__(self, col_size: int = 256, error_feedback: bool = True) -> None:
        self.col_size = col_size
        self.error_feedback = error_feedback


def fsdp_onebit_seide_comm_hook(
    state: OneBitSeideState,
    full_flat_grad: torch.Tensor,
    shard_out: torch.Tensor,
) -> None:
    """1-bit Seide: per-column (a,b) minimize squared error, transmit signs + (a,b), EF."""
    pg = dist.group.WORLD
    world_size = dist.get_world_size(pg)
    rank = dist.get_rank(pg)

    if world_size == 1:
        shard_out.copy_(full_flat_grad)
        return

    # 移除大部分日志，只保留关键信息（首次调用时）
    if not hasattr(state, '_logged_once'):
        logger.info(f"[1bit_seide R{rank}] Hook initialized: col_size={state.col_size}, error_feedback={state.error_feedback}")
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
        if not hasattr(state, 'residual') or state.residual.shape != g.shape:
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
        packed_padded[:packed.numel()] = packed
        packed = packed_padded
    
    # 如果当前 rank 的 ab 较小，需要填充
    ab_vec = torch.cat([a_vec, b_vec])
    if ab_vec.numel() < ab_per_rank:
        ab_padded = torch.zeros(ab_per_rank, device=g.device, dtype=g.dtype)
        ab_padded[:ab_vec.numel()] = ab_vec
        ab_vec = ab_padded
    
    # 同步 num_cols（FSDP 应该保证相同，但添加保护）
    num_cols_tensor = torch.tensor(num_cols, device=g.device, dtype=torch.long)
    dist.all_reduce(num_cols_tensor, op=dist.ReduceOp.MAX, group=pg, async_op=False)
    num_cols = num_cols_tensor.item()
    
    # All-gather（这是主要的通信开销）
    if hasattr(dist, "all_gather_into_tensor"):
        signs_buf = torch.empty(world_size * packed_bytes, device=g.device, dtype=torch.uint8)
        ab_buf = torch.empty(world_size * ab_per_rank, device=g.device, dtype=g.dtype)
        dist.all_gather_into_tensor(signs_buf, packed.contiguous(), group=pg)
        dist.all_gather_into_tensor(ab_buf, ab_vec.contiguous(), group=pg)
        signs_all = signs_buf.view(world_size, packed_bytes)
        ab_all = ab_buf.view(world_size, ab_per_rank)
        a_all = ab_all[:, :num_cols]
        b_all = ab_all[:, num_cols:num_cols*2] if num_cols * 2 <= ab_per_rank else ab_all[:, num_cols:]
    else:
        signs_list = [torch.empty(packed_bytes, device=g.device, dtype=torch.uint8) for _ in range(world_size)]
        ab_list = [torch.empty(ab_per_rank, device=g.device, dtype=g.dtype) for _ in range(world_size)]
        dist.all_gather(signs_list, packed.contiguous(), group=pg)
        dist.all_gather(ab_list, ab_vec.contiguous(), group=pg)
        signs_all = torch.stack(signs_list, dim=0)
        ab_all = torch.stack(ab_list, dim=0)
        a_all = ab_all[:, :num_cols]
        b_all = ab_all[:, num_cols:num_cols*2] if num_cols * 2 <= ab_per_rank else ab_all[:, num_cols:]

    # 重建（已优化为批量操作）
    full_sum = _onebit_seide_reconstruct_from_gathered(
        signs_all, a_all, b_all, numel, col_size, world_size, g.device, g.dtype
    )
    
    # 检查 full_sum 的形状
    if full_sum.shape[0] != numel:
        logger.error(f"[1bit_seide R{rank}] full_sum shape mismatch: full_sum.shape={full_sum.shape}, expected numel={numel}, g.shape={g.shape}, full_flat_grad.shape={full_flat_grad.shape}")
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
            logger.error(f"[1bit_seide R{rank}] Shape mismatch: g.shape={g.shape}, full_reconstructed.shape={full_reconstructed.shape}, numel={numel}, full_sum.shape={full_sum.shape}, full_flat_grad.shape={full_flat_grad.shape}")
            raise RuntimeError(f"Shape mismatch in error feedback: g.shape={g.shape}, full_reconstructed.shape={full_reconstructed.shape}")
        # 确保 residual 的形状正确（应该已经在上面初始化了）
        if not hasattr(state, 'residual') or state.residual.shape != g.shape:
            state.residual = torch.zeros_like(g)
        residual = state.residual
        residual.copy_(g - full_reconstructed)


# ---------------------------------------------------------------------------
# SignSGD / 1-bit (Bernstein): sign of gradient, majority vote
# ---------------------------------------------------------------------------

class SignSGDState:
    def __init__(self, error_feedback: bool = True, use_delta_scale: bool = False) -> None:
        self.error_feedback = error_feedback
        # use_delta_scale: True = paper "x - δ·sign(g)": output direction only (scale=1), δ from optimizer lr
        self.use_delta_scale = use_delta_scale


def fsdp_signsgd_comm_hook(
    state: SignSGDState,
    full_flat_grad: torch.Tensor,
    shard_out: torch.Tensor,
) -> None:
    """1-bit / SignSGD: communicate sign(g), majority vote; scale by L1 or by δ (lr) when use_delta_scale."""
    pg = dist.group.WORLD
    world_size = dist.get_world_size(pg)

    if world_size == 1:
        shard_out.copy_(full_flat_grad)
        return

    g = full_flat_grad.contiguous().view(-1)
    if state.error_feedback:
        residual = _ensure_residual(state, g)
        g = (g + residual).to(g.dtype)

    sign_g = g.sign()
    shard_size = g.numel() // world_size
    temp_shard = torch.empty(shard_size, device=g.device, dtype=g.dtype)
    if hasattr(dist, "reduce_scatter_tensor"):
        dist.reduce_scatter_tensor(temp_shard, sign_g, op=dist.ReduceOp.SUM, group=pg)
    else:
        chunks = list(sign_g.chunk(world_size, dim=0))
        dist.reduce_scatter(temp_shard, chunks, op=dist.ReduceOp.SUM, group=pg)

    sign_sum = temp_shard
    if getattr(state, "use_delta_scale", False):
        scale = 1.0
    else:
        norm_g = g.norm(p=1) / g.numel()
        scale = norm_g * world_size
    deq_avg = (sign_sum.sign() * scale / float(world_size)).to(full_flat_grad.dtype)
    shard_out.copy_(deq_avg)

    if state.error_feedback:
        _apply_error_feedback(state, full_flat_grad, shard_out, g, pg)


# ---------------------------------------------------------------------------
# Natural Compression (NC): round to nearest power-of-two
# ---------------------------------------------------------------------------

class NCState:
    def __init__(self, error_feedback: bool = False) -> None:
        self.error_feedback = error_feedback


def _nc_compress_scalar(t: torch.Tensor) -> torch.Tensor:
    """Natural compression: randomized rounding to nearest ±2^k."""
    out = torch.empty_like(t)
    zero = t == 0
    out[zero] = 0
    t_nz = t[~zero]
    abs_t = t_nz.abs()
    alpha = torch.log2(abs_t)
    lo = torch.pow(2.0, alpha.floor())
    hi = torch.pow(2.0, alpha.ceil())
    p = (abs_t - lo) / (hi - lo + 1e-12)
    u = torch.empty_like(t_nz, device=t.device).uniform_(0, 1)
    choice = (u < p).float() * hi + (u >= p).float() * lo
    out[~zero] = choice * t_nz.sign()
    return out


def fsdp_nc_comm_hook(
    state: NCState,
    full_flat_grad: torch.Tensor,
    shard_out: torch.Tensor,
) -> None:
    """Natural compression: unbiased round to ±2^k, then reduce-scatter."""
    pg = dist.group.WORLD
    world_size = dist.get_world_size(pg)

    if world_size == 1:
        shard_out.copy_(full_flat_grad)
        return

    g = full_flat_grad.contiguous().view(-1)
    if state.error_feedback:
        residual = _ensure_residual(state, g)
        g = (g + residual).to(g.dtype)

    q_g = _nc_compress_scalar(g)
    shard_size = g.numel() // world_size
    temp_shard = torch.empty(shard_size, device=g.device, dtype=g.dtype)
    if hasattr(dist, "reduce_scatter_tensor"):
        dist.reduce_scatter_tensor(temp_shard, q_g, op=dist.ReduceOp.SUM, group=pg)
    else:
        chunks = list(q_g.chunk(world_size, dim=0))
        dist.reduce_scatter(temp_shard, chunks, op=dist.ReduceOp.SUM, group=pg)

    deq_avg = (temp_shard / float(world_size)).to(full_flat_grad.dtype)
    shard_out.copy_(deq_avg)

    if state.error_feedback:
        _apply_error_feedback(state, full_flat_grad, shard_out, g, pg)


# ---------------------------------------------------------------------------
# Sparsification: Top-k, Random-k, Threshold-v (dense or sparse communication)
# ---------------------------------------------------------------------------

class TopKState:
    def __init__(
        self,
        k: int = 0,
        ratio: float = 0.01,
        error_feedback: bool = True,
        sparse_comm: bool = True,
    ) -> None:
        self.k = k
        self.ratio = ratio
        self.error_feedback = error_feedback
        self.sparse_comm = sparse_comm


def fsdp_topk_comm_hook(
    state: TopKState,
    full_flat_grad: torch.Tensor,
    shard_out: torch.Tensor,
) -> None:
    """Top-k: 只保留绝对值最大的 k 个；sparse_comm 时只传 (indices, values)。"""
    pg = dist.group.WORLD
    world_size = dist.get_world_size(pg)

    if world_size == 1:
        shard_out.copy_(full_flat_grad)
        return

    g = full_flat_grad.contiguous().view(-1)
    if state.error_feedback:
        residual = _ensure_residual(state, g)
        g = (g + residual).to(g.dtype)

    numel = g.numel()
    k = state.k if state.k > 0 else max(1, int(numel * state.ratio))
    k = min(k, numel)
    _, indices = g.abs().topk(k, largest=True, sorted=False)
    values = g[indices].to(g.dtype)

    if state.sparse_comm:
        full_sum = _sparse_all_gather_and_merge(indices, values, numel, pg)
        shard_size = numel // world_size
        rank = dist.get_rank(pg)
        shard_start = rank * shard_size
        shard_end = shard_start + shard_size
        deq_avg = (full_sum[shard_start:shard_end] / float(world_size)).to(full_flat_grad.dtype)
        shard_out.copy_(deq_avg)
        if state.error_feedback:
            sparse_full = torch.zeros_like(g)
            sparse_full[indices] = values
            _apply_error_feedback(state, full_flat_grad, shard_out, sparse_full, pg)
        return

    sparse = torch.zeros_like(g)
    sparse[indices] = values
    shard_size = numel // world_size
    temp_shard = torch.empty(shard_size, device=g.device, dtype=g.dtype)
    if hasattr(dist, "reduce_scatter_tensor"):
        dist.reduce_scatter_tensor(temp_shard, sparse, op=dist.ReduceOp.SUM, group=pg)
    else:
        chunks = list(sparse.chunk(world_size, dim=0))
        dist.reduce_scatter(temp_shard, chunks, op=dist.ReduceOp.SUM, group=pg)

    deq_avg = (temp_shard / float(world_size)).to(full_flat_grad.dtype)
    shard_out.copy_(deq_avg)

    if state.error_feedback:
        _apply_error_feedback(state, full_flat_grad, shard_out, sparse, pg)


class RandomKState:
    def __init__(
        self,
        k: int = 0,
        ratio: float = 0.01,
        error_feedback: bool = True,
        sparse_comm: bool = True,
    ) -> None:
        self.k = k
        self.ratio = ratio
        self.error_feedback = error_feedback
        self.sparse_comm = sparse_comm


def fsdp_randomk_comm_hook(
    state: RandomKState,
    full_flat_grad: torch.Tensor,
    shard_out: torch.Tensor,
) -> None:
    """Random-k: 随机保留 k 个坐标并乘 d/k 无偏；sparse_comm 时只传 (indices, values)。"""
    pg = dist.group.WORLD
    world_size = dist.get_world_size(pg)

    if world_size == 1:
        shard_out.copy_(full_flat_grad)
        return

    g = full_flat_grad.contiguous().view(-1)
    if state.error_feedback:
        residual = _ensure_residual(state, g)
        g = (g + residual).to(g.dtype)

    numel = g.numel()
    k = state.k if state.k > 0 else max(1, int(numel * state.ratio))
    k = min(k, numel)
    scale = numel / k
    perm = torch.randperm(numel, device=g.device)
    indices = perm[:k]
    values = (g[indices] * scale).to(g.dtype)

    if state.sparse_comm:
        full_sum = _sparse_all_gather_and_merge(indices, values, numel, pg)
        shard_size = numel // world_size
        rank = dist.get_rank(pg)
        shard_start = rank * shard_size
        shard_end = shard_start + shard_size
        deq_avg = (full_sum[shard_start:shard_end] / float(world_size)).to(full_flat_grad.dtype)
        shard_out.copy_(deq_avg)
        if state.error_feedback:
            sparse_full = torch.zeros_like(g)
            sparse_full[indices] = values
            _apply_error_feedback(state, full_flat_grad, shard_out, sparse_full, pg)
        return

    sparse = torch.zeros_like(g)
    sparse[indices] = values
    shard_size = numel // world_size
    temp_shard = torch.empty(shard_size, device=g.device, dtype=g.dtype)
    if hasattr(dist, "reduce_scatter_tensor"):
        dist.reduce_scatter_tensor(temp_shard, sparse, op=dist.ReduceOp.SUM, group=pg)
    else:
        chunks = list(sparse.chunk(world_size, dim=0))
        dist.reduce_scatter(temp_shard, chunks, op=dist.ReduceOp.SUM, group=pg)

    deq_avg = (temp_shard / float(world_size)).to(full_flat_grad.dtype)
    shard_out.copy_(deq_avg)

    if state.error_feedback:
        _apply_error_feedback(state, full_flat_grad, shard_out, sparse, pg)


class ThresholdVState:
    def __init__(
        self,
        v: float = 0.0,
        v_neg: Optional[float] = None,
        ratio: float = 0.01,
        error_feedback: bool = True,
        sparse_comm: bool = True,
    ) -> None:
        self.v = v
        # v_neg: negative threshold; if None or <0, use v for both (symmetric)
        self.v_neg = v_neg if v_neg is not None and v_neg >= 0 else v

    @property
    def v_pos(self) -> float:
        return self.v

    @property
    def v_neg_val(self) -> float:
        return self.v_neg


def _thresholdv_to_fixed_k_indices_values(
    g: torch.Tensor,
    v_pos: float,
    v_neg: float,
    k: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """保留 g_i >= v_pos 或 g_i <= -v_neg 的坐标；若超过 k 个取绝对值最大的 k 个，不足则用 (0,0) 填充。"""
    mask = (g >= v_pos) | (g <= -v_neg)
    cand_indices = torch.nonzero(mask, as_tuple=False).squeeze(-1)
    if cand_indices.dim() == 0:
        cand_indices = cand_indices.unsqueeze(0)
    cand_values = g[cand_indices]
    n = cand_indices.numel()
    if n >= k:
        _, order = cand_values.abs().topk(k, largest=True, sorted=False)
        indices = cand_indices[order]
        values = cand_values[order]
    else:
        indices = torch.zeros(k, device=g.device, dtype=torch.long)
        values = torch.zeros(k, device=g.device, dtype=g.dtype)
        indices[:n] = cand_indices
        values[:n] = cand_values
    return indices, values


def fsdp_thresholdv_comm_hook(
    state: ThresholdVState,
    full_flat_grad: torch.Tensor,
    shard_out: torch.Tensor,
) -> None:
    """Threshold-v: 只传 |g_i| > v 的维度；sparse_comm 时只传 (indices, values)，固定 k 长。"""
    pg = dist.group.WORLD
    world_size = dist.get_world_size(pg)

    if world_size == 1:
        shard_out.copy_(full_flat_grad)
        return

    g = full_flat_grad.contiguous().view(-1)
    if state.error_feedback:
        residual = _ensure_residual(state, g)
        g = (g + residual).to(g.dtype)

    numel = g.numel()
    k = max(1, int(numel * state.ratio))

    if state.v > 0 or state.v_neg > 0:
        v_pos = state.v_pos
        v_neg = state.v_neg_val
    else:
        abs_g = g.abs()
        idx = max(1, numel - min(k, numel))
        v_pos = v_neg = torch.kthvalue(abs_g, idx).values.item()

    if state.sparse_comm:
        indices, values = _thresholdv_to_fixed_k_indices_values(g, v_pos, v_neg, k)
        full_sum = _sparse_all_gather_and_merge(indices, values, numel, pg)
        shard_size = numel // world_size
        rank = dist.get_rank(pg)
        shard_start = rank * shard_size
        shard_end = shard_start + shard_size
        deq_avg = (full_sum[shard_start:shard_end] / float(world_size)).to(full_flat_grad.dtype)
        shard_out.copy_(deq_avg)
        if state.error_feedback:
            sparse_full = torch.where((g >= v_pos) | (g <= -v_neg), g, torch.zeros_like(g))
            _apply_error_feedback(state, full_flat_grad, shard_out, sparse_full, pg)
        return

    sparse = torch.where((g >= v_pos) | (g <= -v_neg), g, torch.zeros_like(g))
    shard_size = numel // world_size
    temp_shard = torch.empty(shard_size, device=g.device, dtype=g.dtype)
    if hasattr(dist, "reduce_scatter_tensor"):
        dist.reduce_scatter_tensor(temp_shard, sparse, op=dist.ReduceOp.SUM, group=pg)
    else:
        chunks = list(sparse.chunk(world_size, dim=0))
        dist.reduce_scatter(temp_shard, chunks, op=dist.ReduceOp.SUM, group=pg)

    deq_avg = (temp_shard / float(world_size)).to(full_flat_grad.dtype)
    shard_out.copy_(deq_avg)

    if state.error_feedback:
        _apply_error_feedback(state, full_flat_grad, shard_out, sparse, pg)


# ---------------------------------------------------------------------------
# Sketched-SGD: Count Sketch for heavy hitters (simplified: sketch then top-k on recovered)
# ---------------------------------------------------------------------------

def _count_sketch_encode(g: torch.Tensor, width: int, depth: int) -> torch.Tensor:
    d = g.numel()
    sketch = torch.zeros(depth, width, device=g.device, dtype=g.dtype)
    for dep in range(depth):
        h = (torch.arange(d, device=g.device) * (dep + 1) + dep) % width
        s = 2 * (torch.arange(d, device=g.device) % 2) - 1
        sketch.index_add_(1, h, g * s)
    return sketch


def _count_sketch_decode_all(sketch: torch.Tensor, d: int, width: int, depth: int) -> torch.Tensor:
    est = torch.zeros(d, device=sketch.device, dtype=sketch.dtype)
    for dep in range(depth):
        h = (torch.arange(d, device=sketch.device) * (dep + 1) + dep) % width
        s = 2 * (torch.arange(d, device=sketch.device) % 2) - 1
        est += sketch[dep, h] * s
    est /= depth
    return est


def _heavymix_topk_indices(sketch: torch.Tensor, d: int, width: int, depth: int, k: int) -> torch.Tensor:
    """HEAVYMIX (Algorithm 1): from merged sketch get Topk indices. H = {i : hat_g_i^2 >= hat_l2^2/k}, Topk = H cup rand_l(NH)."""
    est = _count_sketch_decode_all(sketch, d, width, depth)
    est_sq = est * est
    hat_l2_sq = est_sq.sum().clamp(min=1e-12)
    thresh = hat_l2_sq / max(1, k)
    H_mask = est_sq >= thresh
    H_indices = torch.nonzero(H_mask, as_tuple=False).squeeze(-1)
    if H_indices.dim() == 0:
        H_indices = H_indices.unsqueeze(0)
    n_H = H_indices.numel()
    NH_mask = ~H_mask
    NH_indices = torch.nonzero(NH_mask, as_tuple=False).squeeze(-1)
    if NH_indices.dim() == 0:
        NH_indices = NH_indices.unsqueeze(0)
    l = min(k - n_H, NH_indices.numel())
    if l <= 0:
        return H_indices[:k]
    perm = torch.randperm(NH_indices.numel(), device=NH_indices.device)[:l]
    rand_NH = NH_indices[perm]
    topk = torch.cat([H_indices, rand_NH], dim=0)
    if topk.numel() > k:
        topk = topk[:k]
    return topk


class SketchState:
    def __init__(self, k: int = 0, ratio: float = 0.01, error_feedback: bool = True, two_round: bool = False) -> None:
        self.k = k
        self.ratio = ratio
        self.error_feedback = error_feedback
        self.two_round = two_round
        self._sketch_depth = 3
        self._sketch_width = 0

    def _sketch_size(self, d: int) -> int:
        k = self.k if self.k > 0 else max(1, int(d * self.ratio))
        return max(k * 8, 256)

    def _count_sketch(self, g: torch.Tensor, width: int, depth: int) -> torch.Tensor:
        return _count_sketch_encode(g, width, depth)

    def _recover_heavy(self, g: torch.Tensor, sketch: torch.Tensor, width: int, depth: int, k: int) -> torch.Tensor:
        d = g.numel()
        est = _count_sketch_decode_all(sketch, d, width, depth)
        _, indices = est.abs().topk(k, largest=True, sorted=False)
        out = torch.zeros_like(g)
        out[indices] = g[indices]
        return out


def fsdp_sketch_comm_hook(
    state: SketchState,
    full_flat_grad: torch.Tensor,
    shard_out: torch.Tensor,
) -> None:
    """Sketched-SGD: two_round=True => round1 send sketch, merge, HEAVYMIX top-k indices; round2 send exact values at Topk."""
    pg = dist.group.WORLD
    world_size = dist.get_world_size(pg)

    if world_size == 1:
        shard_out.copy_(full_flat_grad)
        return

    g = full_flat_grad.contiguous().view(-1)
    if state.error_feedback:
        residual = _ensure_residual(state, g)
        g = (g + residual).to(g.dtype)

    d = g.numel()
    k = state.k if state.k > 0 else max(1, int(d * state.ratio))
    k = min(k, d)
    width = state._sketch_size(d)
    depth = state._sketch_depth

    if getattr(state, "two_round", False):
        sketch = state._count_sketch(g, width, depth)
        sketch_flat = sketch.view(-1)
        if hasattr(dist, "all_reduce"):
            dist.all_reduce(sketch_flat, op=dist.ReduceOp.SUM, group=pg)
        merged_sketch = sketch_flat.view(depth, width) / float(world_size)
        topk_indices = _heavymix_topk_indices(merged_sketch, d, width, depth, k)
        values_at_topk = g[topk_indices]
        if hasattr(dist, "all_gather_into_tensor"):
            all_vals = torch.empty(world_size * k, device=g.device, dtype=g.dtype)
            dist.all_gather_into_tensor(all_vals, values_at_topk.contiguous(), group=pg)
        else:
            all_vals = torch.cat(dist.all_gather(values_at_topk.contiguous(), group=pg), dim=0)
        full_sum = torch.zeros(d, device=g.device, dtype=g.dtype)
        full_sum.scatter_add_(0, topk_indices, all_vals.view(world_size, k).sum(dim=0))
        shard_size = d // world_size
        rank = dist.get_rank(pg)
        shard_start = rank * shard_size
        shard_end = shard_start + shard_size
        deq_avg = (full_sum[shard_start:shard_end] / float(world_size)).to(full_flat_grad.dtype)
        shard_out.copy_(deq_avg)
        if state.error_feedback:
            full_reconstructed = torch.zeros_like(g)
            full_reconstructed[topk_indices] = full_sum[topk_indices] / float(world_size)
            residual = _ensure_residual(state, full_flat_grad)
            residual.copy_(g - full_reconstructed)
        return

    sketch = state._count_sketch(g, width, depth)
    sparse = state._recover_heavy(g, sketch, width, depth, k)
    shard_size = d // world_size
    temp_shard = torch.empty(shard_size, device=g.device, dtype=g.dtype)
    if hasattr(dist, "reduce_scatter_tensor"):
        dist.reduce_scatter_tensor(temp_shard, sparse, op=dist.ReduceOp.SUM, group=pg)
    else:
        chunks = list(sparse.chunk(world_size, dim=0))
        dist.reduce_scatter(temp_shard, chunks, op=dist.ReduceOp.SUM, group=pg)
    deq_avg = (temp_shard / float(world_size)).to(full_flat_grad.dtype)
    shard_out.copy_(deq_avg)
    if state.error_feedback:
        _apply_error_feedback(state, full_flat_grad, shard_out, sparse, pg)


# ---------------------------------------------------------------------------
# Hybrid: Top-k then INT8 on non-zero values (dense reduce_scatter of quantized full)
# ---------------------------------------------------------------------------

class HybridTopKInt8State:
    def __init__(self, k: int = 0, ratio: float = 0.01, error_feedback: bool = True) -> None:
        self.k = k
        self.ratio = ratio
        self.error_feedback = error_feedback


def fsdp_hybrid_topk_int8_comm_hook(
    state: HybridTopKInt8State,
    full_flat_grad: torch.Tensor,
    shard_out: torch.Tensor,
) -> None:
    """Hybrid: apply top-k sparsification then int8 quantize the sparse vector; reduce-scatter in int8."""
    pg = dist.group.WORLD
    world_size = dist.get_world_size(pg)

    if world_size == 1:
        shard_out.copy_(full_flat_grad)
        return

    g = full_flat_grad.contiguous().view(-1)
    if state.error_feedback:
        residual = _ensure_residual(state, g)
        g = (g + residual).to(g.dtype)

    numel = g.numel()
    k = state.k if state.k > 0 else max(1, int(numel * state.ratio))
    k = min(k, numel)
    _, indices = g.abs().topk(k, largest=True, sorted=False)
    sparse = torch.zeros_like(g)
    sparse[indices] = g[indices]

    local_max = sparse.abs().max().to(torch.float32)
    global_max = local_max.clone()
    dist.all_reduce(global_max, op=dist.ReduceOp.MAX, group=pg)
    qr = max(1, 127 // world_size)
    scale = qr / torch.clamp(global_max, min=1e-8)
    q_grad = torch.clamp((sparse * scale).round(), -qr, qr).to(torch.int8)

    temp_shard_out = torch.empty_like(shard_out, dtype=torch.int8)
    if hasattr(dist, "reduce_scatter_tensor"):
        dist.reduce_scatter_tensor(temp_shard_out, q_grad, op=dist.ReduceOp.SUM, group=pg)
    else:
        chunks = list(q_grad.chunk(world_size, dim=0))
        dist.reduce_scatter(temp_shard_out, chunks, op=dist.ReduceOp.SUM, group=pg)

    deq_sum = temp_shard_out.float() / scale
    deq_avg = (deq_sum / float(world_size)).to(full_flat_grad.dtype)
    shard_out.copy_(deq_avg)

    if state.error_feedback:
        _apply_error_feedback(state, full_flat_grad, shard_out, sparse, pg)


# ---------------------------------------------------------------------------
# Build: name -> (state, hook) with optional kwargs
# ---------------------------------------------------------------------------

def build_comm_hook(
    name: str,
    num_bits: int = 8,
    int8_variant: str = "linear",
    error_feedback: bool = False,
    sparse_comm: bool = True,
    qsgd_s: int = 4,
    qsgd_bucket_size: int = 0,
    topk_k: int = 0,
    topk_ratio: float = 0.01,
    randomk_k: int = 0,
    randomk_ratio: float = 0.01,
    threshold_v: float = 0.0,
    threshold_v_neg: Optional[float] = None,
    threshold_ratio: float = 0.01,
    sketch_k: int = 0,
    sketch_ratio: float = 0.01,
    sketch_two_round: bool = False,
    onebit_col_size: int = 256,
    signsgd_use_delta: bool = False,
    **kwargs: Any,
) -> Tuple[Optional[Any], Optional[Any]]:
    if name is None:
        return None, None

    n = name.lower().strip()
    if n == "none":
        return None, None

    ef = error_feedback

    if n == "int8":
        return GradQuantState(num_bits=num_bits, error_feedback=ef, variant=int8_variant), fsdp_quantized_comm_hook
    if n == "fp16":
        return FP16State(error_feedback=ef), fsdp_fp16_comm_hook
    if n == "qsgd":
        return QSGDState(s=qsgd_s, error_feedback=ef, bucket_size=qsgd_bucket_size), fsdp_qsgd_comm_hook
    if n == "signsgd" or n == "onebit":
        return SignSGDState(error_feedback=ef, use_delta_scale=signsgd_use_delta), fsdp_signsgd_comm_hook
    if n == "onebit_seide":
        return OneBitSeideState(col_size=onebit_col_size, error_feedback=ef), fsdp_onebit_seide_comm_hook
    if n == "nc":
        return NCState(error_feedback=ef), fsdp_nc_comm_hook
    if n == "topk":
        return TopKState(k=topk_k, ratio=topk_ratio, error_feedback=ef, sparse_comm=sparse_comm), fsdp_topk_comm_hook
    if n == "randomk":
        return RandomKState(k=randomk_k, ratio=randomk_ratio, error_feedback=ef, sparse_comm=sparse_comm), fsdp_randomk_comm_hook
    if n == "thresholdv":
        return ThresholdVState(v=threshold_v, v_neg=threshold_v_neg, ratio=threshold_ratio, error_feedback=ef, sparse_comm=sparse_comm), fsdp_thresholdv_comm_hook
    if n == "sketch":
        return SketchState(k=sketch_k, ratio=sketch_ratio, error_feedback=ef, two_round=sketch_two_round), fsdp_sketch_comm_hook
    if n == "hybrid_topk_int8" or n == "topk_int8":
        return HybridTopKInt8State(k=topk_k, ratio=topk_ratio, error_feedback=ef), fsdp_hybrid_topk_int8_comm_hook

    raise ValueError(
        f"Unsupported comm hook: {name}. "
        "Supported: none, int8, fp16, qsgd, signsgd, onebit, onebit_seide, nc, topk, randomk, thresholdv, sketch, hybrid_topk_int8"
    )
