"""
Natural Compression (NC): round to ±2^k, optional 9-bit packing for communication.
- use_bit_packing=False: reduce_scatter(float32), same comm volume as baseline.
- use_bit_packing=True: encode to 9 bits/value (1 sign + 8 exponent), pack to bytes, all_to_all, unpack and sum.
  Comm volume ≈ 9/32 of baseline (~3.5× less).
- use_norm_scale=True (default): scale gradient by 1/||g||_2 before quantize, unscale after; reduces loss by
  quantizing in normalized space so small components are not rounded to zero.
"""

import torch
import torch.distributed as dist

from .common import _apply_error_feedback, _ensure_residual, logger
from perf.comm_stats import add_bytes as _comm_add_bytes

# Exponent range for ±2^k: 为了避免 exp2 溢出到 float32 上界，这里使用更保守的指数区间。
# 理论上 float32 的安全指数约在 [-126, 127]，但在多卡求和时极大值仍可能在 reduce/all_to_all
# 后产生 Inf。综合实验规模与数值稳定性，本实现将 k 限制在较窄的区间 [-60, 60]：
#   2^60 ≈ 1e18，远低于 float32 最大值，即便多卡求和也有较大安全裕度。
_K_OFFSET = 128
_K_MIN, _K_MAX = -60, 60


class NCState:
    def __init__(
        self,
        error_feedback: bool = False,
        use_bit_packing: bool = True,
        ef_local: bool = False,
        use_norm_scale: bool = True,
    ) -> None:
        self.error_feedback = error_feedback
        self._ef_local = ef_local
        self.use_bit_packing = use_bit_packing
        # 范数缩放：先 g/||g||_2 再量化，解码后乘回，减轻小分量被舍入为 0，改善 loss
        self.use_norm_scale = use_norm_scale


def _nc_compress_scalar(t: torch.Tensor) -> torch.Tensor:
    """Natural compression: randomized rounding to nearest ±2^k. Uses exp2 to avoid extra pow."""
    out = torch.empty_like(t)
    zero = t == 0
    out[zero] = 0
    t_nz = t[~zero]
    abs_t = t_nz.abs()
    alpha = torch.log2(abs_t)
    # 将指数限制在 [_K_MIN, _K_MAX] 内，避免 exp2 溢出。
    alpha_clamped = alpha.clamp(min=float(_K_MIN), max=float(_K_MAX))
    lo = torch.exp2(alpha_clamped.floor())
    hi = torch.exp2(alpha_clamped.ceil())
    p = (abs_t - lo) / (hi - lo + 1e-12)
    u = torch.empty_like(t_nz, device=t.device).uniform_(0, 1)
    choice = torch.where(u < p, hi, lo)
    out[~zero] = choice * t_nz.sign()
    return out


def _nc_encode_codes(q: torch.Tensor) -> torch.Tensor:
    """Encode ±2^k (or 0) to 9-bit codes. q: float tensor; returns uint16 tensor same shape, values in [0, 511]."""
    zero = q == 0
    abs_q = q.abs().clamp(min=torch.finfo(q.dtype).tiny)
    k = torch.log2(abs_q).round().clamp(_K_MIN, _K_MAX).to(torch.int32)
    k_stored = (k + _K_OFFSET).clamp(1, 255)  # 1..255
    sign = (q < 0).to(torch.int32)
    code = torch.where(zero, torch.zeros_like(k, device=q.device), (sign * 256 + k_stored).to(torch.int32))
    return code.to(torch.uint16)


def _nc_decode_codes(codes: torch.Tensor) -> torch.Tensor:
    """Decode 9-bit codes to float ±2^k (or 0). codes: uint16; returns float32. Use int32 for bitwise (CUDA has no uint16 bitwise)."""
    codes_i = codes.to(torch.int32)
    zero = codes_i == 0
    k_stored = codes_i & 0xFF
    sign = (codes_i >> 8).float() * 2 - 1  # 0 -> 1, 1 -> -1
    exp = (k_stored.float() - _K_OFFSET).to(codes.device)
    val = torch.exp2(exp)
    return torch.where(zero, torch.zeros_like(val, device=codes.device), sign * val)


def _nc_pack_9bit(codes: torch.Tensor) -> torch.Tensor:
    """Pack 9-bit codes (uint16) into bytes. codes numel must be multiple of 8. Returns uint8 tensor of shape (numel*9//8,)."""
    n = codes.numel()
    assert n % 8 == 0, "codes numel must be multiple of 8"
    c = codes.view(-1, 8).to(torch.int32)  # (num_groups, 8)
    b0 = (c[:, 0] & 0xFF).to(torch.uint8)
    b1 = ((c[:, 0] >> 8) | ((c[:, 1] & 0x7F) << 1)).to(torch.uint8)
    b2 = ((c[:, 1] >> 7) | ((c[:, 2] & 0x3F) << 2)).to(torch.uint8)
    b3 = ((c[:, 2] >> 6) | ((c[:, 3] & 0x1F) << 3)).to(torch.uint8)
    b4 = ((c[:, 3] >> 5) | ((c[:, 4] & 0x0F) << 4)).to(torch.uint8)
    b5 = ((c[:, 4] >> 4) | ((c[:, 5] & 0x07) << 5)).to(torch.uint8)
    b6 = ((c[:, 5] >> 3) | ((c[:, 6] & 0x03) << 6)).to(torch.uint8)
    b7 = ((c[:, 6] >> 2) | ((c[:, 7] & 0x01) << 7)).to(torch.uint8)
    b8 = (c[:, 7] >> 1).to(torch.uint8)
    return torch.stack([b0, b1, b2, b3, b4, b5, b6, b7, b8], dim=1).flatten()


def _nc_unpack_9bit(bytes_t: torch.Tensor) -> torch.Tensor:
    """Unpack bytes to 9-bit codes. bytes_t numel must be multiple of 9. Returns uint16 tensor of shape (numel*8//9,)."""
    n = bytes_t.numel()
    assert n % 9 == 0, "bytes numel must be multiple of 9"
    b = bytes_t.view(-1, 9).to(torch.int32)  # (num_groups, 9)
    c0 = (b[:, 0] | ((b[:, 1] & 1) << 8)).to(torch.uint16)
    c1 = ((b[:, 1] >> 1) | ((b[:, 2] & 3) << 7)).to(torch.uint16)
    c2 = ((b[:, 2] >> 2) | ((b[:, 3] & 7) << 5)).to(torch.uint16)
    c3 = ((b[:, 3] >> 3) | ((b[:, 4] & 15) << 4)).to(torch.uint16)
    c4 = ((b[:, 4] >> 4) | ((b[:, 5] & 31) << 3)).to(torch.uint16)
    c5 = ((b[:, 5] >> 5) | ((b[:, 6] & 63) << 2)).to(torch.uint16)
    c6 = ((b[:, 6] >> 6) | ((b[:, 7] & 127) << 1)).to(torch.uint16)
    c7 = ((b[:, 7] >> 7) | (b[:, 8] << 1)).to(torch.uint16)
    return torch.stack([c0, c1, c2, c3, c4, c5, c6, c7], dim=1).flatten()


def _nc_pack_shard(codes: torch.Tensor, shard_size: int) -> torch.Tensor:
    """Pack a shard of codes (length shard_size). If shard_size not multiple of 8, pad with zeros."""
    n = codes.numel()
    if n % 8 != 0:
        pad = 8 - (n % 8)
        codes = torch.nn.functional.pad(codes.view(-1), (0, pad), value=0)
    return _nc_pack_9bit(codes)


def _nc_unpack_shard(bytes_t: torch.Tensor, out_size: int) -> torch.Tensor:
    """Unpack bytes to codes and return first out_size codes (drop padding)."""
    codes = _nc_unpack_9bit(bytes_t)
    return codes[:out_size]


def _nc_pack_all_shards(all_codes: torch.Tensor, world_size: int, shard_size: int) -> torch.Tensor:
    """Pack all shards in one go. all_codes: (world_size * shard_size,). Returns (world_size * packed_shard_size,) uint8."""
    padded_shard_el = ((shard_size + 7) // 8) * 8
    codes_2d = all_codes.view(world_size, shard_size)
    if padded_shard_el > shard_size:
        codes_2d = torch.nn.functional.pad(codes_2d, (0, padded_shard_el - shard_size), value=0)
    codes_flat = codes_2d.reshape(-1)
    return _nc_pack_9bit(codes_flat)


def _nc_unpack_all_shards_and_decode(
    recv_buf: torch.Tensor, world_size: int, shard_size: int, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    """Unpack entire recv_buf, decode in one go, sum over shards. Returns (shard_size,) float."""
    padded_shard_el = ((shard_size + 7) // 8) * 8
    all_codes = _nc_unpack_9bit(recv_buf)
    decoded = _nc_decode_codes(all_codes)
    decoded_2d = decoded.view(world_size, padded_shard_el)
    return decoded_2d[:, :shard_size].sum(dim=0).to(dtype=dtype, device=device)


def fsdp_nc_comm_hook(
    state: NCState,
    full_flat_grad: torch.Tensor,
    shard_out: torch.Tensor,
) -> None:
    """Natural compression: round to ±2^k. If use_bit_packing: 9-bit encode, all_to_all bytes, unpack and sum. Else: reduce_scatter(float32)."""
    pg = dist.group.WORLD
    world_size = dist.get_world_size(pg)

    if world_size == 1:
        shard_out.copy_(full_flat_grad)
        return

    g = full_flat_grad.contiguous().view(-1)

    # 安全检查 1：输入梯度必须是有限值
    if not torch.isfinite(g).all():
        raise RuntimeError(
            "[nc] detected non-finite gradient (NaN/Inf) before compression; aborting this training run"
        )
    if state.error_feedback:
        residual, start, end = _ensure_residual(state, g, pg)
        if start is None:
            g = (g + residual).to(g.dtype)
        else:
            g[start:end] += residual

    scale = 1.0
    # EF 必须用原始空间的 g：shard_out 最终是 (decoded/world_size)/scale，也在原始空间，故 residual = g_for_ef - reconstructed 一致
    g_for_ef = g
    if getattr(state, "use_norm_scale", True):
        # 范数缩放：在归一化空间量化，避免小分量被舍入为 0，改善 loss
        # 第一步数值稳定改动：
        # 1）对 global_norm 加下界 tiny，避免除 0；
        # 2）对缩放因子 1/global_norm 做上界裁剪，避免 scale 过大导致量化空间过窄；
        # 3）当 global_norm 极小时，直接跳过范数缩放（scale=1），视为该步梯度近似 0。
        norm_sq = (g * g).sum().to(torch.float32)
        dist.all_reduce(norm_sq, op=dist.ReduceOp.SUM, group=pg)
        global_norm = norm_sq.sqrt()

        eps = 1e-6
        tiny_norm = 1e-3
        max_scale = 1e3

        if not torch.isfinite(global_norm):
            # 若范数本身非有限，放弃缩放，保持原梯度（后续安全检查仍会兜底）
            global_norm_val = 1.0
            do_scale = False
        else:
            global_norm_val = float(global_norm.item())
            # 极小范数：梯度整体接近 0，直接跳过缩放，避免被 1/tiny 放大
            do_scale = global_norm_val > tiny_norm

        if do_scale:
            inv_norm = 1.0 / max(global_norm_val, eps)
            inv_norm = min(inv_norm, max_scale)
            scale = inv_norm
            g = g * scale

    q_g = _nc_compress_scalar(g)

    # 安全检查 2：量化后的张量也必须是有限值
    if not torch.isfinite(q_g).all():
        raise RuntimeError(
            "[nc] detected non-finite value after NC quantization; aborting this training run"
        )
    shard_size = g.numel() // world_size

    if state.use_bit_packing:
        # 9 bits/value: pack per shard, all_to_all, unpack and sum. Comm volume ≈ 9/32 of float32.
        # 优化：批量编码 + 一次性打包 + 一次性解包解码，消除 per-shard 循环
        all_codes = _nc_encode_codes(q_g)
        send_buf = _nc_pack_all_shards(all_codes, world_size, shard_size)
        recv_buf = torch.empty_like(send_buf)
        if hasattr(dist, "all_to_all_single"):
            dist.all_to_all_single(recv_buf, send_buf, group=pg)
        else:
            send_list = list(send_buf.chunk(world_size, dim=0))
            recv_list = list(recv_buf.chunk(world_size, dim=0))
            dist.all_to_all(recv_list, send_list, group=pg)
        # 近似统计：按 send_buf 元素数估算 all_to_all 负载
        try:
            _comm_add_bytes(state, send_buf.numel() * send_buf.element_size())
        except Exception:
            pass
        temp_shard = _nc_unpack_all_shards_and_decode(recv_buf, world_size, shard_size, g.dtype, g.device)
    else:
        temp_shard = torch.empty(shard_size, device=g.device, dtype=g.dtype)
        if hasattr(dist, "reduce_scatter_tensor"):
            dist.reduce_scatter_tensor(temp_shard, q_g, op=dist.ReduceOp.SUM, group=pg)
        else:
            chunks = list(q_g.chunk(world_size, dim=0))
            dist.reduce_scatter(temp_shard, chunks, op=dist.ReduceOp.SUM, group=pg)
        # 近似统计：按 q_g 元素数估算 reduce_scatter 负载
        try:
            _comm_add_bytes(state, q_g.numel() * q_g.element_size())
        except Exception:
            pass

    deq_avg = (temp_shard / float(world_size)).to(full_flat_grad.dtype)
    if getattr(state, "use_norm_scale", True):
        deq_avg = deq_avg / scale

    # 安全检查 3：反量化并求平均后的 shard 也必须是有限值。
    # 若检测到非有限值，附带打印相关的指数范围信息，便于诊断是哪一侧的 k 仍然过大。
    if not torch.isfinite(deq_avg).all():
        try:
            with torch.no_grad():
                abs_g = g.abs()
                # 避免 log2(0)
                alpha = torch.log2(abs_g.clamp(min=1e-12))
                finite_alpha = alpha[torch.isfinite(alpha)]
                if finite_alpha.numel() > 0:
                    max_k_before = float(finite_alpha.max().item())
                    min_k_before = float(finite_alpha.min().item())
                else:
                    max_k_before = float("nan")
                    min_k_before = float("nan")

                alpha_clamped = alpha.clamp(min=float(_K_MIN), max=float(_K_MAX))
                max_k_after = float(alpha_clamped.max().item())
                min_k_after = float(alpha_clamped.min().item())

                bad_mask = ~torch.isfinite(deq_avg)
                bad_count = int(bad_mask.sum().item())

                logger.error(
                    "[nc] non-finite dequantized shard detected: bad_count=%s, "
                    "k_before in [%.3f, %.3f], k_after in [%.3f, %.3f]",
                    bad_count,
                    min_k_before,
                    max_k_before,
                    min_k_after,
                    max_k_after,
                )
        except Exception as log_exc:
            logger.error("[nc] failed to log debug info for non-finite dequantization: %s", log_exc)

        raise RuntimeError(
            "[nc] detected non-finite value after NC dequantization; aborting this training run"
        )

    shard_out.copy_(deq_avg)

    if state.error_feedback:
        _apply_error_feedback(state, full_flat_grad, shard_out, g_for_ef, pg)


__all__ = ["NCState", "fsdp_nc_comm_hook"]


