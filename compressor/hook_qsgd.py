import struct
import torch
import torch.distributed as dist

from .common import _apply_error_feedback, _ensure_residual, logger

# Log only first few hook invocations per process to locate hang (each param triggers hook once)
_QSGD_LOG_CAP = 2
_qsgd_hook_log_count = 0


def _qsgd_bits_per_elem(s: int) -> int:
    """Bits per coordinate: 1 sign + ceil(log2(s+1)) for level in [0..s]."""
    return 1 + (s + 1).bit_length()


def _qsgd_encode_chunk(norm: float, q_chunk: torch.Tensor, s: int) -> torch.Tensor:
    """Encode one chunk: 4 bytes norm (float32) + packed (sign 1b, level) per element.
    Returns uint8 tensor of shape (4 + chunk_bytes,). Vectorized on GPU to avoid .item() sync.
    """
    n = q_chunk.numel()
    if n == 0:
        return torch.zeros(4, dtype=torch.uint8, device=q_chunk.device)
    bits = _qsgd_bits_per_elem(s)
    chunk_bytes = (n * bits + 7) // 8
    out = torch.empty(4 + chunk_bytes, dtype=torch.uint8, device=q_chunk.device)
    norm_bytes = struct.pack("<f", float(norm))
    for i in range(4):
        out[i].fill_(norm_bytes[i])
    norm_t = max(float(norm), 1e-12)
    level_float = (q_chunk.abs() / norm_t * s).clamp(0, s)
    level = level_float.round().long().clamp(0, s)
    sign = (q_chunk >= 0).long()
    level_bits = (s + 1).bit_length()
    val = (sign << level_bits) + level
    val = val.clamp(0, (1 << bits) - 1)
    # Vectorized pack (no Python loop / .item() to avoid GPU sync hang)
    if 8 % bits == 0:
        elems_per_byte = 8 // bits
        # Pad to multiple of elems_per_byte
        pad = (elems_per_byte - n % elems_per_byte) % elems_per_byte
        if pad > 0:
            val = torch.nn.functional.pad(val, (0, pad), value=0)
        # val[0], val[1] -> byte 0; val[2], val[3] -> byte 1; ...
        num_bytes = val.numel() // elems_per_byte
        packed = val[::elems_per_byte].clamp(0, 255).long()
        for j in range(1, elems_per_byte):
            packed = packed + (val[j::elems_per_byte].clamp(0, 255).long() << (j * bits))
        out[4 : 4 + num_bytes].copy_(packed.to(torch.uint8))
    else:
        # Non-byte-aligned: fallback with minimal .item() - only for small n or use torch ops
        out[4:].zero_()
        for i in range(n):
            byte_off = 4 + (i * bits) // 8
            bit_off = (i * bits) % 8
            v = int(val[i].item())
            out[byte_off] = out[byte_off] | ((v << bit_off) & 0xFF)
            if bit_off + bits > 8 and byte_off + 1 < out.numel():
                out[byte_off + 1] = out[byte_off + 1] | ((v >> (8 - bit_off)) & 0xFF)
    return out


def _qsgd_encode_bucketed_chunk(
    g_chunk: torch.Tensor,
    q_chunk: torch.Tensor,
    bucket_size: int,
    s: int,
) -> torch.Tensor:
    """Encode a shard that may span multiple buckets: per-bucket (4 bytes norm + packed). Fixed stride per bucket for all_to_all."""
    n = q_chunk.numel()
    if n == 0:
        return torch.zeros(4, dtype=torch.uint8, device=q_chunk.device)
    bits = _qsgd_bits_per_elem(s)
    max_bucket_bytes = 4 + (bucket_size * bits + 7) // 8
    num_buckets = (n + bucket_size - 1) // bucket_size
    chunk_bytes = num_buckets * max_bucket_bytes
    out = torch.empty(chunk_bytes, dtype=torch.uint8, device=q_chunk.device)
    offset = 0
    for b in range(num_buckets):
        start = b * bucket_size
        end = min((b + 1) * bucket_size, n)
        seg = q_chunk[start:end]
        norm = max(g_chunk[start:end].norm(p=2).item(), 1e-12)
        enc = _qsgd_encode_chunk(norm, seg, s)
        out[offset : offset + enc.numel()].copy_(enc)
        if enc.numel() < max_bucket_bytes:
            out[offset + enc.numel() : offset + max_bucket_bytes].zero_()
        offset += max_bucket_bytes
    return out


def _qsgd_decode_bucketed_chunk(
    buf: torch.Tensor,
    s: int,
    shard_size: int,
    bucket_size: int,
) -> torch.Tensor:
    """Decode buffer produced by _qsgd_encode_bucketed_chunk to float shard."""
    bits = _qsgd_bits_per_elem(s)
    max_bucket_bytes = 4 + (bucket_size * bits + 7) // 8
    num_buckets = (shard_size + bucket_size - 1) // bucket_size
    parts = []
    offset = 0
    for b in range(num_buckets):
        n_elems = min(bucket_size, shard_size - b * bucket_size)
        seg_buf = buf[offset : offset + max_bucket_bytes]
        _, seg = _qsgd_decode_chunk(seg_buf, s, n_elems)
        parts.append(seg[:n_elems])
        offset += max_bucket_bytes
    return torch.cat(parts)


def _qsgd_decode_chunk(buf: torch.Tensor, s: int, shard_size: int) -> tuple[float, torch.Tensor]:
    """Decode bytes to (norm, float_shard). Vectorized for 8%%bits==0 to avoid Python loop over millions."""
    if buf.numel() < 4:
        return 0.0, torch.zeros(shard_size, dtype=torch.float32, device=buf.device)
    norm = struct.unpack("<f", bytes(buf[:4].cpu().tolist()))[0]
    bits = _qsgd_bits_per_elem(s)
    level_bits = (s + 1).bit_length()
    n = min(shard_size, (buf.numel() - 4) * 8 // bits)
    if n <= 0:
        return norm, torch.zeros(shard_size, dtype=torch.float32, device=buf.device)
    if 8 % bits == 0:
        elems_per_byte = 8 // bits
        mask = (1 << bits) - 1
        payload = buf[4:].long()
        # Unpack: byte b gives elems_per_byte values at (b >> (j*bits)) & mask for j=0..elems_per_byte-1
        val_list = []
        for j in range(elems_per_byte):
            val_list.append((payload >> (j * bits)) & mask)
        val = torch.stack(val_list, dim=1).flatten()[:n].to(buf.device)
        sign = ((val >> level_bits) & 1).float()
        level = (val & ((1 << level_bits) - 1)).clamp(0, s).float()
        scale = (1.0 - 2.0 * sign) * (norm / max(s, 1))
        shard = scale * level
    else:
        vals = []
        level_mask = (1 << level_bits) - 1
        for i in range(n):
            byte_off = 4 + (i * bits) // 8
            bit_off = (i * bits) % 8
            v = int(buf[byte_off].item()) >> bit_off
            if bit_off + bits > 8 and byte_off + 1 < buf.numel():
                v |= int(buf[byte_off + 1].item()) << (8 - bit_off)
            v &= (1 << bits) - 1
            sign = (v >> level_bits) & 1
            level = min(v & level_mask, s)
            vals.append((1 if sign == 0 else -1) * norm * (level / max(s, 1)))
        shard = torch.tensor(vals, dtype=torch.float32, device=buf.device)
    out = torch.zeros(shard_size, dtype=torch.float32, device=buf.device)
    out[: shard.numel()] = shard
    return norm, out


class QSGDState:
    def __init__(
        self,
        s: int = 4,
        error_feedback: bool = False,
        ef_local: bool = False,
        bucket_size: int = 0,
        low_bit_comm: bool = True,
    ) -> None:
        self.s = max(2, int(s))
        self.error_feedback = error_feedback
        self._ef_local = ef_local
        # bucket_size <= 0 or >= numel: whole vector one bucket; else per-bucket QSGD (paper Section 4)
        self.bucket_size = bucket_size
        # low_bit_comm: True = all_reduce(scale) + reduce_scatter(quantized float16), same collective as baseline, half data; False = reduce_scatter(float16)
        self.low_bit_comm = low_bit_comm


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
    """QSGD per bucket: each bucket gets independent L2 norm and s-level quantization. Vectorized (no Python loop)."""
    numel = g.numel()
    if bucket_size <= 0 or bucket_size >= numel:
        return _qsgd_quantize(g, s)
    num_buckets = (numel + bucket_size - 1) // bucket_size
    numel_padded = num_buckets * bucket_size
    g_padded = torch.nn.functional.pad(g.contiguous().view(-1), (0, numel_padded - numel), value=0.0)
    g_2d = g_padded.view(num_buckets, bucket_size)
    if numel % bucket_size != 0:
        g_2d = g_2d.clone()
        g_2d[-1, numel % bucket_size :] = 0.0
    norm_per_bucket = g_2d.norm(p=2, dim=1).clamp(min=1e-12)
    norm_expanded = norm_per_bucket.unsqueeze(1).expand(-1, bucket_size).reshape(-1)[:numel]
    g_n = g / norm_expanded
    abs_v = g_n.abs()
    level_float = abs_v * s
    lo = level_float.floor().clamp(0, s - 1)
    hi = level_float.ceil().clamp(0, s)
    p = level_float - lo
    u = torch.empty_like(g, device=g.device).uniform_(0, 1)
    lev = torch.where(u < p, hi, lo)
    q_abs = lev.float() / s
    q_n = q_abs * g_n.sign()
    return (q_n * norm_expanded).to(g.dtype)


def fsdp_qsgd_comm_hook(
    state: QSGDState,
    full_flat_grad: torch.Tensor,
    shard_out: torch.Tensor,
) -> None:
    """QSGD: s-level stochastic quantization, then reduce-scatter.

    low_bit_comm=True: all_reduce(1 float for scale) + reduce_scatter(quantized float16). Same collective
    as baseline, half the data, no all_to_all. low_bit_comm=False: reduce_scatter(float16).
    """
    pg = dist.group.WORLD
    world_size = dist.get_world_size(pg)
    rank = dist.get_rank(pg)

    if world_size == 1:
        shard_out.copy_(full_flat_grad)
        return

    numel = full_flat_grad.numel()
    shard_size = numel // world_size
    use_low_bits_whole = (state.bucket_size <= 0 or state.bucket_size >= numel) and (shard_size > 0)
    use_low_bits_bucketed = (0 < state.bucket_size < numel) and (shard_size > 0)
    use_low_bits = getattr(state, "low_bit_comm", True) and (use_low_bits_whole or use_low_bits_bucketed)

    global _qsgd_hook_log_count
    _qsgd_hook_log_count += 1
    do_log = _qsgd_hook_log_count <= _QSGD_LOG_CAP

    if do_log:
        if use_low_bits_whole:
            logger.info(
                "[QSGD hook] rank=%s whole-vector (low-bit) path numel=%s shard_size=%s ef=%s (call#%s)",
                rank, numel, shard_size, state.error_feedback, _qsgd_hook_log_count,
            )
        elif use_low_bits_bucketed:
            logger.info(
                "[QSGD hook] rank=%s bucketed (low-bit) path numel=%s bucket_size=%s shard_size=%s (call#%s)",
                rank, numel, state.bucket_size, shard_size, _qsgd_hook_log_count,
            )

    g = full_flat_grad.contiguous().view(-1)
    if state.error_feedback:
        if do_log:
            logger.info("[QSGD hook] rank=%s before error_feedback residual", rank)
        residual, start, end = _ensure_residual(state, g, pg)
        if start is None:
            g = (g + residual).to(g.dtype)
        else:
            g[start:end] += residual
        if do_log:
            logger.info("[QSGD hook] rank=%s after error_feedback residual", rank)

    if do_log:
        logger.info("[QSGD hook] rank=%s before quantize", rank)
    q_g = _qsgd_quantize_bucketed(g, state.s, state.bucket_size)
    if do_log:
        logger.info("[QSGD hook] rank=%s after quantize", rank)

    if use_low_bits:
        # reduce_scatter(quantized fp16) + one all_reduce(scale). NCCL does not support int16; use float16 for same 2-byte comm.
        norm_val = max(g.norm(p=2).item(), 1e-12)
        norm_t = torch.tensor([norm_val], device=g.device, dtype=torch.float32)
        dist.all_reduce(norm_t, op=dist.ReduceOp.MAX, group=pg)
        global_max_norm = norm_t.item()
        scale = 32767.0 / (float(world_size) * max(global_max_norm, 1e-12))
        q_scaled = (q_g * scale).round().clamp(-32767, 32767).to(torch.float16)
        temp_fp16 = torch.empty(shard_size, device=g.device, dtype=torch.float16)
        if hasattr(dist, "reduce_scatter_tensor"):
            dist.reduce_scatter_tensor(temp_fp16, q_scaled, op=dist.ReduceOp.SUM, group=pg)
        else:
            chunks = list(q_scaled.chunk(world_size, dim=0))
            dist.reduce_scatter(temp_fp16, chunks, op=dist.ReduceOp.SUM, group=pg)
        temp_shard = temp_fp16.float() / scale
    else:
        # Float path: same collective as baseline. Use float16 reduce_scatter to cut comm volume in half (saves time when bandwidth-bound).
        q_g_fp16 = q_g.half()
        temp_shard_fp16 = torch.empty(shard_size, device=g.device, dtype=torch.float16)
        if hasattr(dist, "reduce_scatter_tensor"):
            dist.reduce_scatter_tensor(temp_shard_fp16, q_g_fp16, op=dist.ReduceOp.SUM, group=pg)
        else:
            chunks = list(q_g_fp16.chunk(world_size, dim=0))
            dist.reduce_scatter(temp_shard_fp16, chunks, op=dist.ReduceOp.SUM, group=pg)
        temp_shard = temp_shard_fp16.float()

    deq_avg = (temp_shard / float(world_size)).to(full_flat_grad.dtype)
    shard_out.copy_(deq_avg)

    if state.error_feedback:
        if do_log:
            logger.info("[QSGD hook] rank=%s before _apply_error_feedback (all_gather)", rank)
        _apply_error_feedback(state, full_flat_grad, shard_out, g, pg)
        if do_log:
            logger.info("[QSGD hook] rank=%s after _apply_error_feedback", rank)


__all__ = ["QSGDState", "fsdp_qsgd_comm_hook"]


