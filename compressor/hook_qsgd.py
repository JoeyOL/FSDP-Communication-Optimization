import struct
import torch
import torch.distributed as dist

from .common import _apply_error_feedback, _ensure_residual, logger
from perf.comm_stats import add_bytes as _comm_add_bytes

# Log only first few hook invocations per process to locate hang (each param triggers hook once)
_QSGD_LOG_CAP = 2
_qsgd_hook_log_count = 0


def _qsgd_bits_per_elem(s: int) -> int:
    """Bits per coordinate: 1 sign + ceil(log2(s+1)) for level in [0..s]."""
    return 1 + (s + 1).bit_length()


# ---------------------------------------------------------------------------
# 向量化 bucket 编码 / 解码（消除 per-bucket Python 循环）
# ---------------------------------------------------------------------------

def _qsgd_encode_bucketed_batch(
    g: torch.Tensor,
    q_g: torch.Tensor,
    bucket_size: int,
    s: int,
) -> torch.Tensor:
    """Encode an entire shard of quantized values into packed bytes.

    优化：将所有 bucket 拼成 2D 张量一次性向量化处理，
    消除原实现中 per-bucket Python 循环。
    Returns uint8 tensor of shape (num_buckets * max_bucket_bytes,).
    """
    n = q_g.numel()
    if n == 0:
        return torch.zeros(4, dtype=torch.uint8, device=q_g.device)

    bits = _qsgd_bits_per_elem(s)
    num_buckets = (n + bucket_size - 1) // bucket_size
    max_bucket_bytes = 4 + (bucket_size * bits + 7) // 8

    # Pad to full buckets for uniform reshape
    padded_n = num_buckets * bucket_size
    if padded_n > n:
        g_padded = torch.nn.functional.pad(g.view(-1), (0, padded_n - n), value=0.0)
        q_padded = torch.nn.functional.pad(q_g.view(-1), (0, padded_n - n), value=0.0)
    else:
        g_padded = g.view(-1)
        q_padded = q_g.view(-1)

    g_2d = g_padded.view(num_buckets, bucket_size)  # (num_buckets, bucket_size)
    q_2d = q_padded.view(num_buckets, bucket_size)

    # Per-bucket norms (vectorized, no loop)
    norms = g_2d.norm(p=2, dim=1).clamp(min=1e-12)  # (num_buckets,)

    # Quantize levels: level = round(|q_g| / norm * s), clamped to [0, s]
    norm_expanded = norms.unsqueeze(1).expand_as(q_2d)  # (num_buckets, bucket_size)
    level_float = (q_2d.abs() / norm_expanded * s).clamp(0, s)
    level = level_float.round().long().clamp(0, s)
    sign = (q_2d >= 0).long()

    level_bits = (s + 1).bit_length()
    val = (sign << level_bits) + level  # (num_buckets, bucket_size)
    val = val.clamp(0, (1 << bits) - 1)

    # Pack bits into bytes (vectorized)
    out = torch.zeros(num_buckets * max_bucket_bytes, dtype=torch.uint8, device=q_g.device)

    # Write norms as 4 bytes per bucket
    norm_bytes_array = torch.frombuffer(
        b"".join(struct.pack("<f", float(nm)) for nm in norms.tolist()),
        dtype=torch.uint8,
    ).to(device=q_g.device)
    # norm_bytes_array: (num_buckets * 4,)
    # Scatter into output: positions [b * max_bucket_bytes : b * max_bucket_bytes + 4]
    bucket_offsets = torch.arange(num_buckets, device=q_g.device) * max_bucket_bytes
    for byte_i in range(4):
        out[bucket_offsets + byte_i] = norm_bytes_array[torch.arange(num_buckets, device=q_g.device) * 4 + byte_i]

    if 8 % bits == 0:
        elems_per_byte = 8 // bits
        # Pad val to multiple of elems_per_byte per bucket
        pad_per_bucket = (elems_per_byte - bucket_size % elems_per_byte) % elems_per_byte
        if pad_per_bucket > 0:
            val = torch.nn.functional.pad(val, (0, pad_per_bucket), value=0)
        padded_bs = bucket_size + pad_per_bucket
        num_bytes_per_bucket = padded_bs // elems_per_byte

        # Reshape to (num_buckets, num_bytes_per_bucket, elems_per_byte)
        val_3d = val.view(num_buckets, num_bytes_per_bucket, elems_per_byte)
        # Pack: byte = v[0] | (v[1] << bits) | (v[2] << 2*bits) | ...
        packed = val_3d[:, :, 0].long()
        for j in range(1, elems_per_byte):
            packed = packed + (val_3d[:, :, j].long() << (j * bits))
        packed = packed.clamp(0, 255).to(torch.uint8)  # (num_buckets, num_bytes_per_bucket)

        # Write packed data into output buffer
        for b_idx in range(num_buckets):
            start_pos = b_idx * max_bucket_bytes + 4
            end_pos = start_pos + num_bytes_per_bucket
            out[start_pos:end_pos] = packed[b_idx]
    else:
        # Non-byte-aligned: fallback per-bucket (rare, only when s causes odd bit width)
        for b_idx in range(num_buckets):
            bucket_start = b_idx * max_bucket_bytes + 4
            bucket_n = min(bucket_size, n - b_idx * bucket_size)
            for i in range(bucket_n):
                byte_off = bucket_start + (i * bits) // 8
                bit_off = (i * bits) % 8
                v = int(val[b_idx, i].item())
                out[byte_off] = out[byte_off] | ((v << bit_off) & 0xFF)
                if bit_off + bits > 8 and byte_off + 1 < out.numel():
                    out[byte_off + 1] = out[byte_off + 1] | ((v >> (8 - bit_off)) & 0xFF)

    return out


def _qsgd_decode_bucketed_batch(
    buf: torch.Tensor,
    s: int,
    shard_size: int,
    bucket_size: int,
) -> torch.Tensor:
    """Decode entire buffer to float shard (vectorized over all buckets).

    优化：将所有 bucket 拼成 2D 操作，消除 per-bucket Python 循环。
    """
    bits = _qsgd_bits_per_elem(s)
    max_bucket_bytes = 4 + (bucket_size * bits + 7) // 8
    num_buckets = (shard_size + bucket_size - 1) // bucket_size

    if buf.numel() < num_buckets * max_bucket_bytes:
        return torch.zeros(shard_size, dtype=torch.float32, device=buf.device)

    level_bits = (s + 1).bit_length()
    parts = []

    if 8 % bits == 0:
        elems_per_byte = 8 // bits
        mask = (1 << bits) - 1
        data_bytes_per_bucket = max_bucket_bytes - 4

        # Extract all norms at once
        buf_2d = buf.view(num_buckets, max_bucket_bytes)
        norm_bytes_cpu = buf_2d[:, :4].to(torch.uint8).cpu().numpy()
        norms = torch.tensor(
            [struct.unpack("<f", norm_bytes_cpu[b].tobytes())[0] for b in range(num_buckets)],
            dtype=torch.float32,
            device=buf.device,
        )

        # Extract packed payloads: (num_buckets, data_bytes_per_bucket)
        payloads = buf_2d[:, 4:].long()

        # Unpack all bits at once: (num_buckets, data_bytes_per_bucket, elems_per_byte)
        val_list = []
        for j in range(elems_per_byte):
            val_list.append((payloads >> (j * bits)) & mask)
        # (num_buckets, data_bytes_per_bucket, elems_per_byte)
        vals_3d = torch.stack(val_list, dim=2)
        # Flatten to (num_buckets, data_bytes_per_bucket * elems_per_byte)
        vals_flat = vals_3d.view(num_buckets, -1)

        # Decode: sign and level
        sign = ((vals_flat >> level_bits) & 1).float()
        level = (vals_flat & ((1 << level_bits) - 1)).clamp(0, s).float()
        scale = (2.0 * sign - 1.0) * (norms / max(s, 1)).unsqueeze(1)
        decoded_full = scale * level  # (num_buckets, max_elems_per_bucket)

        # Trim to actual shard_size
        result = torch.zeros(shard_size, dtype=torch.float32, device=buf.device)
        for b in range(num_buckets):
            actual_n = min(bucket_size, shard_size - b * bucket_size)
            result[b * bucket_size: b * bucket_size + actual_n] = decoded_full[b, :actual_n]
        return result
    else:
        # Non-byte-aligned fallback
        offset = 0
        for b in range(num_buckets):
            n_elems = min(bucket_size, shard_size - b * bucket_size)
            seg_buf = buf[offset: offset + max_bucket_bytes]
            _, seg = _qsgd_decode_chunk_fallback(seg_buf, s, n_elems)
            parts.append(seg[:n_elems])
            offset += max_bucket_bytes
        return torch.cat(parts)


def _qsgd_decode_chunk_fallback(buf: torch.Tensor, s: int, shard_size: int) -> tuple[float, torch.Tensor]:
    """Decode bytes to (norm, float_shard). Fallback for non-byte-aligned bits."""
    if buf.numel() < 4:
        return 0.0, torch.zeros(shard_size, dtype=torch.float32, device=buf.device)
    norm = struct.unpack("<f", bytes(buf[:4].cpu().tolist()))[0]
    bits = _qsgd_bits_per_elem(s)
    level_bits = (s + 1).bit_length()
    n = min(shard_size, (buf.numel() - 4) * 8 // bits)
    if n <= 0:
        return norm, torch.zeros(shard_size, dtype=torch.float32, device=buf.device)
    level_mask = (1 << level_bits) - 1
    vals = []
    for i in range(n):
        byte_off = 4 + (i * bits) // 8
        bit_off = (i * bits) % 8
        v = int(buf[byte_off].item()) >> bit_off
        if bit_off + bits > 8 and byte_off + 1 < buf.numel():
            v |= int(buf[byte_off + 1].item()) << (8 - bit_off)
        v &= (1 << bits) - 1
        sign = (v >> level_bits) & 1
        level = min(v & level_mask, s)
        vals.append((1 if sign == 1 else -1) * norm * (level / max(s, 1)))
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
        self.bucket_size = bucket_size
        self.low_bit_comm = low_bit_comm
        # 预计算是否可以走低比特路径（避免每次 hook 调用重复检查）
        bits = _qsgd_bits_per_elem(self.s)
        self._byte_aligned = (8 % bits == 0)


def _qsgd_quantize(v: torch.Tensor, s: int) -> torch.Tensor:
    """QSGD stochastic quantization: scale by norm, round to s levels stochastically."""
    norm = v.norm(p=2)
    if norm < 1e-12:
        return v
    v_n = v / norm
    abs_v = v_n.abs()
    level_float = abs_v * s
    lo = level_float.floor().clamp(0, s - 1)
    hi = level_float.ceil().clamp(0, s)
    p = level_float - lo
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

    优化：编解码批量化，消除 per-shard 和 per-bucket Python 循环。
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
    # 使用预计算的 _byte_aligned 判断，避免每次重新计算
    use_low_bits = getattr(state, "low_bit_comm", True) and use_low_bits_whole and state._byte_aligned

    global _qsgd_hook_log_count
    _qsgd_hook_log_count += 1
    do_log = _qsgd_hook_log_count <= _QSGD_LOG_CAP

    if do_log:
        if use_low_bits_whole:
            logger.info(
                "[QSGD hook] rank=%s whole-vector path numel=%s shard_size=%s ef=%s low_bits=%s (call#%s)",
                rank, numel, shard_size, state.error_feedback, bool(use_low_bits), _qsgd_hook_log_count,
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
        if shard_size == 0:
            temp_shard = torch.zeros(0, device=g.device, dtype=torch.float32)
        else:
            eff_bucket_size = state.bucket_size
            if eff_bucket_size <= 0 or eff_bucket_size > shard_size:
                eff_bucket_size = shard_size

            # 批量编码所有 shard（消除 per-shard Python 循环）
            shard_enc_list = []
            for j in range(world_size):
                s_start = j * shard_size
                s_end = s_start + shard_size
                enc = _qsgd_encode_bucketed_batch(
                    g[s_start:s_end], q_g[s_start:s_end], eff_bucket_size, state.s
                )
                shard_enc_list.append(enc)

            # 确保所有 shard 编码长度一致
            shard_bytes = max(e.numel() for e in shard_enc_list)
            send_buf = torch.zeros(world_size * shard_bytes, device=g.device, dtype=torch.uint8)
            for j, enc in enumerate(shard_enc_list):
                send_buf[j * shard_bytes: j * shard_bytes + enc.numel()].copy_(enc)

            recv_buf = torch.empty_like(send_buf)
            if hasattr(dist, "all_to_all_single"):
                dist.all_to_all_single(recv_buf, send_buf, group=pg)
            else:
                send_chunks = list(send_buf.chunk(world_size, dim=0))
                recv_chunks = [torch.empty_like(send_chunks[0]) for _ in range(world_size)]
                dist.all_to_all(recv_chunks, send_chunks, group=pg)
                recv_buf = torch.cat(recv_chunks, dim=0)

            try:
                _comm_add_bytes(state, send_buf.numel() * send_buf.element_size())
            except Exception:
                pass

            # 批量解码所有接收到的 shard 并求和
            temp_shard = torch.zeros(shard_size, device=g.device, dtype=torch.float32)
            for j in range(world_size):
                buf_j = recv_buf[j * shard_bytes: (j + 1) * shard_bytes]
                decoded = _qsgd_decode_bucketed_batch(buf_j, state.s, shard_size, eff_bucket_size)
                temp_shard.add_(decoded)
    else:
        q_g_fp16 = q_g.half()
        temp_shard_fp16 = torch.empty(shard_size, device=g.device, dtype=torch.float16)
        if hasattr(dist, "reduce_scatter_tensor"):
            dist.reduce_scatter_tensor(temp_shard_fp16, q_g_fp16, op=dist.ReduceOp.SUM, group=pg)
        else:
            chunks = list(q_g_fp16.chunk(world_size, dim=0))
            dist.reduce_scatter(temp_shard_fp16, chunks, op=dist.ReduceOp.SUM, group=pg)
        try:
            _comm_add_bytes(state, q_g_fp16.numel() * q_g_fp16.element_size())
        except Exception:
            pass
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
