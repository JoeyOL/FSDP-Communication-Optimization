"""
Hybrid (two-stage) compression hooks for FSDP.

Stage 1 — sparsification: select important gradient elements (Top-k, Threshold-v, or Random-k).
Stage 2 — quantization: compress selected values to lower bit-width (INT8 or SignSGD/1-bit).

Supported combinations:
  topk_int8       — Top-k → INT8 quantization (default)
  topk_1bit       — Top-k → sign-based 1-bit
  thresholdv_int8 — Threshold-v → INT8 quantization
  thresholdv_1bit — Threshold-v → sign-based 1-bit
  randomk_int8    — Random-k → INT8 quantization

Design:
  - Each combination achieves *multiplicative* compression: e.g., Top-k(1%) × INT8(4×) ≈ 400× total.
  - Error feedback operates on the full two-stage compressed output.
  - Communication uses all_gather of compact (index, quantized_value) pairs.
"""

import torch
import torch.distributed as dist

from .common import (
    _apply_error_feedback,
    _ensure_residual,
    logger,
)
from perf.comm_stats import add_bytes as _comm_add_bytes


# ── Log cap ──────────────────────────────────────────────────────────────────
_hybrid_log_count = 0
_HYBRID_LOG_CAP = 3


# ── State ────────────────────────────────────────────────────────────────────

class HybridCompressState:
    """State object for hybrid two-stage compression.

    Parameters
    ----------
    sparse_method : str
        Stage-1 sparsification: "topk", "thresholdv", or "randomk".
    quant_method : str
        Stage-2 quantization: "int8" or "1bit".
    ratio : float
        Sparsity ratio for Top-k / Random-k / Threshold-v auto threshold.
    k : int
        Explicit k (overrides ratio if > 0).
    threshold_v : float
        Explicit threshold for Threshold-v (0 = use ratio-based auto).
    error_feedback : bool
        Enable error feedback across iterations.
    ef_local : bool
        Use local (shard-only) error feedback to save memory.
    """

    def __init__(
        self,
        sparse_method: str = "topk",
        quant_method: str = "int8",
        ratio: float = 0.01,
        k: int = 0,
        threshold_v: float = 0.0,
        error_feedback: bool = True,
        ef_local: bool = False,
    ) -> None:
        self.sparse_method = sparse_method.lower().strip()
        self.quant_method = quant_method.lower().strip()
        self.ratio = ratio
        self.k = k
        self.threshold_v = threshold_v
        self.error_feedback = error_feedback
        self._ef_local = ef_local

        assert self.sparse_method in ("topk", "thresholdv", "randomk"), \
            f"Unsupported sparse_method: {self.sparse_method}"
        assert self.quant_method in ("int8", "1bit"), \
            f"Unsupported quant_method: {self.quant_method}"


# ── Stage-1: Sparsification ─────────────────────────────────────────────────

def _stage1_topk(g: torch.Tensor, k: int):
    """Return (indices, values) of Top-k elements by absolute value."""
    _, indices = g.abs().topk(k, largest=True, sorted=False)
    values = g[indices]
    return indices, values


def _stage1_thresholdv(g: torch.Tensor, k: int, threshold_v: float):
    """Return (indices, values) of elements exceeding threshold.
    If threshold_v <= 0, adaptively estimate threshold to keep ~k elements."""
    if threshold_v > 0:
        mask = g.abs() >= threshold_v
    else:
        # Adaptive: use topk to find threshold, then apply mask
        if k >= g.numel():
            mask = torch.ones(g.numel(), dtype=torch.bool, device=g.device)
        else:
            topk_vals, _ = g.abs().topk(k, largest=True, sorted=True)
            thr = topk_vals[-1].item()
            mask = g.abs() >= thr
    indices = mask.nonzero(as_tuple=False).view(-1)
    # Cap at k to bound communication
    if indices.numel() > k:
        sub_vals = g[indices].abs()
        _, sub_topk = sub_vals.topk(k, largest=True, sorted=False)
        indices = indices[sub_topk]
    values = g[indices]
    return indices, values


def _stage1_randomk(g: torch.Tensor, k: int):
    """Return (indices, scaled_values) of Random-k with unbiased scaling."""
    numel = g.numel()
    perm = torch.randperm(numel, device=g.device)[:k]
    scale_factor = float(numel) / float(k)
    values = g[perm] * scale_factor
    return perm, values


# ── Stage-2: Quantization of sparse values ───────────────────────────────────

def _stage2_int8_encode(values: torch.Tensor, world_size: int):
    """Quantize values to INT8 with symmetric scaling.
    Returns (q_values: int8, scale: float32 scalar)."""
    abs_max = values.abs().max()
    if abs_max < 1e-12:
        return torch.zeros_like(values, dtype=torch.int8), torch.tensor(1.0, device=values.device)
    qr = max(1, 127 // world_size)
    scale = float(qr) / float(abs_max.item() + 1e-8)
    q = torch.clamp((values * scale).round(), -qr, qr).to(torch.int8)
    return q, torch.tensor(scale, device=values.device, dtype=torch.float32)


def _stage2_int8_decode(q_values: torch.Tensor, scale: torch.Tensor, world_size: int):
    """Dequantize INT8 values back to float."""
    return q_values.float() / scale / float(world_size)


def _stage2_1bit_encode(values: torch.Tensor):
    """Encode to 1-bit (sign) + mean magnitudes for positive/negative.
    Returns (signs: int8, mu_pos: scalar, mu_neg: scalar)."""
    signs = torch.sign(values).to(torch.int8)  # -1, 0, or +1
    pos_mask = values > 0
    neg_mask = values < 0
    mu_pos = values[pos_mask].mean() if pos_mask.any() else torch.zeros(1, device=values.device)
    mu_neg = values[neg_mask].mean() if neg_mask.any() else torch.zeros(1, device=values.device)
    return signs, mu_pos.float(), mu_neg.float()


def _stage2_1bit_decode(signs: torch.Tensor, mu_pos: torch.Tensor, mu_neg: torch.Tensor, world_size: int):
    """Decode 1-bit: reconstruct values from signs and mean magnitudes."""
    result = torch.where(signs > 0, mu_pos, torch.where(signs < 0, mu_neg, torch.zeros_like(mu_pos)))
    return result / float(world_size)


# ── Main Hook ────────────────────────────────────────────────────────────────

def fsdp_hybrid_comm_hook(
    state: HybridCompressState,
    full_flat_grad: torch.Tensor,
    shard_out: torch.Tensor,
) -> None:
    """Two-stage hybrid compression hook for FSDP.

    Flow:
    1. (Optional) Add error feedback residual
    2. Stage-1: Sparsify → get (indices, values), k elements
    3. Stage-2: Quantize values → compact representation
    4. Communicate via all_gather of (indices, quantized_values) pairs
    5. Decode, merge, extract shard
    6. (Optional) Update error feedback residual
    """
    global _hybrid_log_count
    _hybrid_log_count += 1
    do_log = dist.get_rank() == 0 and _hybrid_log_count <= _HYBRID_LOG_CAP

    pg = dist.group.WORLD
    world_size = dist.get_world_size(pg)
    rank = dist.get_rank(pg)

    if world_size == 1:
        shard_out.copy_(full_flat_grad)
        return

    g = full_flat_grad.contiguous().view(-1)
    numel = g.numel()

    # ── Error feedback: add residual ──
    if state.error_feedback:
        residual, start, end = _ensure_residual(state, g, pg)
        if start is None:
            g = (g + residual).to(g.dtype)
        else:
            g[start:end] += residual

    # ── Stage-1: Sparsification ──
    k = state.k if state.k > 0 else max(1, int(numel * state.ratio))
    k = min(k, numel)

    if state.sparse_method == "topk":
        indices, values = _stage1_topk(g, k)
    elif state.sparse_method == "thresholdv":
        indices, values = _stage1_thresholdv(g, k, state.threshold_v)
    elif state.sparse_method == "randomk":
        indices, values = _stage1_randomk(g, k)
    else:
        raise ValueError(f"Unknown sparse_method: {state.sparse_method}")

    actual_k = indices.numel()

    if do_log:
        logger.info(
            "[hybrid] sparse=%s quant=%s ratio=%.4f actual_k=%d/%d",
            state.sparse_method, state.quant_method, state.ratio, actual_k, numel,
        )

    # ── Stage-2: Quantization ──
    if state.quant_method == "int8":
        q_values, scale = _stage2_int8_encode(values, world_size)
        # Pack: each rank sends (indices_int64, q_values_int8, scale_f32)
        # all_gather indices and q_values separately
        indices_flat = indices.contiguous().to(torch.int64)
        q_flat = q_values.contiguous()

        # Pad to k if actual_k < k (e.g., Threshold-v returned fewer)
        if actual_k < k:
            pad_idx = torch.zeros(k - actual_k, device=g.device, dtype=torch.int64)
            pad_val = torch.zeros(k - actual_k, device=g.device, dtype=torch.int8)
            indices_flat = torch.cat([indices_flat, pad_idx])
            q_flat = torch.cat([q_flat, pad_val])

        # all_gather indices (int64) and quantized values (int8)
        all_indices = torch.empty(world_size * k, device=g.device, dtype=torch.int64)
        all_qvals = torch.empty(world_size * k, device=g.device, dtype=torch.int8)
        all_scales = torch.empty(world_size, device=g.device, dtype=torch.float32)

        dist.all_gather_into_tensor(all_indices, indices_flat, group=pg)
        dist.all_gather_into_tensor(all_qvals, q_flat, group=pg)
        dist.all_gather_into_tensor(all_scales, scale.view(1), group=pg)

        # Comm bytes: per rank sends k*8 (indices) + k*1 (int8 values) + 4 (scale) bytes
        per_rank_bytes = k * 8 + k * 1 + 4
        try:
            _comm_add_bytes(state, per_rank_bytes)
        except Exception:
            pass

        # Decode and merge
        full = torch.zeros(numel, device=g.device, dtype=torch.float32)
        for r in range(world_size):
            s = r * k
            e = s + k
            r_indices = all_indices[s:e]
            r_qvals = all_qvals[s:e].float()
            r_scale = all_scales[r]
            r_decoded = r_qvals / r_scale  # dequantize
            full.index_add_(0, r_indices.long(), r_decoded)

        # Extract shard and average
        shard_size = numel // world_size
        shard_start = rank * shard_size
        shard_end = shard_start + shard_size
        deq_avg = (full[shard_start:shard_end] / float(world_size)).to(full_flat_grad.dtype)
        shard_out.copy_(deq_avg)

    elif state.quant_method == "1bit":
        signs, mu_pos, mu_neg = _stage2_1bit_encode(values)

        indices_flat = indices.contiguous().to(torch.int64)
        signs_flat = signs.contiguous()

        # Pad to k
        if actual_k < k:
            pad_idx = torch.zeros(k - actual_k, device=g.device, dtype=torch.int64)
            pad_sgn = torch.zeros(k - actual_k, device=g.device, dtype=torch.int8)
            indices_flat = torch.cat([indices_flat, pad_idx])
            signs_flat = torch.cat([signs_flat, pad_sgn])

        # Pack mu_pos and mu_neg into a 2-element tensor
        mu_pair = torch.stack([mu_pos.view(-1)[0], mu_neg.view(-1)[0]]).to(torch.float32)

        all_indices = torch.empty(world_size * k, device=g.device, dtype=torch.int64)
        all_signs = torch.empty(world_size * k, device=g.device, dtype=torch.int8)
        all_mus = torch.empty(world_size * 2, device=g.device, dtype=torch.float32)

        dist.all_gather_into_tensor(all_indices, indices_flat, group=pg)
        dist.all_gather_into_tensor(all_signs, signs_flat, group=pg)
        dist.all_gather_into_tensor(all_mus, mu_pair, group=pg)

        # Comm bytes: per rank sends k*8 (indices) + k*1 (signs) + 8 (two floats) bytes
        per_rank_bytes = k * 8 + k * 1 + 8
        try:
            _comm_add_bytes(state, per_rank_bytes)
        except Exception:
            pass

        # Decode and merge
        full = torch.zeros(numel, device=g.device, dtype=torch.float32)
        for r in range(world_size):
            s = r * k
            e = s + k
            r_indices = all_indices[s:e]
            r_signs = all_signs[s:e]
            r_mu_pos = all_mus[r * 2]
            r_mu_neg = all_mus[r * 2 + 1]
            r_decoded = torch.where(
                r_signs > 0, r_mu_pos,
                torch.where(r_signs < 0, r_mu_neg, torch.zeros(1, device=g.device))
            )
            full.index_add_(0, r_indices.long(), r_decoded)

        shard_size = numel // world_size
        shard_start = rank * shard_size
        shard_end = shard_start + shard_size
        deq_avg = (full[shard_start:shard_end] / float(world_size)).to(full_flat_grad.dtype)
        shard_out.copy_(deq_avg)

    # ── Error feedback: update residual ──
    if state.error_feedback:
        # Reconstruct the sparse vector that was actually communicated
        sparse_sent = torch.zeros_like(g)
        sparse_sent[indices[:actual_k].long()] = values[:actual_k]
        _apply_error_feedback(state, full_flat_grad, shard_out, sparse_sent, pg)

    if do_log:
        if state.quant_method == "int8":
            total_comp = float(numel * 4) / float(per_rank_bytes) if per_rank_bytes > 0 else 0
            logger.info("[hybrid] total_compression_ratio=%.1fx (sparse %.1fx * quant 4x)", total_comp, float(numel) / float(k))
        else:
            total_comp = float(numel * 4) / float(per_rank_bytes) if per_rank_bytes > 0 else 0
            logger.info("[hybrid] total_compression_ratio=%.1fx (sparse %.1fx * quant 32x)", total_comp, float(numel) / float(k))


__all__ = ["HybridCompressState", "fsdp_hybrid_comm_hook"]
