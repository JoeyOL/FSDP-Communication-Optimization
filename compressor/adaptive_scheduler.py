"""
Adaptive compression scheduler for FSDP.

Dynamically adjusts compression strength (sparsity ratio / quantization bits)
across training stages based on the training step or gradient statistics.

Three scheduling strategies:
  1. warmup_decay   — Linear warmup (light → aggressive) then cosine decay
                      (aggressive → moderate).  Intuition: early training needs
                      accurate gradients; mid-training tolerates more compression;
                      late training benefits from slightly higher fidelity for fine-tuning.
  2. grad_adaptive  — Monitor gradient L2 norm variance; increase compression when
                      gradients are stable, reduce compression when they are noisy.
  3. step_linear    — Simple linear interpolation from min_ratio to max_ratio over
                      the total training steps.

Design:
  - The scheduler wraps an inner sparsification hook (Top-k / Random-k / Threshold-v)
    and dynamically adjusts its `ratio` parameter each step.
  - Quantization (INT8) is always applied as Stage-2 when the inner hook is sparse.
  - Error feedback operates on the composite compressed output.
  - Compatible with the FSDP hook interface: (state, full_flat_grad, shard_out).
"""

import math
from typing import Any, Dict, Optional

import torch
import torch.distributed as dist

from .common import (
    _apply_error_feedback,
    _ensure_residual,
    logger,
)
from perf.comm_stats import add_bytes as _comm_add_bytes
from perf.adaptive_ratio_stats import add_sample as _adaptive_add_sample


# ── Log cap ──────────────────────────────────────────────────────────────────
_adaptive_log_count = 0
_ADAPTIVE_LOG_CAP = 10


# ── Scheduling Functions ────────────────────────────────────────────────────

def _schedule_warmup_decay(
    step: int,
    total_steps: int,
    warmup_fraction: float,
    min_ratio: float,
    max_ratio: float,
) -> float:
    """Warmup-then-cosine-decay schedule for sparsity ratio.

    Phase 1 (warmup, 0 ~ warmup_steps):
        ratio linearly increases from max_ratio (light compression, more elements kept)
        to min_ratio (aggressive compression, fewer elements kept).
        Rationale: early training needs exploration, large gradients carry useful signal.

    Phase 2 (decay, warmup_steps ~ total_steps):
        ratio follows cosine schedule from min_ratio back to a moderate level
        (midpoint of min_ratio and max_ratio).
        Rationale: late-stage fine-tuning benefits from slightly less aggressive compression.

    Note: Here ratio = fraction of elements KEPT (higher = less compression).
    """
    warmup_steps = int(total_steps * warmup_fraction)

    if step < warmup_steps:
        # Warmup: high ratio → low ratio (increasing compression)
        t = float(step) / max(1.0, float(warmup_steps))
        return max_ratio - t * (max_ratio - min_ratio)
    else:
        # Cosine decay: low ratio → moderate ratio (relaxing compression)
        remaining = total_steps - warmup_steps
        t = float(step - warmup_steps) / max(1.0, float(remaining))
        moderate_ratio = (min_ratio + max_ratio) / 2.0
        cos_val = 0.5 * (1.0 + math.cos(math.pi * t))
        return min_ratio + (moderate_ratio - min_ratio) * (1.0 - cos_val)


def _schedule_step_linear(
    step: int,
    total_steps: int,
    min_ratio: float,
    max_ratio: float,
) -> float:
    """Simple linear interpolation from max_ratio to min_ratio.

    Starts with light compression (high ratio = many elements kept) and
    linearly increases compression strength over training.
    """
    t = min(float(step) / max(1.0, float(total_steps)), 1.0)
    return max_ratio - t * (max_ratio - min_ratio)


def _schedule_grad_adaptive(
    step: int,
    grad_norm: float,
    grad_norm_ema: float,
    min_ratio: float,
    max_ratio: float,
) -> float:
    """Gradient-norm-adaptive schedule.

    If the current gradient norm is significantly larger than the EMA
    (indicating important gradient signal), use lighter compression (higher ratio).
    If the gradient is close to or below EMA, use heavier compression.

    Returns the adjusted sparsity ratio.
    """
    if grad_norm_ema < 1e-8:
        return max_ratio  # No history yet, be conservative

    relative = grad_norm / grad_norm_ema
    # Sigmoid-like mapping: relative >> 1 → high ratio, relative << 1 → low ratio
    # Clamp relative to [0.1, 10] for numerical stability
    relative = max(0.1, min(10.0, relative))
    # Map to [min_ratio, max_ratio]
    # When relative=1 → midpoint; relative>1 → closer to max; relative<1 → closer to min
    t = 1.0 / (1.0 + math.exp(-2.0 * (relative - 1.0)))  # sigmoid centered at 1
    return min_ratio + t * (max_ratio - min_ratio)


# ── State ────────────────────────────────────────────────────────────────────

class AdaptiveCompressState:
    """State for adaptive compression scheduling.

    Parameters
    ----------
    base_hook_name : str
        Inner compression hook type: "topk", "randomk", "thresholdv", or "hybrid".
    schedule : str
        Scheduling strategy: "warmup_decay", "step_linear", or "grad_adaptive".
    total_steps : int
        Total number of training steps (for schedule computation).
    warmup_fraction : float
        Fraction of total_steps used for warmup phase (warmup_decay only).
    min_ratio : float
        Minimum sparsity ratio (most aggressive compression).
    max_ratio : float
        Maximum sparsity ratio (lightest compression).
    error_feedback : bool
        Enable error feedback.
    ef_local : bool
        Use local error feedback.
    hook_kwargs : dict
        Additional keyword arguments forwarded to configure inner hooks.
    """

    def __init__(
        self,
        base_hook_name: str = "topk",
        schedule: str = "warmup_decay",
        total_steps: int = 1000,
        warmup_fraction: float = 0.1,
        min_ratio: float = 0.001,
        max_ratio: float = 0.1,
        error_feedback: bool = True,
        ef_local: bool = False,
        hook_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.base_hook_name = base_hook_name.lower().strip()
        self.schedule = schedule.lower().strip()
        self.total_steps = max(1, total_steps)
        self.warmup_fraction = warmup_fraction
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.error_feedback = error_feedback
        self._ef_local = ef_local
        self.hook_kwargs = hook_kwargs or {}

        # Step counter (incremented each hook call)
        self._step = 0

        # For grad_adaptive: exponential moving average of gradient L2 norm
        self._grad_norm_ema = 0.0
        self._ema_alpha = 0.1  # EMA smoothing factor

        assert self.base_hook_name in ("topk", "randomk", "thresholdv", "hybrid"), \
            f"Unsupported base_hook_name: {self.base_hook_name}"
        assert self.schedule in ("warmup_decay", "step_linear", "grad_adaptive"), \
            f"Unsupported schedule: {self.schedule}"


# ── Sparsification helpers (same as hybrid, reused for consistency) ──────────

def _sparse_topk(g: torch.Tensor, k: int):
    _, indices = g.abs().topk(k, largest=True, sorted=False)
    return indices, g[indices]


def _sparse_randomk(g: torch.Tensor, k: int):
    numel = g.numel()
    perm = torch.randperm(numel, device=g.device)[:k]
    scale_factor = float(numel) / float(k)
    return perm, g[perm] * scale_factor


def _sparse_thresholdv(g: torch.Tensor, k: int):
    if k >= g.numel():
        indices = torch.arange(g.numel(), device=g.device)
    else:
        topk_vals, _ = g.abs().topk(k, largest=True, sorted=True)
        thr = topk_vals[-1].item()
        mask = g.abs() >= thr
        indices = mask.nonzero(as_tuple=False).view(-1)
        if indices.numel() > k:
            sub_vals = g[indices].abs()
            _, sub_topk = sub_vals.topk(k, largest=True, sorted=False)
            indices = indices[sub_topk]
    return indices, g[indices]


def _int8_encode(values: torch.Tensor, world_size: int):
    abs_max = values.abs().max()
    if abs_max < 1e-12:
        return torch.zeros_like(values, dtype=torch.int8), torch.tensor(1.0, device=values.device)
    qr = max(1, 127 // world_size)
    scale = float(qr) / float(abs_max.item() + 1e-8)
    q = torch.clamp((values * scale).round(), -qr, qr).to(torch.int8)
    return q, torch.tensor(scale, device=values.device, dtype=torch.float32)


# ── Main Hook ────────────────────────────────────────────────────────────────

def fsdp_adaptive_comm_hook(
    state: AdaptiveCompressState,
    full_flat_grad: torch.Tensor,
    shard_out: torch.Tensor,
) -> None:
    """Adaptive compression hook for FSDP.

    Flow:
    1. Compute current step's gradient L2 norm (for grad_adaptive schedule)
    2. Determine current sparsity ratio via the scheduling function
    3. Apply sparsification at the computed ratio
    4. Apply INT8 quantization (Stage-2)
    5. Communicate via all_gather
    6. Decode, merge, extract shard
    7. Update step counter and EMA
    """
    global _adaptive_log_count
    _adaptive_log_count += 1
    do_log = dist.get_rank() == 0 and _adaptive_log_count <= _ADAPTIVE_LOG_CAP

    pg = dist.group.WORLD
    world_size = dist.get_world_size(pg)
    rank = dist.get_rank(pg)

    if world_size == 1:
        shard_out.copy_(full_flat_grad)
        state._step += 1
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

    # ── Compute gradient norm (for grad_adaptive) ──
    # Use a synchronized global norm proxy so all ranks derive identical ratio/k.
    grad_norm_local = float(g.norm().item())
    grad_norm = grad_norm_local
    if world_size > 1:
        gn = torch.tensor([grad_norm_local], device=g.device, dtype=torch.float32)
        dist.all_reduce(gn, op=dist.ReduceOp.SUM, group=pg)
        grad_norm = float((gn / float(world_size)).item())

    # ── Determine current ratio ──
    if state.schedule == "warmup_decay":
        current_ratio = _schedule_warmup_decay(
            state._step, state.total_steps,
            state.warmup_fraction, state.min_ratio, state.max_ratio,
        )
    elif state.schedule == "step_linear":
        current_ratio = _schedule_step_linear(
            state._step, state.total_steps,
            state.min_ratio, state.max_ratio,
        )
    elif state.schedule == "grad_adaptive":
        current_ratio = _schedule_grad_adaptive(
            state._step, grad_norm, state._grad_norm_ema,
            state.min_ratio, state.max_ratio,
        )
    else:
        current_ratio = state.max_ratio

    # Clamp ratio
    current_ratio = max(state.min_ratio, min(state.max_ratio, current_ratio))
    k = max(1, int(numel * current_ratio))
    k = min(k, numel)
    if world_size > 1:
        k_tensor = torch.tensor([k], device=g.device, dtype=torch.int64)
        dist.broadcast(k_tensor, src=0, group=pg)
        k = int(k_tensor.item())
        # Keep a consistent derived ratio for logging/recording after k sync.
        current_ratio = float(k) / float(numel)

    try:
        _adaptive_add_sample(
            state=state,
            comm_step=state._step,
            schedule=state.schedule,
            ratio_keep=current_ratio,
            k=k,
            numel=numel,
            grad_norm=grad_norm,
            ema_norm=state._grad_norm_ema,
        )
    except Exception:
        pass

    if do_log:
        logger.info(
            "[adaptive] step=%d/%d schedule=%s ratio=%.6f k=%d/%d base=%s grad_norm(local=%.4f,global=%.4f)",
            state._step, state.total_steps, state.schedule,
            current_ratio, k, numel, state.base_hook_name, grad_norm_local, grad_norm,
        )

    # ── Stage-1: Sparsification ──
    if state.base_hook_name == "topk" or state.base_hook_name == "hybrid":
        indices, values = _sparse_topk(g, k)
    elif state.base_hook_name == "randomk":
        indices, values = _sparse_randomk(g, k)
    elif state.base_hook_name == "thresholdv":
        indices, values = _sparse_thresholdv(g, k)
    else:
        indices, values = _sparse_topk(g, k)

    actual_k = indices.numel()

    # ── Stage-2: INT8 Quantization ──
    q_values, scale = _int8_encode(values, world_size)

    # Pad to k
    indices_flat = indices.contiguous().to(torch.int64)
    q_flat = q_values.contiguous()
    if actual_k < k:
        pad_idx = torch.zeros(k - actual_k, device=g.device, dtype=torch.int64)
        pad_val = torch.zeros(k - actual_k, device=g.device, dtype=torch.int8)
        indices_flat = torch.cat([indices_flat, pad_idx])
        q_flat = torch.cat([q_flat, pad_val])

    # ── Communication: all_gather ──
    all_indices = torch.empty(world_size * k, device=g.device, dtype=torch.int64)
    all_qvals = torch.empty(world_size * k, device=g.device, dtype=torch.int8)
    all_scales = torch.empty(world_size, device=g.device, dtype=torch.float32)

    dist.all_gather_into_tensor(all_indices, indices_flat, group=pg)
    dist.all_gather_into_tensor(all_qvals, q_flat, group=pg)
    dist.all_gather_into_tensor(all_scales, scale.view(1), group=pg)

    # Track communication bytes
    per_rank_bytes = k * 8 + k * 1 + 4  # indices(int64) + qvals(int8) + scale(float32)
    try:
        _comm_add_bytes(state, per_rank_bytes)
    except Exception:
        pass

    # ── Decode and merge ──
    full = torch.zeros(numel, device=g.device, dtype=torch.float32)
    for r in range(world_size):
        s = r * k
        e = s + k
        r_indices = all_indices[s:e]
        r_qvals = all_qvals[s:e].float()
        r_scale = all_scales[r]
        r_decoded = r_qvals / r_scale
        full.index_add_(0, r_indices.long(), r_decoded)

    # ── Extract shard ──
    shard_size = numel // world_size
    shard_start = rank * shard_size
    shard_end = shard_start + shard_size
    deq_avg = (full[shard_start:shard_end] / float(world_size)).to(full_flat_grad.dtype)
    shard_out.copy_(deq_avg)

    # ── Error feedback ──
    if state.error_feedback:
        sparse_sent = torch.zeros_like(g)
        sparse_sent[indices[:actual_k].long()] = values[:actual_k]
        _apply_error_feedback(state, full_flat_grad, shard_out, sparse_sent, pg)

    # ── Update step and EMA ──
    state._step += 1
    state._grad_norm_ema = (
        state._ema_alpha * grad_norm
        + (1.0 - state._ema_alpha) * state._grad_norm_ema
    )

    if do_log:
        total_comp = float(numel * 4) / float(per_rank_bytes) if per_rank_bytes > 0 else 0
        logger.info(
            "[adaptive] compression_ratio=%.1fx next_step=%d ema_norm=%.4f",
            total_comp, state._step, state._grad_norm_ema,
        )


__all__ = ["AdaptiveCompressState", "fsdp_adaptive_comm_hook"]
