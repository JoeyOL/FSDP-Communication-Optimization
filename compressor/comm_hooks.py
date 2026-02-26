"""
Thin registry for FSDP communication hooks.
具体算法的实现已经拆分到独立的 hook_*.py 文件中，这里只负责统一构建入口。
"""

from typing import Any, Callable, Optional, Tuple

import torch
import torch.distributed as dist

from perf.comm_stats import attach_to_state
from perf.grad_error_stats import attach_to_state as attach_grad_error_to_state
from perf.grad_error_stats import add_sample as _grad_add_sample

from .hook_fp16 import FP16State, fsdp_fp16_comm_hook
from .hook_hybrid_topk_int8 import HybridTopKInt8State, fsdp_hybrid_topk_int8_comm_hook
from .hook_int8 import GradQuantState, fsdp_quantized_comm_hook
from .hook_nc import NCState, fsdp_nc_comm_hook
from .hook_onebit_seide import OneBitSeideState, fsdp_onebit_seide_comm_hook
from .hook_qsgd import QSGDState, fsdp_qsgd_comm_hook
from .hook_randomk import RandomKState, fsdp_randomk_comm_hook
from .hook_signsgd import SignSGDState, fsdp_signsgd_comm_hook
from .hook_sketch import SketchState, fsdp_sketch_comm_hook
from .hook_thresholdv import ThresholdVState, fsdp_thresholdv_comm_hook
from .hook_topk import TopKState, fsdp_topk_comm_hook


# ---------------------------------------------------------------------------
# Helpers: wrap hooks with optional gradient error statistics
# ---------------------------------------------------------------------------


def _wrap_with_grad_error_stats(
    hook: Callable[[Any, torch.Tensor, torch.Tensor], None]
) -> Callable[[Any, torch.Tensor, torch.Tensor], None]:
    """
    在不改变原有 hook 接口的前提下，额外记录一次“压缩前后梯度的相对 L2 误差”。

    记 full_flat_grad 为当前 FSDP 单元在某个 rank 上的完整扁平梯度，shard_out 为通信
    完成后该 rank 上的梯度分片。本包装器在原 hook 执行完毕（即 shard_out 已被写入）
    后，按如下方式估计本 rank shard 上的压缩误差：

        diff = full_flat_grad[start:end] - shard_out
        rel_l2 = ||diff||_2 / ||full_flat_grad[start:end]||_2

    其中 [start, end) 为当前 rank 在 full_flat_grad 上对应的 shard 范围。为避免在误差记录
    上引入额外通信，该计算仅使用本 rank 数据，不做 all_gather。
    """

    def wrapped(state: Any, full_flat_grad: torch.Tensor, shard_out: torch.Tensor) -> None:
        # 先执行原 hook
        hook(state, full_flat_grad, shard_out)

        # 然后在本 rank shard 上估计一次相对 L2 误差
        try:
            if not dist.is_initialized():
                return
            pg = dist.group.WORLD
            world_size = dist.get_world_size(pg)
            rank = dist.get_rank(pg)
            if world_size <= 1:
                return

            g = full_flat_grad.contiguous().view(-1)
            numel = g.numel()
            if numel == 0 or numel % world_size != 0:
                return

            shard_size = numel // world_size
            start = rank * shard_size
            end = start + shard_size

            ref = g[start:end]
            out = shard_out.contiguous().view(-1)
            if out.numel() != ref.numel():
                return

            denom_norm = float(ref.norm().item()) if ref.numel() > 0 else 0.0
            if denom_norm <= 0.0:
                return

            diff = ref - out
            rel_l2 = float(diff.norm().item() / denom_norm)
            _grad_add_sample(state, rel_l2)
        except Exception:
            # 误差统计为附加信息，任何异常都不应影响正常训练流程
            pass

    return wrapped


# ---------------------------------------------------------------------------
# Build: name -> (state, hook) with optional kwargs
# ---------------------------------------------------------------------------

def build_comm_hook(
    name: str,
    num_bits: int = 8,
    int8_variant: str = "linear",
    error_feedback: bool = False,
    ef_local: bool = True,
    sparse_comm: bool = True,
    qsgd_s: int = 4,
    qsgd_bucket_size: int = 0,
    qsgd_low_bit_comm: bool = True,
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
    onebit_use_cpp: bool = True,
    signsgd_use_delta: bool = False,
    nc_use_bit_packing: bool = True,
    nc_use_norm_scale: bool = True,
    **kwargs: Any,
) -> Tuple[Optional[Any], Optional[Any]]:
    if name is None:
        return None, None

    n = name.lower().strip()
    # 兼容脚本中使用的 "threshold_v" 写法
    if n == "threshold_v":
        n = "thresholdv"
    if n == "none":
        return None, None

    ef = error_feedback

    if n == "int8":
        state = GradQuantState(num_bits=num_bits, error_feedback=ef, variant=int8_variant, ef_local=ef_local)
        attach_to_state(state, n)
        attach_grad_error_to_state(state, n)
        return state, _wrap_with_grad_error_stats(fsdp_quantized_comm_hook)
    if n == "fp16":
        state = FP16State(error_feedback=ef, ef_local=ef_local)
        attach_to_state(state, n)
        attach_grad_error_to_state(state, n)
        return state, _wrap_with_grad_error_stats(fsdp_fp16_comm_hook)
    if n == "qsgd":
        state = QSGDState(s=qsgd_s, error_feedback=ef, ef_local=ef_local, bucket_size=qsgd_bucket_size, low_bit_comm=qsgd_low_bit_comm)
        attach_to_state(state, n)
        attach_grad_error_to_state(state, n)
        return state, _wrap_with_grad_error_stats(fsdp_qsgd_comm_hook)
    if n == "signsgd" or n == "onebit":
        state = SignSGDState(error_feedback=ef, use_delta_scale=signsgd_use_delta, ef_local=ef_local)
        attach_to_state(state, n)
        attach_grad_error_to_state(state, n)
        return state, _wrap_with_grad_error_stats(fsdp_signsgd_comm_hook)
    if n == "onebit_seide":
        state = OneBitSeideState(
            col_size=onebit_col_size,
            error_feedback=ef,
            use_cpp=onebit_use_cpp,
            ef_local=ef_local,
        )
        attach_to_state(state, n)
        attach_grad_error_to_state(state, n)
        return state, _wrap_with_grad_error_stats(fsdp_onebit_seide_comm_hook)
    if n == "nc":
        state = NCState(error_feedback=ef, use_bit_packing=nc_use_bit_packing, ef_local=ef_local, use_norm_scale=nc_use_norm_scale)
        attach_to_state(state, n)
        attach_grad_error_to_state(state, n)
        return state, _wrap_with_grad_error_stats(fsdp_nc_comm_hook)
    if n == "topk":
        state = TopKState(k=topk_k, ratio=topk_ratio, error_feedback=ef, sparse_comm=sparse_comm, ef_local=ef_local)
        attach_to_state(state, n)
        attach_grad_error_to_state(state, n)
        return state, _wrap_with_grad_error_stats(fsdp_topk_comm_hook)
    if n == "randomk":
        state = RandomKState(k=randomk_k, ratio=randomk_ratio, error_feedback=ef, sparse_comm=sparse_comm, ef_local=ef_local)
        attach_to_state(state, n)
        attach_grad_error_to_state(state, n)
        return state, _wrap_with_grad_error_stats(fsdp_randomk_comm_hook)
    if n == "thresholdv":
        state = ThresholdVState(v=threshold_v, v_neg=threshold_v_neg, ratio=threshold_ratio, error_feedback=ef, sparse_comm=sparse_comm, ef_local=ef_local)
        attach_to_state(state, n)
        attach_grad_error_to_state(state, n)
        return state, _wrap_with_grad_error_stats(fsdp_thresholdv_comm_hook)
    if n == "sketch":
        state = SketchState(k=sketch_k, ratio=sketch_ratio, error_feedback=ef, two_round=sketch_two_round, ef_local=ef_local)
        attach_to_state(state, n)
        attach_grad_error_to_state(state, n)
        return state, _wrap_with_grad_error_stats(fsdp_sketch_comm_hook)
    if n == "hybrid_topk_int8" or n == "topk_int8":
        state = HybridTopKInt8State(k=topk_k, ratio=topk_ratio, error_feedback=ef, ef_local=ef_local)
        attach_to_state(state, n)
        attach_grad_error_to_state(state, n)
        return state, _wrap_with_grad_error_stats(fsdp_hybrid_topk_int8_comm_hook)

    raise ValueError(
        f"Unsupported comm hook: {name}. "
        "Supported: none, int8, fp16, qsgd, signsgd, onebit, onebit_seide, nc, topk, randomk, thresholdv, sketch, hybrid_topk_int8"
    )
