"""
Thin registry for FSDP communication hooks.
具体算法的实现已经拆分到独立的 hook_*.py 文件中，这里只负责统一构建入口。
"""

from typing import Any, Optional, Tuple

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
    onebit_use_cpp: bool = True,
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
        return OneBitSeideState(
            col_size=onebit_col_size,
            error_feedback=ef,
            use_cpp=onebit_use_cpp,
        ), fsdp_onebit_seide_comm_hook
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
