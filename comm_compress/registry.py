from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable

import torch
import torch.distributed as dist

from .types import FSDPCommHook, MethodConfig


@dataclass
class _HookFactory:
    build: Callable[[dict[str, Any]], tuple[Any, FSDPCommHook]]
    description: str


_REGISTRY: dict[str, _HookFactory] = {}


def register(method: str, *, description: str) -> Callable[[Callable[[dict[str, Any]], tuple[Any, FSDPCommHook]]], Callable[[dict[str, Any]], tuple[Any, FSDPCommHook]]]:
    method = method.strip().lower()

    def deco(fn: Callable[[dict[str, Any]], tuple[Any, FSDPCommHook]]):
        if method in _REGISTRY:
            raise ValueError(f"Duplicate comm_compress method: {method}")
        _REGISTRY[method] = _HookFactory(build=fn, description=description)
        return fn

    return deco


def list_methods() -> dict[str, str]:
    return {k: v.description for k, v in sorted(_REGISTRY.items(), key=lambda kv: kv[0])}


def parse_method_config(method: str, comm_config_json: str | None) -> MethodConfig:
    method = (method or "none").strip().lower()
    params: dict[str, Any] = {}
    if comm_config_json:
        obj = json.loads(comm_config_json)
        if not isinstance(obj, dict):
            raise ValueError("--comm_config_json must be a JSON object")
        params = obj
    return MethodConfig(method=method, params=params)


def make_comm_hook(method: str, comm_config_json: str | None = None) -> tuple[Any, FSDPCommHook] | tuple[None, None]:
    """按 method + JSON 配置创建可注册到 FSDP 的 comm hook。

    返回：
    - (state, hook) 可直接传给 `model.register_comm_hook(state, hook)`
    - 若 method 为 none，则返回 (None, None)
    """

    cfg = parse_method_config(method, comm_config_json)
    if cfg.method in ("none", "off", "disable", "disabled"):
        return (None, None)

    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("torch.distributed must be initialized before creating comm hook")

    if cfg.method not in _REGISTRY:
        supported = ", ".join(list_methods().keys())
        raise ValueError(f"Unknown --comm_compress '{cfg.method}'. Supported: {supported}")

    state, hook = _REGISTRY[cfg.method].build(cfg.params)
    return state, hook


# ---------- Built-in methods ----------


@register("int8", description="对称 int8 量化 + int8 reduce-scatter(sum) + 反量化求平均（需要 all_reduce(MAX) 同步 scale）")
def _build_int8(params: dict[str, Any]) -> tuple[Any, FSDPCommHook]:
    from .int8 import GradQuantState, fsdp_int8_quantized_comm_hook

    num_bits = int(params.get("num_bits", 8))
    state = GradQuantState(num_bits=num_bits)
    return state, fsdp_int8_quantized_comm_hook
