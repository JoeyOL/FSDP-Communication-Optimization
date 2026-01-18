from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol

import torch


FSDPCommHook = Callable[[Any, torch.Tensor, torch.Tensor], None]


class Compressor(Protocol):
    """压缩器接口（面向 FSDP 的 full_flat_grad -> shard_out reduce-scatter）。

    约定：
    - `full_flat_grad` 为扁平化梯度 (1D contiguous)。
    - `shard_out` 为预分配输出缓冲，写入本 rank 的梯度分片。
    - 允许内部使用额外 collective（例如同步 scale），但需要在文档/指标中说明。
    """

    def comm_hook(self) -> tuple[Any, FSDPCommHook]: ...


@dataclass(frozen=True)
class MethodConfig:
    method: str
    params: dict[str, Any]
