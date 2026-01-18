from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass
class GradQuantState:
    num_bits: int = 8


def fsdp_int8_quantized_comm_hook(
    state: GradQuantState,
    full_flat_grad: torch.Tensor,
    shard_out: torch.Tensor,
) -> None:
    """FSDP comm hook: 对称 int8 量化 + int8 reduce-scatter(sum) + 反量化并求平均。

    说明：
    - 该实现会做一次 `all_reduce(MAX)` 来同步 global_max，从而保证 scale 一致。
    - 为避免 int8 sum 溢出，使用 Qr=127//world_size 的安全范围（会影响量化精度）。
    """

    assert isinstance(state, GradQuantState)
    pg = dist.group.WORLD
    world_size = dist.get_world_size(pg)

    if world_size == 1:
        shard_out.copy_(full_flat_grad)
        return

    g = full_flat_grad.contiguous().view(-1)
    numel = g.numel()
    assert numel % world_size == 0, (
        f"Flat grad numel {numel} must be divisible by world_size {world_size}"
    )

    local_max = g.abs().max().to(torch.float32)
    global_max = local_max.clone()
    dist.all_reduce(global_max, op=dist.ReduceOp.MAX, group=pg)

    Q = 127
    Qr = max(1, Q // world_size)
    scale = Qr / torch.clamp(global_max, min=1e-8)

    q = torch.clamp((g * scale).round(), -Qr, Qr).to(torch.int8)
    tmp_out = torch.empty_like(shard_out, dtype=torch.int8)

    if hasattr(dist, "reduce_scatter_tensor"):
        dist.reduce_scatter_tensor(tmp_out, q, op=dist.ReduceOp.SUM, group=pg)
    else:
        chunks = list(q.chunk(world_size, dim=0))
        dist.reduce_scatter(tmp_out, chunks, op=dist.ReduceOp.SUM, group=pg)

    deq_sum = tmp_out.float() / scale
    deq_avg = (deq_sum / float(world_size)).to(full_flat_grad.dtype)
    shard_out.copy_(deq_avg)
