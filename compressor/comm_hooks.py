import torch
import torch.distributed as dist


class GradQuantState:
    def __init__(self, num_bits: int = 8) -> None:
        self.num_bits = num_bits


def fsdp_quantized_comm_hook(
    state: GradQuantState,
    full_flat_grad: torch.Tensor,
    shard_out: torch.Tensor,
) -> None:
    """
    FSDP communication hook for int8 symmetric quantization before reduce-scatter.
    Writes the dequantized, averaged shard into shard_out.
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
        f"flat grad numel {numel} must be divisible by world_size {world_size}"
    )

    local_max = g.abs().max().to(torch.float32)
    global_max = local_max.clone()
    dist.all_reduce(global_max, op=dist.ReduceOp.MAX, group=pg)

    q = 127
    qr = max(1, q // world_size)
    scale = qr / torch.clamp(global_max, min=1e-8)
    q_grad = torch.clamp((g * scale).round(), -qr, qr).to(torch.int8)

    temp_shard_out = torch.empty_like(shard_out, dtype=torch.int8)

    if hasattr(dist, "reduce_scatter_tensor"):
        dist.reduce_scatter_tensor(temp_shard_out, q_grad, op=dist.ReduceOp.SUM, group=pg)
    else:
        chunks = list(q_grad.chunk(world_size, dim=0))
        dist.reduce_scatter(temp_shard_out, chunks, op=dist.ReduceOp.SUM, group=pg)

    deq_sum = temp_shard_out.float() / scale
    deq_avg = (deq_sum / float(world_size)).to(full_flat_grad.dtype)
    shard_out.copy_(deq_avg)


def build_comm_hook(name: str, num_bits: int = 8):
    if name is None:
        return None, None

    name = name.lower().strip()
    if name == "none":
        return None, None
    if name == "int8":
        return GradQuantState(num_bits=num_bits), fsdp_quantized_comm_hook

    raise ValueError(f"Unsupported comm hook: {name}")
