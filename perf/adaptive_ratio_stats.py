from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class AdaptiveRatioStats:
    """Per-step adaptive sparsity records for adaptive communication hooks."""

    name: str
    records: List[Dict[str, float]] = field(default_factory=list)

    def add(
        self,
        comm_step: int,
        schedule: str,
        ratio_keep: float,
        k: int,
        numel: int,
        grad_norm: float,
        ema_norm: float,
    ) -> None:
        rk = float(max(0.0, min(1.0, ratio_keep)))
        self.records.append(
            {
                "comm_step": int(comm_step),
                "schedule": str(schedule),
                "ratio_keep": rk,
                "sparsity": float(1.0 - rk),
                "k": int(k),
                "numel": int(numel),
                "grad_norm": float(grad_norm),
                "ema_norm": float(ema_norm),
            }
        )


_ALL_STATS: List[AdaptiveRatioStats] = []


def attach_to_state(state: Any, name: str) -> None:
    """Attach an AdaptiveRatioStats instance to a comm_state and register it globally."""
    if state is None:
        return
    stats = AdaptiveRatioStats(name=name)
    setattr(state, "adaptive_ratio_stats", stats)
    _ALL_STATS.append(stats)


def add_sample(
    state: Any,
    comm_step: int,
    schedule: str,
    ratio_keep: float,
    k: int,
    numel: int,
    grad_norm: float,
    ema_norm: float,
) -> None:
    """Record one adaptive ratio sample for the given comm_state, if enabled."""
    if state is None:
        return
    stats = getattr(state, "adaptive_ratio_stats", None)
    if stats is None:
        return
    stats.add(comm_step, schedule, ratio_keep, k, numel, grad_norm, ema_norm)


def snapshot() -> List[Dict[str, Any]]:
    """Return a read-only snapshot of current adaptive ratio records for logging."""
    return [
        {
            "name": stats.name,
            "count": len(stats.records),
            "records": list(stats.records),
        }
        for stats in _ALL_STATS
    ]


__all__ = ["AdaptiveRatioStats", "attach_to_state", "add_sample", "snapshot"]
