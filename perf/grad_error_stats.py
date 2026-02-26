from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class GradErrorStats:
    """Lightweight gradient compression error statistics, aggregated per comm hook."""

    name: str
    rel_l2_last: float = 0.0
    rel_l2_sum: float = 0.0
    samples: int = 0

    def add(self, rel_l2: float) -> None:
        v = float(rel_l2)
        self.rel_l2_last = v
        self.rel_l2_sum += v
        self.samples += 1

    @property
    def rel_l2_mean(self) -> float:
        if self.samples <= 0:
            return 0.0
        return float(self.rel_l2_sum / self.samples)


_ALL_STATS: List[GradErrorStats] = []


def attach_to_state(state: Any, name: str) -> None:
    """Attach a GradErrorStats instance to a comm_state and register it globally."""
    if state is None:
        return
    stats = GradErrorStats(name=name)
    setattr(state, "grad_error_stats", stats)
    _ALL_STATS.append(stats)


def add_sample(state: Any, rel_l2: float) -> None:
    """Record one relative L2 error sample for the given comm_state, if enabled."""
    if state is None:
        return
    stats = getattr(state, "grad_error_stats", None)
    if stats is None:
        return
    stats.add(rel_l2)


def snapshot() -> List[Dict[str, float]]:
    """Return a read-only snapshot of current statistics for logging."""
    return [
        {
            "name": stats.name,
            "rel_l2_last": float(stats.rel_l2_last),
            "rel_l2_mean": float(stats.rel_l2_mean),
            "samples": int(stats.samples),
        }
        for stats in _ALL_STATS
    ]


__all__ = ["GradErrorStats", "attach_to_state", "add_sample", "snapshot"]

