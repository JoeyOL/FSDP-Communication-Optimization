from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class GradErrorStats:
    """Lightweight gradient compression error statistics, aggregated per comm hook.

    [B19 enhancement] Now also tracks max and sum-of-squares for variance/std computation.
    """

    name: str
    rel_l2_last: float = 0.0
    rel_l2_sum: float = 0.0
    rel_l2_max: float = 0.0
    rel_l2_sum_sq: float = 0.0  # sum of squared values for variance
    samples: int = 0

    def add(self, rel_l2: float) -> None:
        v = float(rel_l2)
        self.rel_l2_last = v
        self.rel_l2_sum += v
        self.rel_l2_sum_sq += v * v
        if v > self.rel_l2_max:
            self.rel_l2_max = v
        self.samples += 1

    @property
    def rel_l2_mean(self) -> float:
        if self.samples <= 0:
            return 0.0
        return float(self.rel_l2_sum / self.samples)

    @property
    def rel_l2_var(self) -> float:
        """Population variance of relative L2 error."""
        if self.samples <= 1:
            return 0.0
        mean = self.rel_l2_mean
        return float(self.rel_l2_sum_sq / self.samples - mean * mean)

    @property
    def rel_l2_std(self) -> float:
        """Population standard deviation of relative L2 error."""
        v = self.rel_l2_var
        return float(v ** 0.5) if v > 0 else 0.0


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
            "rel_l2_max": float(stats.rel_l2_max),
            "rel_l2_std": float(stats.rel_l2_std),
            "samples": int(stats.samples),
        }
        for stats in _ALL_STATS
    ]


__all__ = ["GradErrorStats", "attach_to_state", "add_sample", "snapshot"]

