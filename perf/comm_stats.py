from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class CommStats:
    """Per-comm-hook lightweight communication statistics."""

    name: str
    bytes_this_step: int = 0
    bytes_total: int = 0

    def add_bytes(self, nbytes: int) -> None:
        if nbytes <= 0:
            return
        n = int(nbytes)
        self.bytes_this_step += n
        self.bytes_total += n

    def reset_step(self) -> None:
        self.bytes_this_step = 0


_ALL_STATS: List[CommStats] = []


def attach_to_state(state: Any, name: str) -> None:
    """Attach a CommStats instance to a comm_state and register it globally."""
    if state is None:
        return
    stats = CommStats(name=name)
    setattr(state, "comm_stats", stats)
    _ALL_STATS.append(stats)


def add_bytes(state: Any, nbytes: int) -> None:
    """Add bytes to the stats attached to the given comm_state, if any."""
    if state is None or nbytes <= 0:
        return
    stats = getattr(state, "comm_stats", None)
    if stats is None:
        return
    stats.add_bytes(nbytes)


def reset_step() -> None:
    """Reset per-step counters for all registered stats objects."""
    for stats in _ALL_STATS:
        stats.reset_step()


def snapshot() -> List[Dict[str, int]]:
    """Return a read-only snapshot of current statistics for logging."""
    return [
        {
            "name": stats.name,
            "bytes_this_step": int(stats.bytes_this_step),
            "bytes_total": int(stats.bytes_total),
        }
        for stats in _ALL_STATS
    ]


def total_bytes() -> int:
    """Return total bytes across all registered stats (sum over hooks)."""
    return sum(int(stats.bytes_total) for stats in _ALL_STATS)


__all__ = ["CommStats", "attach_to_state", "add_bytes", "reset_step", "snapshot", "total_bytes"]

