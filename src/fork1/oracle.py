"""Synthetic/procedural playbook oracles for unit tests.

Three families (earnings, crisis, filings) with deterministic success functions.
These are NOT real model outputs — they are procedural oracles for validating
the harness before plugging in an open-weight model.
"""

from __future__ import annotations

import hashlib
import struct
from typing import Protocol

from fork1.schedule import ArmType, Task


class Oracle(Protocol):
    """Protocol for task success oracles."""

    def success(self, task: Task, seed: int) -> float:
        """Return success probability ∈ [0, 1] for this task."""
        ...

    def predict(self, task: Task, seed: int) -> bool:
        """Return deterministic binary outcome for this task."""
        ...


def _deterministic_hash(family: str, episode: int, seed: int) -> float:
    """Deterministic hash → float in [0, 1]. Same inputs → same output always."""
    raw = f"{family}:{episode}:{seed}".encode()
    h = hashlib.sha256(raw).digest()
    val = struct.unpack(">I", h[:4])[0]
    return val / 0xFFFFFFFF


class PlaybookOracle:
    """Deterministic procedural oracle for a single family.

    Parameters
    ----------
    family : str
        Family name.
    base_success : float
        Baseline success rate during active (non-dormant) episodes.
    dormancy_decay : float
        Per-episode decay in success probability during dormancy.
        At dormancy d, effective success = base_success * (1 - dormancy_decay)^d.
    drift_penalty : float
        Additive penalty when the task is marked as drifted.
    """

    def __init__(
        self,
        family: str,
        base_success: float = 0.85,
        dormancy_decay: float = 0.003,
        drift_penalty: float = 0.10,
    ) -> None:
        self.family = family
        self.base_success = base_success
        self.dormancy_decay = dormancy_decay
        self.drift_penalty = drift_penalty

    def success(self, task: Task, seed: int) -> float:
        p = self.base_success
        if task.dormancy > 0:
            p *= (1 - self.dormancy_decay) ** task.dormancy
        if task.drifted:
            p = max(0.0, p - self.drift_penalty)
        return p

    def predict(self, task: Task, seed: int) -> bool:
        p = self.success(task, seed)
        h = _deterministic_hash(task.family, task.episode, seed)
        return h < p


FAMILY_ORACLES: dict[str, PlaybookOracle] = {
    "earnings": PlaybookOracle(
        family="earnings",
        base_success=0.85,
        dormancy_decay=0.004,
        drift_penalty=0.10,
    ),
    "crisis": PlaybookOracle(
        family="crisis",
        base_success=0.80,
        dormancy_decay=0.005,
        drift_penalty=0.12,
    ),
    "filings": PlaybookOracle(
        family="filings",
        base_success=0.82,
        dormancy_decay=0.003,
        drift_penalty=0.08,
    ),
}


class ModelAdapterStub:
    """Stub adapter that delegates to playbook oracles.

    Replace with real open-weight model adapter when ready.
    The interface stays the same: success() and predict().
    """

    def __init__(self, oracles: dict[str, PlaybookOracle] | None = None) -> None:
        self.oracles = oracles or FAMILY_ORACLES

    def success(self, task: Task, seed: int) -> float:
        oracle = self.oracles.get(task.family)
        if oracle is None:
            raise ValueError(f"No oracle for family {task.family!r}")
        return oracle.success(task, seed)

    def predict(self, task: Task, seed: int) -> bool:
        oracle = self.oracles.get(task.family)
        if oracle is None:
            raise ValueError(f"No oracle for family {task.family!r}")
        return oracle.predict(task, seed)
