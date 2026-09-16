"""Synthetic/procedural playbook oracles for unit tests.

Three families (earnings, crisis, filings) with deterministic success functions.
These are NOT real model outputs — they are procedural oracles for validating
the harness before plugging in an open-weight model.
"""

from __future__ import annotations

import hashlib
import struct
from typing import Optional, Protocol

from fork1.memory import MemoryEntry
from fork1.schedule import ArmType, Task

MEMORY_HIT_BONUS = 0.12


class Oracle(Protocol):
    """Protocol for task success oracles."""

    def success(
        self, task: Task, seed: int, memory: Optional[MemoryEntry] = None
    ) -> float:
        """Return success probability ∈ [0, 1] for this task."""
        ...

    def predict(
        self, task: Task, seed: int, memory: Optional[MemoryEntry] = None
    ) -> bool:
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
        Only applied when debug_dormancy_decay=True (DEBUG flag, off by
        default).  Baking ∂p/∂d into the oracle fakes dormancy effects
        that should come from the memory mechanism, so production runs
        MUST leave this off.
    drift_penalty : float
        Additive penalty when the task is marked as drifted.
    debug_dormancy_decay : bool
        When True, apply dormancy_decay.  Default False — P0-2.
    """

    def __init__(
        self,
        family: str,
        base_success: float = 0.85,
        dormancy_decay: float = 0.003,
        drift_penalty: float = 0.10,
        debug_dormancy_decay: bool = False,
    ) -> None:
        self.family = family
        self.base_success = base_success
        self.dormancy_decay = dormancy_decay
        self.drift_penalty = drift_penalty
        self._debug_dormancy_decay = debug_dormancy_decay

    def success(
        self, task: Task, seed: int, memory: Optional[MemoryEntry] = None
    ) -> float:
        p = self.base_success
        if self._debug_dormancy_decay and task.dormancy > 0:
            p *= (1 - self.dormancy_decay) ** task.dormancy
        if task.drifted:
            p = max(0.0, p - self.drift_penalty)
        if memory is not None:
            p = min(1.0, p + MEMORY_HIT_BONUS * memory.skill)
        return p

    def predict(
        self, task: Task, seed: int, memory: Optional[MemoryEntry] = None
    ) -> bool:
        p = self.success(task, seed, memory)
        h = _deterministic_hash(task.family, task.episode, seed)
        return h < p


FAMILY_ORACLES: dict[str, PlaybookOracle] = {
    "earnings": PlaybookOracle(
        family="earnings",
        base_success=0.85,
        dormancy_decay=0.004,
        drift_penalty=0.10,
        debug_dormancy_decay=False,
    ),
    "crisis": PlaybookOracle(
        family="crisis",
        base_success=0.80,
        dormancy_decay=0.005,
        drift_penalty=0.12,
        debug_dormancy_decay=False,
    ),
    "filings": PlaybookOracle(
        family="filings",
        base_success=0.82,
        dormancy_decay=0.003,
        drift_penalty=0.08,
        debug_dormancy_decay=False,
    ),
}


class ModelAdapterStub:
    """Stub adapter that delegates to playbook oracles.

    Replace with real open-weight model adapter when ready.
    The interface stays the same: success() and predict().
    """

    def __init__(self, oracles: dict[str, PlaybookOracle] | None = None) -> None:
        self.oracles = oracles or FAMILY_ORACLES

    def success(
        self, task: Task, seed: int, memory: Optional[MemoryEntry] = None
    ) -> float:
        oracle = self.oracles.get(task.family)
        if oracle is None:
            raise ValueError(f"No oracle for family {task.family!r}")
        return oracle.success(task, seed, memory)

    def predict(
        self, task: Task, seed: int, memory: Optional[MemoryEntry] = None
    ) -> bool:
        oracle = self.oracles.get(task.family)
        if oracle is None:
            raise ValueError(f"No oracle for family {task.family!r}")
        return oracle.predict(task, seed, memory)
