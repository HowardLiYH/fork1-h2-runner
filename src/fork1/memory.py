"""Family-tagged store with κ-weighted retention and LRU eviction.

Inherited schedule pattern: NicheMem evict_lru with skill/fidelity tie-break.
This is the H2-specific memory — NOT the NicheMem compete→pin mechanism.
"""

from __future__ import annotations

import dataclasses
from collections import OrderedDict
from typing import Optional


@dataclasses.dataclass(frozen=True)
class MemoryEntry:
    """Single entry in the family-tagged store."""

    family: str
    skill: float
    fidelity: float
    episode_stored: int
    last_accessed: int


class FamilyTaggedStore:
    """Family-tagged store with κ-weighted retention and LRU eviction.

    Parameters
    ----------
    capacity : int
        Maximum number of entries the store can hold.
    kappa : float
        Retention strength ∈ (0, 1]. Higher κ → more aggressive retention of
        recently accessed entries during eviction.
    """

    def __init__(self, capacity: int, kappa: float) -> None:
        if not 0 < kappa <= 1.0:
            raise ValueError(f"κ must be in (0, 1], got {kappa}")
        if capacity < 1:
            raise ValueError(f"capacity must be ≥1, got {capacity}")
        self.capacity = capacity
        self.kappa = kappa
        self._store: OrderedDict[str, MemoryEntry] = OrderedDict()

    @property
    def size(self) -> int:
        return len(self._store)

    def families_present(self) -> set[str]:
        return {e.family for e in self._store.values()}

    def store(
        self,
        family: str,
        skill: float,
        fidelity: float,
        episode: int,
    ) -> Optional[MemoryEntry]:
        """Store a new entry; returns evicted entry if capacity was full."""
        evicted: Optional[MemoryEntry] = None
        key = f"{family}:{episode}"
        if key in self._store:
            self._store.move_to_end(key)
            self._store[key] = MemoryEntry(
                family=family,
                skill=skill,
                fidelity=fidelity,
                episode_stored=episode,
                last_accessed=episode,
            )
            return None

        if len(self._store) >= self.capacity:
            evicted = self._evict_lru()

        self._store[key] = MemoryEntry(
            family=family,
            skill=skill,
            fidelity=fidelity,
            episode_stored=episode,
            last_accessed=episode,
        )
        self._store.move_to_end(key)
        return evicted

    def access(self, family: str, episode: int) -> Optional[MemoryEntry]:
        """Access (probe) the most recent entry for a family."""
        best_key: Optional[str] = None
        best_entry: Optional[MemoryEntry] = None
        for k, e in self._store.items():
            if e.family == family:
                if best_entry is None or e.episode_stored > best_entry.episode_stored:
                    best_key = k
                    best_entry = e
        if best_key is None or best_entry is None:
            return None
        updated = MemoryEntry(
            family=best_entry.family,
            skill=best_entry.skill,
            fidelity=best_entry.fidelity,
            episode_stored=best_entry.episode_stored,
            last_accessed=episode,
        )
        self._store[best_key] = updated
        self._store.move_to_end(best_key)
        return updated

    def has_family(self, family: str) -> bool:
        return any(e.family == family for e in self._store.values())

    def family_entries(self, family: str) -> list[MemoryEntry]:
        return [e for e in self._store.values() if e.family == family]

    def _evict_lru(self) -> MemoryEntry:
        """Evict using κ-weighted LRU with skill/fidelity tie-break.

        Eviction priority (lowest priority evicted first):
        1. Least recently accessed (LRU order, weighted by κ)
        2. Tie-break: lowest skill, then lowest fidelity
        """
        candidates = list(self._store.items())

        def eviction_score(item: tuple[str, MemoryEntry]) -> tuple[float, float, float]:
            _, entry = item
            recency_rank = list(self._store.keys()).index(item[0])
            weighted_recency = recency_rank * self.kappa
            return (weighted_recency, entry.skill, entry.fidelity)

        victim_key, victim_entry = min(candidates, key=eviction_score)
        del self._store[victim_key]
        return victim_entry

    def clear(self) -> None:
        self._store.clear()

    def snapshot(self) -> list[MemoryEntry]:
        return list(self._store.values())
