"""Tests for family-tagged store + κ + LRU memory."""

from __future__ import annotations

import pytest

from fork1.memory import FamilyTaggedStore, MemoryEntry


class TestFamilyTaggedStore:
    def test_basic_store_and_access(self) -> None:
        store = FamilyTaggedStore(capacity=10, kappa=0.5)
        store.store("earnings", skill=0.8, fidelity=0.9, episode=1)
        entry = store.access("earnings", episode=2)
        assert entry is not None
        assert entry.family == "earnings"
        assert entry.skill == 0.8

    def test_capacity_eviction(self) -> None:
        store = FamilyTaggedStore(capacity=3, kappa=0.5)
        store.store("earnings", skill=0.8, fidelity=0.9, episode=1)
        store.store("crisis", skill=0.7, fidelity=0.8, episode=2)
        store.store("filings", skill=0.6, fidelity=0.7, episode=3)
        evicted = store.store("earnings", skill=0.9, fidelity=1.0, episode=4)
        assert evicted is not None
        assert store.size == 3

    def test_high_kappa_evicts_lowest_skill(self) -> None:
        """At κ=1.0, eviction is skill-based: lowest-skill entry evicted."""
        store = FamilyTaggedStore(capacity=3, kappa=1.0)
        store.store("earnings", skill=0.8, fidelity=0.9, episode=1)
        store.store("crisis", skill=0.7, fidelity=0.8, episode=2)
        store.store("filings", skill=0.6, fidelity=0.7, episode=3)
        store.access("earnings", episode=4)
        evicted = store.store("new_fam", skill=0.5, fidelity=0.5, episode=5)
        assert evicted is not None
        assert evicted.family == "filings"

    def test_low_kappa_evicts_oldest(self) -> None:
        """At very low κ, eviction is position-based: oldest entry evicted."""
        store = FamilyTaggedStore(capacity=3, kappa=0.01)
        store.store("earnings", skill=0.8, fidelity=0.9, episode=1)
        store.store("crisis", skill=0.7, fidelity=0.8, episode=2)
        store.store("filings", skill=0.6, fidelity=0.7, episode=3)
        store.access("earnings", episode=4)
        evicted = store.store("new_fam", skill=0.5, fidelity=0.5, episode=5)
        assert evicted is not None
        assert evicted.family == "crisis"

    def test_kappa_affects_which_entry_evicted(self) -> None:
        """P1: different κ must evict different entries under same state."""
        store_high = FamilyTaggedStore(capacity=3, kappa=1.0)
        store_low = FamilyTaggedStore(capacity=3, kappa=0.01)
        for s in [store_high, store_low]:
            s.store("a", skill=0.1, fidelity=0.1, episode=1)
            s.store("b", skill=0.9, fidelity=0.9, episode=2)
            s.store("c", skill=0.5, fidelity=0.5, episode=3)
            s.access("a", episode=4)
        evicted_high = store_high.store("d", skill=0.5, fidelity=0.5, episode=5)
        evicted_low = store_low.store("d", skill=0.5, fidelity=0.5, episode=5)
        assert evicted_high is not None and evicted_low is not None
        assert evicted_high.family == "a", "κ=1 should evict lowest-skill (a)"
        assert evicted_low.family == "b", "κ→0 should evict oldest-position (b)"

    def test_families_present(self) -> None:
        store = FamilyTaggedStore(capacity=10, kappa=0.5)
        store.store("earnings", skill=0.8, fidelity=0.9, episode=1)
        store.store("crisis", skill=0.7, fidelity=0.8, episode=2)
        assert store.families_present() == {"earnings", "crisis"}

    def test_has_family(self) -> None:
        store = FamilyTaggedStore(capacity=10, kappa=0.5)
        store.store("earnings", skill=0.8, fidelity=0.9, episode=1)
        assert store.has_family("earnings")
        assert not store.has_family("crisis")

    def test_invalid_kappa(self) -> None:
        with pytest.raises(ValueError, match="κ must be in"):
            FamilyTaggedStore(capacity=10, kappa=0.0)
        with pytest.raises(ValueError, match="κ must be in"):
            FamilyTaggedStore(capacity=10, kappa=1.5)

    def test_invalid_capacity(self) -> None:
        with pytest.raises(ValueError, match="capacity must be"):
            FamilyTaggedStore(capacity=0, kappa=0.5)

    def test_access_missing_family_returns_none(self) -> None:
        store = FamilyTaggedStore(capacity=10, kappa=0.5)
        assert store.access("nonexistent", episode=1) is None

    def test_clear(self) -> None:
        store = FamilyTaggedStore(capacity=10, kappa=0.5)
        store.store("earnings", skill=0.8, fidelity=0.9, episode=1)
        store.clear()
        assert store.size == 0
        assert not store.has_family("earnings")

    def test_snapshot(self) -> None:
        store = FamilyTaggedStore(capacity=10, kappa=0.5)
        store.store("earnings", skill=0.8, fidelity=0.9, episode=1)
        store.store("crisis", skill=0.7, fidelity=0.8, episode=2)
        snap = store.snapshot()
        assert len(snap) == 2
        assert all(isinstance(e, MemoryEntry) for e in snap)
