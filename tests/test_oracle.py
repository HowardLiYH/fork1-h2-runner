"""Tests for playbook oracles."""

from __future__ import annotations

from fork1.memory import MemoryEntry
from fork1.oracle import (
    FAMILY_ORACLES,
    MEMORY_HIT_BONUS,
    ModelAdapterStub,
    PlaybookOracle,
    _deterministic_hash,
)
from fork1.schedule import ArmType, Task


class TestDeterministicHash:
    def test_same_inputs_same_output(self) -> None:
        a = _deterministic_hash("earnings", 42, 7)
        b = _deterministic_hash("earnings", 42, 7)
        assert a == b

    def test_different_inputs_different_output(self) -> None:
        a = _deterministic_hash("earnings", 42, 7)
        b = _deterministic_hash("crisis", 42, 7)
        assert a != b

    def test_range(self) -> None:
        for i in range(100):
            h = _deterministic_hash("test", i, 0)
            assert 0.0 <= h <= 1.0


class TestPlaybookOracle:
    def test_base_success(self) -> None:
        oracle = PlaybookOracle("earnings", base_success=0.85)
        task = Task(
            family="earnings", episode=1, dormancy=0,
            pos_in_activation=0, drifted=False, is_probe=False,
            arm_type=ArmType.BUSY,
        )
        assert oracle.success(task, seed=0) == 0.85

    def test_dormancy_decay_off_by_default(self) -> None:
        """P0-2: dormancy_decay must NOT apply by default."""
        oracle = PlaybookOracle("earnings", base_success=0.85, dormancy_decay=0.01)
        active = Task(
            family="earnings", episode=1, dormancy=0,
            pos_in_activation=0, drifted=False, is_probe=False,
            arm_type=ArmType.BUSY,
        )
        dormant = Task(
            family="earnings", episode=100, dormancy=50,
            pos_in_activation=0, drifted=False, is_probe=True,
            arm_type=ArmType.BUSY,
        )
        assert oracle.success(dormant, seed=0) == oracle.success(active, seed=0)

    def test_dormancy_decay_with_debug_flag(self) -> None:
        """P0-2: dormancy_decay applies only under debug flag."""
        oracle = PlaybookOracle(
            "earnings", base_success=0.85, dormancy_decay=0.01,
            debug_dormancy_decay=True,
        )
        active = Task(
            family="earnings", episode=1, dormancy=0,
            pos_in_activation=0, drifted=False, is_probe=False,
            arm_type=ArmType.BUSY,
        )
        dormant = Task(
            family="earnings", episode=100, dormancy=50,
            pos_in_activation=0, drifted=False, is_probe=True,
            arm_type=ArmType.BUSY,
        )
        assert oracle.success(dormant, seed=0) < oracle.success(active, seed=0)

    def test_drift_penalty(self) -> None:
        oracle = PlaybookOracle("crisis", drift_penalty=0.15)
        normal = Task(
            family="crisis", episode=1, dormancy=0,
            pos_in_activation=0, drifted=False, is_probe=False,
            arm_type=ArmType.BUSY,
        )
        drifted = Task(
            family="crisis", episode=1, dormancy=0,
            pos_in_activation=0, drifted=True, is_probe=False,
            arm_type=ArmType.BUSY,
        )
        assert oracle.success(drifted, seed=0) < oracle.success(normal, seed=0)

    def test_predict_deterministic(self) -> None:
        oracle = PlaybookOracle("filings")
        task = Task(
            family="filings", episode=10, dormancy=0,
            pos_in_activation=0, drifted=False, is_probe=False,
            arm_type=ArmType.BUSY,
        )
        results = [oracle.predict(task, seed=42) for _ in range(100)]
        assert all(r == results[0] for r in results)

    def test_memory_hit_changes_success(self) -> None:
        """P0-1: memory hit must change success probability."""
        oracle = PlaybookOracle("earnings", base_success=0.85)
        task = Task(
            family="earnings", episode=10, dormancy=50,
            pos_in_activation=0, drifted=False, is_probe=True,
            arm_type=ArmType.BUSY,
        )
        mem = MemoryEntry(
            family="earnings", skill=0.85, fidelity=1.0,
            episode_stored=5, last_accessed=5,
        )
        p_miss = oracle.success(task, seed=0, memory=None)
        p_hit = oracle.success(task, seed=0, memory=mem)
        assert p_hit > p_miss
        assert abs(p_hit - p_miss - MEMORY_HIT_BONUS * mem.skill) < 1e-9

    def test_memory_hit_predict_differs(self) -> None:
        """P0-1: predict(task, seed, hit) must differ from predict(task, seed, miss)
        for at least some task/seed combos."""
        oracle = PlaybookOracle("earnings", base_success=0.80)
        mem = MemoryEntry(
            family="earnings", skill=0.85, fidelity=1.0,
            episode_stored=5, last_accessed=5,
        )
        diffs = 0
        for ep in range(200):
            task = Task(
                family="earnings", episode=ep, dormancy=50,
                pos_in_activation=0, drifted=False, is_probe=True,
                arm_type=ArmType.BUSY,
            )
            r_miss = oracle.predict(task, seed=0, memory=None)
            r_hit = oracle.predict(task, seed=0, memory=mem)
            if r_miss != r_hit:
                diffs += 1
        assert diffs > 0, "memory hit and miss must produce different outcomes"


class TestModelAdapterStub:
    def test_delegates_to_oracles(self) -> None:
        adapter = ModelAdapterStub()
        task = Task(
            family="earnings", episode=1, dormancy=0,
            pos_in_activation=0, drifted=False, is_probe=False,
            arm_type=ArmType.BUSY,
        )
        s = adapter.success(task, seed=0)
        assert s == FAMILY_ORACLES["earnings"].success(task, seed=0)

    def test_unknown_family_raises(self) -> None:
        adapter = ModelAdapterStub()
        task = Task(
            family="unknown", episode=1, dormancy=0,
            pos_in_activation=0, drifted=False, is_probe=False,
            arm_type=ArmType.BUSY,
        )
        import pytest
        with pytest.raises(ValueError, match="No oracle"):
            adapter.success(task, seed=0)

    def test_all_families_covered(self) -> None:
        from fork1.schedule import FAMILIES
        adapter = ModelAdapterStub()
        for fam in FAMILIES:
            task = Task(
                family=fam, episode=1, dormancy=0,
                pos_in_activation=0, drifted=False, is_probe=False,
                arm_type=ArmType.BUSY,
            )
            assert isinstance(adapter.predict(task, seed=0), bool)

    def test_memory_passthrough(self) -> None:
        """P0-1: adapter must pass memory to oracle."""
        adapter = ModelAdapterStub()
        task = Task(
            family="earnings", episode=10, dormancy=50,
            pos_in_activation=0, drifted=False, is_probe=True,
            arm_type=ArmType.BUSY,
        )
        mem = MemoryEntry(
            family="earnings", skill=0.85, fidelity=1.0,
            episode_stored=5, last_accessed=5,
        )
        s_miss = adapter.success(task, seed=0, memory=None)
        s_hit = adapter.success(task, seed=0, memory=mem)
        assert s_hit > s_miss
