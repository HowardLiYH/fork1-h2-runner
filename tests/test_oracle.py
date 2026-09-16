"""Tests for playbook oracles."""

from __future__ import annotations

from fork1.oracle import (
    FAMILY_ORACLES,
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

    def test_dormancy_decay(self) -> None:
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
