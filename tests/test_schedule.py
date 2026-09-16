"""Tests for schedule and arm management."""

from __future__ import annotations

import pytest

from fork1.schedule import (
    FAMILIES,
    ArmType,
    Schedule,
    Task,
    validate_busy_no_dormant_reuse,
)


class TestSchedule:
    def test_busy_arm_never_reuses_dormant_family(self) -> None:
        """INVARIANT: busy arms never reuse a dormant family."""
        schedule = Schedule(families=FAMILIES, active_length=10)
        for family in FAMILIES:
            for d in (0, 50, 200):
                tasks = schedule.build_stream(
                    arm_type=ArmType.BUSY,
                    target_family=family,
                    dormancy_d=d,
                    n_instances=5,
                )
                dormancy_tasks = [
                    t for t in tasks
                    if t.dormancy > 0 and not t.is_probe
                ]
                for t in dormancy_tasks:
                    assert t.family != family, (
                        f"Busy arm used dormant family {family} during dormancy"
                    )

    def test_busy_arm_validate_helper(self) -> None:
        schedule = Schedule(families=FAMILIES, active_length=10)
        tasks = schedule.build_stream(
            arm_type=ArmType.BUSY,
            target_family="earnings",
            dormancy_d=50,
            n_instances=3,
        )
        assert validate_busy_no_dormant_reuse(tasks, "earnings")

    def test_idle_arm_no_tasks_during_dormancy(self) -> None:
        schedule = Schedule(families=FAMILIES, active_length=10)
        tasks = schedule.build_stream(
            arm_type=ArmType.IDLE,
            target_family="earnings",
            dormancy_d=50,
            n_instances=3,
        )
        dormancy_tasks = [
            t for t in tasks
            if t.dormancy > 0 and not t.is_probe
        ]
        assert len(dormancy_tasks) == 0

    def test_never_learned_only_probes(self) -> None:
        schedule = Schedule(families=FAMILIES, active_length=10)
        tasks = schedule.build_stream(
            arm_type=ArmType.NEVER_LEARNED,
            target_family="earnings",
            dormancy_d=0,
            n_instances=5,
        )
        assert all(t.is_probe for t in tasks)
        assert all(t.arm_type == ArmType.NEVER_LEARNED for t in tasks)
        assert all(t.dormancy == -1 for t in tasks)

    def test_deletion_arm_has_dormancy(self) -> None:
        schedule = Schedule(families=FAMILIES, active_length=10)
        tasks = schedule.build_stream(
            arm_type=ArmType.DELETION,
            target_family="crisis",
            dormancy_d=50,
            n_instances=3,
        )
        has_active = any(t.dormancy == 0 and not t.is_probe for t in tasks)
        has_probe = any(t.is_probe for t in tasks)
        assert has_active
        assert has_probe

    def test_probe_window_after_dormancy(self) -> None:
        pw = 2
        schedule = Schedule(families=FAMILIES, active_length=10, probe_window=pw)
        tasks = schedule.build_stream(
            arm_type=ArmType.BUSY,
            target_family="earnings",
            dormancy_d=50,
            n_instances=2,
        )
        probe_tasks = [t for t in tasks if t.is_probe]
        assert len(probe_tasks) == 2 * pw

    def test_n_instances_respected(self) -> None:
        n = 7
        schedule = Schedule(families=FAMILIES, active_length=10, probe_window=2)
        tasks = schedule.build_stream(
            arm_type=ArmType.BUSY,
            target_family="filings",
            dormancy_d=50,
            n_instances=n,
        )
        probe_tasks = [t for t in tasks if t.is_probe]
        assert len(probe_tasks) == n * 2

    def test_unknown_family_raises(self) -> None:
        schedule = Schedule(families=FAMILIES)
        with pytest.raises(ValueError, match="Unknown family"):
            schedule.build_stream(
                arm_type=ArmType.BUSY,
                target_family="unknown",
                dormancy_d=50,
                n_instances=1,
            )

    def test_d0_busy_has_no_dormancy_gap(self) -> None:
        schedule = Schedule(families=FAMILIES, active_length=10, probe_window=2)
        tasks = schedule.build_stream(
            arm_type=ArmType.BUSY,
            target_family="earnings",
            dormancy_d=0,
            n_instances=3,
        )
        dormancy_tasks = [
            t for t in tasks
            if t.dormancy > 0 and not t.is_probe
        ]
        assert len(dormancy_tasks) == 0

    def test_all_three_families_present(self) -> None:
        assert len(FAMILIES) == 3
        assert "earnings" in FAMILIES
        assert "crisis" in FAMILIES
        assert "filings" in FAMILIES

    def test_probe_window_default_is_w2(self) -> None:
        """Stack froze w=2. Default probe_window must be 2, not 5."""
        schedule = Schedule(families=FAMILIES)
        assert schedule.probe_window == 2


class TestArmType:
    def test_arm_types_complete(self) -> None:
        assert set(ArmType) == {
            ArmType.BUSY, ArmType.IDLE,
            ArmType.NEVER_LEARNED, ArmType.DELETION,
        }

    def test_arm_type_values(self) -> None:
        assert ArmType.BUSY.value == "busy"
        assert ArmType.IDLE.value == "idle"
        assert ArmType.NEVER_LEARNED.value == "never_learned"
        assert ArmType.DELETION.value == "deletion"
