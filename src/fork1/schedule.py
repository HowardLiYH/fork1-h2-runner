"""Schedule and arm management for H2 experiment.

Inherited schedule patterns:
- NicheMem: Schedule cyclic sliding-window, rare={family: D}, add_families_at,
  Environment.stream() Task fields, probe convention, ACTIVE_AFTER
- RISP: RealMarket.schedule() dormancy between regime blocks, StitchedMarket

This module implements the H2-specific arm types — NOT NicheMem policies.
"""

from __future__ import annotations

import dataclasses
import enum
from typing import Optional


FAMILIES = ("earnings", "crisis", "filings")


class ArmType(enum.Enum):
    """Arm types for the H2 experiment."""
    BUSY = "busy"
    IDLE = "idle"
    NEVER_LEARNED = "never_learned"
    DELETION = "deletion"


@dataclasses.dataclass(frozen=True)
class Task:
    """A single task in the episode stream.

    Inherited from NicheMem Environment.stream() Task fields.
    """
    family: str
    episode: int
    dormancy: int
    pos_in_activation: int
    drifted: bool
    is_probe: bool
    arm_type: ArmType


@dataclasses.dataclass
class FamilyBlock:
    """A contiguous block of episodes for one family."""
    family: str
    start: int
    length: int
    drifted: bool = False


class Schedule:
    """Cyclic schedule with dormancy gaps and arm-type constraints.

    Parameters
    ----------
    families : tuple of str
        Family names (e.g., ("earnings", "crisis", "filings")).
    active_length : int
        Episodes per active block per family.
    dormancy_map : dict mapping family → dormancy length in episodes.
    probe_window : int
        Number of episodes after reactivation to count as probe window.
    min_dormancy_before_probe : int
        Minimum dormancy before a probe is valid.
    """

    def __init__(
        self,
        families: tuple[str, ...] = FAMILIES,
        active_length: int = 20,
        dormancy_map: Optional[dict[str, int]] = None,
        probe_window: int = 5,
        min_dormancy_before_probe: int = 10,
    ) -> None:
        self.families = families
        self.active_length = active_length
        self.dormancy_map = dormancy_map or {}
        self.probe_window = probe_window
        self.min_dormancy_before_probe = min_dormancy_before_probe

    def build_stream(
        self,
        arm_type: ArmType,
        target_family: str,
        dormancy_d: int,
        n_instances: int,
        start_episode: int = 0,
    ) -> list[Task]:
        """Build an episode stream for one experimental cell.

        For busy arms: other families are active during the target's dormancy.
        For idle arms: nothing happens during dormancy.
        For never_learned: target family never appears in training.
        For deletion: target family entries are explicitly removed after training.

        INVARIANT: busy arms never reuse a dormant family.
        """
        if target_family not in self.families:
            raise ValueError(f"Unknown family: {target_family}")

        tasks: list[Task] = []
        ep = start_episode

        if arm_type == ArmType.NEVER_LEARNED:
            return self._build_never_learned_stream(
                target_family, n_instances, ep
            )

        other_families = [f for f in self.families if f != target_family]

        for _instance in range(n_instances):
            for pos in range(self.active_length):
                tasks.append(Task(
                    family=target_family,
                    episode=ep,
                    dormancy=0,
                    pos_in_activation=pos,
                    drifted=False,
                    is_probe=False,
                    arm_type=arm_type,
                ))
                ep += 1

            dormancy_start = ep
            if arm_type == ArmType.BUSY:
                busy_idx = 0
                for d_step in range(dormancy_d):
                    busy_family = other_families[busy_idx % len(other_families)]
                    assert busy_family != target_family, (
                        "busy arm must never reuse the dormant family"
                    )
                    tasks.append(Task(
                        family=busy_family,
                        episode=ep,
                        dormancy=d_step + 1,
                        pos_in_activation=d_step % self.active_length,
                        drifted=False,
                        is_probe=False,
                        arm_type=arm_type,
                    ))
                    busy_idx += 1
                    ep += 1
            elif arm_type == ArmType.IDLE:
                ep += dormancy_d
            elif arm_type == ArmType.DELETION:
                ep += dormancy_d

            actual_dormancy = ep - dormancy_start
            is_valid_probe = (
                dormancy_d == 0
                or actual_dormancy >= self.min_dormancy_before_probe
            )

            for pos in range(self.probe_window):
                tasks.append(Task(
                    family=target_family,
                    episode=ep,
                    dormancy=actual_dormancy,
                    pos_in_activation=pos,
                    drifted=False,
                    is_probe=is_valid_probe,
                    arm_type=arm_type,
                ))
                ep += 1

        return tasks

    def _build_never_learned_stream(
        self,
        target_family: str,
        n_instances: int,
        start_episode: int,
    ) -> list[Task]:
        """Never-learned arm: probes only, no prior training on this family."""
        tasks: list[Task] = []
        ep = start_episode
        for _instance in range(n_instances):
            for pos in range(self.probe_window):
                tasks.append(Task(
                    family=target_family,
                    episode=ep,
                    dormancy=-1,
                    pos_in_activation=pos,
                    drifted=False,
                    is_probe=True,
                    arm_type=ArmType.NEVER_LEARNED,
                ))
                ep += 1
        return tasks


def validate_busy_no_dormant_reuse(tasks: list[Task], target_family: str) -> bool:
    """Validate that busy-arm tasks never use the dormant target family
    during its dormancy period."""
    in_dormancy = False
    for t in tasks:
        if t.arm_type != ArmType.BUSY:
            continue
        if t.family == target_family and t.dormancy == 0:
            in_dormancy = False
        elif t.family != target_family and t.dormancy > 0:
            in_dormancy = True
            if t.family == target_family:
                return False
        elif t.family == target_family and t.dormancy > 0 and t.is_probe:
            in_dormancy = False
    return True
