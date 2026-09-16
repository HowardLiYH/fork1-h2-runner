"""Fail ladder logic.

Spec fail ladder:
1. G-D holds + κ=1 flat → proceed
2. G-D fails → H2 kill, bounce Stack
3. H2 flat/kill AND S<0 under drift (withheld-era partition) → harmfulness
4. rise at κ=1 → harness bug, stop

Rung-3 (harmfulness) fires ONLY when H2 is flat or killed — never when G-D holds.
"""

from __future__ import annotations

import enum
from typing import Optional

from fork1.metrics import (
    CellResult,
    FailLadderDecision,
    GDResult,
    SMetric,
    check_gd,
    check_kappa1_flat,
)
from fork1.schedule import ArmType


class LadderOutcome(enum.Enum):
    PROCEED = "proceed"
    H2_KILL = "h2_kill"
    HARMFULNESS = "harmfulness"
    HARNESS_BUG = "harness_bug"


def evaluate_fail_ladder(
    results_by_d: dict[int, list[CellResult]],
    kappa_values: tuple[float, ...] = (0.25, 0.50),
    s_metrics: Optional[list[SMetric]] = None,
) -> FailLadderDecision:
    """Evaluate the fail ladder and return the decision.

    Steps (order matters):
    1. Rise at κ=1 → harness bug, stop  (checked first — blocks all else)
    2. G-D holds at tested κ + κ=1 flat → proceed
    3. G-D fails → H2 kill; THEN check rung-3
    4. H2 flat/kill AND S<0 under drift (withheld-era only) → harmfulness
    """
    gd_results: dict[float, GDResult] = {}
    for kappa in kappa_values:
        kappa_cells: dict[int, list[CellResult]] = {}
        for d, cells in results_by_d.items():
            matching = [
                r for r in cells
                if abs(r.kappa - kappa) < 1e-9 and r.arm_type == ArmType.BUSY
            ]
            if matching:
                kappa_cells[d] = matching
        if kappa_cells:
            gd_results[kappa] = check_gd(kappa_cells, kappa)

    kappa1_flat = check_kappa1_flat(results_by_d)

    kappa1_results_by_d: dict[int, list[CellResult]] = {}
    for d, cells in results_by_d.items():
        k1 = [r for r in cells if abs(r.kappa - 1.0) < 1e-9 and r.arm_type == ArmType.BUSY]
        if k1:
            kappa1_results_by_d[d] = k1

    kappa1_gd = check_gd(kappa1_results_by_d, kappa=1.0) if kappa1_results_by_d else None
    kappa1_rise = False
    if kappa1_gd is not None:
        kappa1_rise = kappa1_gd.pr_d200 > kappa1_gd.pr_d0 + 0.05

    if kappa1_rise:
        return FailLadderDecision(
            step=4,
            label=LadderOutcome.HARNESS_BUG.value,
            detail=(
                f"κ=1 shows rising Pr(d): Pr(d=200)={kappa1_gd.pr_d200:.3f} > "  # type: ignore[union-attr]
                f"Pr(d=0)={kappa1_gd.pr_d0:.3f}. Harness bug suspected."  # type: ignore[union-attr]
            ),
        )

    any_gd_holds = any(g.gd_holds for g in gd_results.values())
    all_gd_holds = all(g.gd_holds for g in gd_results.values()) if gd_results else False

    if kappa1_flat and all_gd_holds:
        return FailLadderDecision(
            step=1,
            label=LadderOutcome.PROCEED.value,
            detail=(
                f"G-D holds at all tested κ ({_format_gaps(gd_results)}) "
                "and κ=1 is flat. Proceed."
            ),
        )

    if any_gd_holds and not all_gd_holds:
        return FailLadderDecision(
            step=1,
            label=LadderOutcome.PROCEED.value,
            detail=(
                f"G-D holds at some κ ({_format_gaps(gd_results)}), "
                f"κ=1 flat={kappa1_flat}. Partial proceed."
            ),
        )

    h2_is_flat_or_kill = not any_gd_holds

    if h2_is_flat_or_kill and s_metrics:
        withheld_s = [s for s in s_metrics if s.in_withheld_era]
        drift_s_negative = any(
            s.s_value < 0
            for s in withheld_s
            if s.kappa in kappa_values
        )
        if drift_s_negative:
            return FailLadderDecision(
                step=3,
                label=LadderOutcome.HARMFULNESS.value,
                detail=(
                    "G-D fails (H2 flat/kill) AND S<0 under drift "
                    "in withheld-era partition at tested κ. "
                    f"Gaps: {_format_gaps(gd_results)}. Harmfulness path triggered."
                ),
            )

    if h2_is_flat_or_kill:
        return FailLadderDecision(
            step=2,
            label=LadderOutcome.H2_KILL.value,
            detail=(
                "G-D fails at all tested κ values. "
                f"Gaps: {_format_gaps(gd_results)}. κ=1 flat={kappa1_flat}. "
                "H2 killed — bounce Stack."
            ),
        )

    return FailLadderDecision(
        step=2,
        label=LadderOutcome.H2_KILL.value,
        detail=(
            f"G-D inconclusive ({_format_gaps(gd_results)}). H2 killed."
        ),
    )


def _format_gaps(gd_results: dict[float, GDResult]) -> str:
    parts = []
    for kappa, gd in sorted(gd_results.items()):
        parts.append(f"κ={kappa}: gap={gd.gap:.3f} ({'holds' if gd.gd_holds else 'fails'})")
    return "; ".join(parts) if parts else "no data"
