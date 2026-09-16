"""Metrics: G-D checker (primary), c(d) secondary, S harmfulness (rung-3).

Spec references:
- Primary G-D: P(c≤B) at d=200 ≤ P(c≤B) at d=0 − 0.25 at κ∈{0.25,0.50};
  κ=1 flat under the same Pr definition.
- Probe-window mean (pr_restore) is DIAGNOSTIC ONLY — never the kill decision.
- Secondary c(d): episodes to recover within ε=0.05 of pre-dormancy mean;
  censored=max; per-cell censor rate
- S = P(c ≤ B | store) − P(c ≤ B | never-learned);
  B = 0.5 * c_never, frozen once
"""

from __future__ import annotations

import dataclasses
import math
from typing import Optional

from fork1.schedule import ArmType, Task


EPSILON = 0.05
THETA = 0.80
PROBE_WINDOW_W = 2
GD_THRESHOLD = 0.25


@dataclasses.dataclass
class CellResult:
    """Result for one (family, arm_type, d, κ) cell."""
    family: str
    arm_type: ArmType
    d: int
    kappa: float
    seed: int
    pr_restore: float
    c_episodes: float
    censored: bool
    n_probes: int
    n_successes: int
    pre_dormancy_mean: float
    probe_outcomes: list[bool] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class GDResult:
    """Result of the G-D check for one κ value.

    pr_d0 / pr_d200 are P(c ≤ B) — the fraction of busy-arm runs whose
    recovery cost c_episodes is within the frozen budget B.
    probe_mean_d0 / probe_mean_d200 are the probe-window mean success
    rates, retained as DIAGNOSTIC fields only.
    """
    kappa: float
    pr_d0: float
    pr_d200: float
    gap: float
    gd_holds: bool
    probe_mean_d0: float = 0.0
    probe_mean_d200: float = 0.0


@dataclasses.dataclass
class SMetric:
    """S = P(c ≤ B | store) − P(c ≤ B | never-learned)."""
    family: str
    d: int
    kappa: float
    pr_store: float
    pr_never: float
    s_value: float
    b_frozen: float
    in_withheld_era: bool = False


@dataclasses.dataclass
class FailLadderDecision:
    """Output of the fail ladder."""
    step: int
    label: str
    detail: str


def compute_pr_restore(
    outcomes: list[bool],
    n_probes: int,
) -> float:
    """Restoration probability: fraction of probe successes.

    DIAGNOSTIC ONLY — not used for G-D or κ=1 flat kill decisions.
    """
    if n_probes == 0:
        return 0.0
    return sum(outcomes) / n_probes


def compute_pre_dormancy_mean(
    pre_dormancy_outcomes: list[bool],
) -> float:
    """Mean success rate during the active (pre-dormancy) block."""
    if not pre_dormancy_outcomes:
        return 0.0
    return sum(pre_dormancy_outcomes) / len(pre_dormancy_outcomes)


def compute_c_episodes(
    post_dormancy_outcomes: list[bool],
    pre_dormancy_mean: float,
    max_episodes: int,
) -> tuple[float, bool]:
    """Episodes to recover within ε=0.05 absolute of pre-dormancy mean.

    Returns (c, censored). If recovery never happens, c=max_episodes and
    censored=True.
    """
    if not post_dormancy_outcomes:
        return float(max_episodes), True

    running_sum = 0
    for i, outcome in enumerate(post_dormancy_outcomes, 1):
        running_sum += int(outcome)
        running_mean = running_sum / i
        if abs(running_mean - pre_dormancy_mean) <= EPSILON:
            return float(i), False

    return float(max_episodes), True


def evaluate_cell(
    probe_outcomes: list[bool],
    pre_dormancy_outcomes: list[bool],
    post_dormancy_outcomes: list[bool],
    family: str,
    arm_type: ArmType,
    d: int,
    kappa: float,
    seed: int,
    max_episodes: int,
) -> CellResult:
    """Evaluate all metrics for a single cell."""
    n_probes = len(probe_outcomes)
    n_successes = sum(probe_outcomes)
    pr_restore = compute_pr_restore(probe_outcomes, n_probes)
    pre_mean = compute_pre_dormancy_mean(pre_dormancy_outcomes)
    c_eps, censored = compute_c_episodes(
        post_dormancy_outcomes, pre_mean, max_episodes
    )
    return CellResult(
        family=family,
        arm_type=arm_type,
        d=d,
        kappa=kappa,
        seed=seed,
        pr_restore=pr_restore,
        c_episodes=c_eps,
        censored=censored,
        n_probes=n_probes,
        n_successes=n_successes,
        pre_dormancy_mean=pre_mean,
        probe_outcomes=list(probe_outcomes),
    )


def check_gd(
    results_by_d: dict[int, list[CellResult]],
    kappa: float,
    b_frozen: float,
) -> GDResult:
    """G-D check: P(c≤B|d=200) ≤ P(c≤B|d=0) − 0.25 at the given κ.

    Uses busy-arm results only.  The decision metric is P(c ≤ B) — the
    fraction of runs whose recovery cost c_episodes is within the frozen
    budget B.  Probe-window mean is computed alongside as a diagnostic.
    """
    def _busy_cells(d: int) -> list[CellResult]:
        return [r for r in results_by_d.get(d, []) if r.arm_type == ArmType.BUSY]

    cells_d0 = _busy_cells(0)
    cells_d200 = _busy_cells(200)

    pr_d0 = compute_p_recovery_within_b(cells_d0, b_frozen)
    pr_d200 = compute_p_recovery_within_b(cells_d200, b_frozen)
    gap = pr_d0 - pr_d200
    gd_holds = pr_d200 <= pr_d0 - GD_THRESHOLD

    probe_mean_d0 = (
        (sum(r.pr_restore for r in cells_d0) / len(cells_d0)) if cells_d0 else 0.0
    )
    probe_mean_d200 = (
        (sum(r.pr_restore for r in cells_d200) / len(cells_d200)) if cells_d200 else 0.0
    )

    return GDResult(
        kappa=kappa,
        pr_d0=pr_d0,
        pr_d200=pr_d200,
        gap=gap,
        gd_holds=gd_holds,
        probe_mean_d0=probe_mean_d0,
        probe_mean_d200=probe_mean_d200,
    )


def check_kappa1_flat(
    results_by_d: dict[int, list[CellResult]],
    b_frozen: float,
) -> bool:
    """Check that κ=1 produces flat restoration (no significant dormancy effect).

    Flat means P(c≤B|d=200) is NOT significantly below P(c≤B|d=0) — i.e.,
    the G-D gap does NOT hold at κ=1 under the same P(c≤B) definition.
    If G-D holds at κ=1, the control fails.
    """
    kappa1_results: dict[int, list[CellResult]] = {}
    for d, cells in results_by_d.items():
        k1_cells = [r for r in cells if abs(r.kappa - 1.0) < 1e-9]
        if k1_cells:
            kappa1_results[d] = k1_cells

    if not kappa1_results:
        return False

    gd = check_gd(kappa1_results, kappa=1.0, b_frozen=b_frozen)
    return not gd.gd_holds


def estimate_c_never(
    never_learned_results: list[CellResult],
) -> float:
    """Estimate c_never ONCE from never-learned arm results.

    This is the mean c (episodes to criterion) across all never-learned cells.
    Used to freeze B = 0.5 * c_never for ALL subsequent S, P(c ≤ B),
    and reacquisition budget computations.
    """
    if not never_learned_results:
        raise ValueError("No never-learned results to estimate c_never from")
    return sum(r.c_episodes for r in never_learned_results) / len(never_learned_results)


def freeze_b(c_never: float) -> float:
    """B = 0.5 * c_never, frozen for all cells."""
    return 0.5 * c_never


def compute_pr_within_b(
    post_dormancy_outcomes: list[bool],
    b: float,
) -> float:
    """Pr[restore within B episodes] — fraction of first B post-dormancy
    episodes that are successes.

    NOTE: This is a per-episode success rate, NOT the same as P(c ≤ B).
    For S metric computation, use compute_p_recovery_within_b() instead.
    """
    b_int = max(1, int(math.ceil(b)))
    truncated = post_dormancy_outcomes[:b_int]
    if not truncated:
        return 0.0
    return sum(truncated) / len(truncated)


def compute_p_recovery_within_b(
    cell_results: list[CellResult],
    b_frozen: float,
) -> float:
    """P(c ≤ B) — fraction of runs where recovery cost c ≤ B.

    Each CellResult has c_episodes (episodes to recover within ε of
    pre-dormancy mean). This returns the fraction of cells where that
    cost is ≤ the frozen budget B.

    Used for G-D primary, κ=1 flat, and S metric computations.
    """
    if not cell_results:
        return 0.0
    n_within = sum(1 for r in cell_results if r.c_episodes <= b_frozen)
    return n_within / len(cell_results)


def compute_s(
    store_results: list[CellResult],
    never_results: list[CellResult],
    b_frozen: float,
    family: str,
    d: int,
    kappa: float,
    in_withheld_era: bool = False,
) -> SMetric:
    """S = P(c ≤ B | store) − P(c ≤ B | never-learned).

    Each cell has a recovery cost c (c_episodes). P(c ≤ B) is the fraction
    of runs where c ≤ frozen B. This has the same sign as (c_never − c_store).
    """
    pr_store = compute_p_recovery_within_b(store_results, b_frozen)
    pr_never = compute_p_recovery_within_b(never_results, b_frozen)
    return SMetric(
        family=family,
        d=d,
        kappa=kappa,
        pr_store=pr_store,
        pr_never=pr_never,
        s_value=pr_store - pr_never,
        b_frozen=b_frozen,
        in_withheld_era=in_withheld_era,
    )


def censor_rate(results: list[CellResult]) -> float:
    """Per-cell censor rate: fraction of censored results."""
    if not results:
        return 0.0
    return sum(1 for r in results if r.censored) / len(results)
