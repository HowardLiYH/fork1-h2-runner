"""Metrics: G-D checker (primary), c(d) secondary, S harmfulness (rung-3).

Spec references:
- Primary G-D: Pr(d=200) ≤ Pr(d=0) − 0.25 at κ∈{0.25,0.50}; κ=1 flat
- Secondary c(d): episodes to recover within ε=0.05 of pre-dormancy mean;
  censored=max; per-cell censor rate
- S = Pr[restore within B|store] − Pr[restore within B|never-learned];
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
    """Result of the G-D check for one κ value."""
    kappa: float
    pr_d0: float
    pr_d200: float
    gap: float
    gd_holds: bool


@dataclasses.dataclass
class SMetric:
    """S = Pr[restore within B|store] − Pr[restore within B|never-learned]."""
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
    """Restoration probability: fraction of probe successes."""
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
) -> GDResult:
    """G-D check: Pr(d=200) ≤ Pr(d=0) − 0.25 at the given κ.

    Uses busy-arm results only.
    """
    def mean_pr(d: int) -> float:
        cells = [r for r in results_by_d.get(d, []) if r.arm_type == ArmType.BUSY]
        if not cells:
            return 0.0
        return sum(r.pr_restore for r in cells) / len(cells)

    pr_d0 = mean_pr(0)
    pr_d200 = mean_pr(200)
    gap = pr_d0 - pr_d200
    gd_holds = pr_d200 <= pr_d0 - GD_THRESHOLD
    return GDResult(
        kappa=kappa,
        pr_d0=pr_d0,
        pr_d200=pr_d200,
        gap=gap,
        gd_holds=gd_holds,
    )


def check_kappa1_flat(
    results_by_d: dict[int, list[CellResult]],
) -> bool:
    """Check that κ=1 produces flat restoration (no significant dormancy effect).

    Flat means Pr(d=200) is NOT significantly below Pr(d=0) — i.e., the G-D
    gap does NOT hold at κ=1. If G-D holds at κ=1, the control fails.
    """
    kappa1_results: dict[int, list[CellResult]] = {}
    for d, cells in results_by_d.items():
        k1_cells = [r for r in cells if abs(r.kappa - 1.0) < 1e-9]
        if k1_cells:
            kappa1_results[d] = k1_cells

    if not kappa1_results:
        return False

    gd = check_gd(kappa1_results, kappa=1.0)
    return not gd.gd_holds


def estimate_c_never(
    never_learned_results: list[CellResult],
) -> float:
    """Estimate c_never ONCE from never-learned arm results.

    This is the mean c (episodes to criterion) across all never-learned cells.
    Used to freeze B = 0.5 * c_never for ALL subsequent S, Pr[restore within B],
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
    episodes that are successes."""
    b_int = max(1, int(math.ceil(b)))
    truncated = post_dormancy_outcomes[:b_int]
    if not truncated:
        return 0.0
    return sum(truncated) / len(truncated)


def compute_s(
    store_post_outcomes: list[bool],
    never_post_outcomes: list[bool],
    b_frozen: float,
    family: str,
    d: int,
    kappa: float,
) -> SMetric:
    """S = Pr[restore within B|store] − Pr[restore within B|never-learned].

    Identical probes (same family, d, κ, seed).
    """
    pr_store = compute_pr_within_b(store_post_outcomes, b_frozen)
    pr_never = compute_pr_within_b(never_post_outcomes, b_frozen)
    return SMetric(
        family=family,
        d=d,
        kappa=kappa,
        pr_store=pr_store,
        pr_never=pr_never,
        s_value=pr_store - pr_never,
        b_frozen=b_frozen,
    )


def censor_rate(results: list[CellResult]) -> float:
    """Per-cell censor rate: fraction of censored results."""
    if not results:
        return 0.0
    return sum(1 for r in results if r.censored) / len(results)
