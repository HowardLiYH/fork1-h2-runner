"""Tests for metrics: G-D checker, c(d) secondary, S metric, fail ladder."""

from __future__ import annotations

import pytest

from fork1.fail_ladder import LadderOutcome, evaluate_fail_ladder
from fork1.metrics import (
    CellResult,
    SMetric,
    censor_rate,
    check_gd,
    check_kappa1_flat,
    compute_c_episodes,
    compute_p_recovery_within_b,
    compute_pr_restore,
    compute_pr_within_b,
    compute_pre_dormancy_mean,
    compute_s,
    estimate_c_never,
    freeze_b,
)
from fork1.schedule import ArmType


def _make_result(
    family: str = "earnings",
    arm_type: ArmType = ArmType.BUSY,
    d: int = 0,
    kappa: float = 0.5,
    seed: int = 0,
    pr_restore: float = 0.8,
    c_episodes: float = 10.0,
    censored: bool = False,
    pre_dormancy_mean: float = 0.85,
) -> CellResult:
    n_probes = 50
    return CellResult(
        family=family,
        arm_type=arm_type,
        d=d,
        kappa=kappa,
        seed=seed,
        pr_restore=pr_restore,
        c_episodes=c_episodes,
        censored=censored,
        n_probes=n_probes,
        n_successes=int(pr_restore * n_probes),
        pre_dormancy_mean=pre_dormancy_mean,
    )


class TestPrRestore:
    def test_all_success(self) -> None:
        assert compute_pr_restore([True] * 10, 10) == 1.0

    def test_all_fail(self) -> None:
        assert compute_pr_restore([False] * 10, 10) == 0.0

    def test_half(self) -> None:
        assert compute_pr_restore([True, False] * 5, 10) == 0.5

    def test_empty(self) -> None:
        assert compute_pr_restore([], 0) == 0.0


class TestPreDormancyMean:
    def test_basic(self) -> None:
        assert compute_pre_dormancy_mean([True, True, False, True]) == 0.75

    def test_empty(self) -> None:
        assert compute_pre_dormancy_mean([]) == 0.0


class TestCEpisodes:
    def test_immediate_recovery(self) -> None:
        outcomes = [True, False, True, True, False, True, True, True, False, True]
        c, censored = compute_c_episodes(
            outcomes, pre_dormancy_mean=0.70, max_episodes=100
        )
        assert not censored
        assert c <= len(outcomes)

    def test_censored(self) -> None:
        c, censored = compute_c_episodes(
            [False] * 50, pre_dormancy_mean=0.80, max_episodes=50
        )
        assert censored
        assert c == 50.0

    def test_empty_outcomes(self) -> None:
        c, censored = compute_c_episodes(
            [], pre_dormancy_mean=0.80, max_episodes=100
        )
        assert censored
        assert c == 100.0


class TestGDChecker:
    def test_gd_holds_when_big_gap(self) -> None:
        results_by_d = {
            0: [_make_result(d=0, kappa=0.5, pr_restore=0.9)],
            200: [_make_result(d=200, kappa=0.5, pr_restore=0.5)],
        }
        gd = check_gd(results_by_d, kappa=0.5)
        assert gd.gd_holds
        assert gd.gap >= 0.25

    def test_gd_fails_when_small_gap(self) -> None:
        results_by_d = {
            0: [_make_result(d=0, kappa=0.5, pr_restore=0.8)],
            200: [_make_result(d=200, kappa=0.5, pr_restore=0.7)],
        }
        gd = check_gd(results_by_d, kappa=0.5)
        assert not gd.gd_holds

    def test_gd_uses_busy_only(self) -> None:
        results_by_d = {
            0: [
                _make_result(d=0, kappa=0.5, pr_restore=0.9, arm_type=ArmType.BUSY),
                _make_result(d=0, kappa=0.5, pr_restore=0.1, arm_type=ArmType.IDLE),
            ],
            200: [
                _make_result(d=200, kappa=0.5, pr_restore=0.5, arm_type=ArmType.BUSY),
                _make_result(d=200, kappa=0.5, pr_restore=0.9, arm_type=ArmType.IDLE),
            ],
        }
        gd = check_gd(results_by_d, kappa=0.5)
        assert gd.pr_d0 == 0.9
        assert gd.pr_d200 == 0.5


class TestKappa1Flat:
    def test_flat_when_no_gap(self) -> None:
        results_by_d = {
            0: [_make_result(d=0, kappa=1.0, pr_restore=0.8)],
            200: [_make_result(d=200, kappa=1.0, pr_restore=0.78)],
        }
        assert check_kappa1_flat(results_by_d)

    def test_not_flat_when_gap(self) -> None:
        results_by_d = {
            0: [_make_result(d=0, kappa=1.0, pr_restore=0.9)],
            200: [_make_result(d=200, kappa=1.0, pr_restore=0.5)],
        }
        assert not check_kappa1_flat(results_by_d)


class TestFreezeB:
    def test_b_frozen_formula(self) -> None:
        assert freeze_b(20.0) == 10.0
        assert freeze_b(100.0) == 50.0

    def test_c_never_estimation(self) -> None:
        results = [
            _make_result(arm_type=ArmType.NEVER_LEARNED, c_episodes=20.0),
            _make_result(arm_type=ArmType.NEVER_LEARNED, c_episodes=30.0),
        ]
        c_never = estimate_c_never(results)
        assert c_never == 25.0
        assert freeze_b(c_never) == 12.5

    def test_empty_never_raises(self) -> None:
        with pytest.raises(ValueError):
            estimate_c_never([])


class TestSMetric:
    def test_s_positive(self) -> None:
        store = [
            _make_result(c_episodes=5.0, seed=0),
            _make_result(c_episodes=7.0, seed=1),
        ]
        never = [
            _make_result(c_episodes=15.0, seed=0, arm_type=ArmType.NEVER_LEARNED),
            _make_result(c_episodes=20.0, seed=1, arm_type=ArmType.NEVER_LEARNED),
        ]
        s = compute_s(store, never, 10.0, "earnings", 200, 0.5)
        assert s.s_value > 0
        assert s.pr_store > s.pr_never

    def test_s_negative(self) -> None:
        store = [
            _make_result(c_episodes=20.0, seed=0),
            _make_result(c_episodes=25.0, seed=1),
        ]
        never = [
            _make_result(c_episodes=3.0, seed=0, arm_type=ArmType.NEVER_LEARNED),
            _make_result(c_episodes=5.0, seed=1, arm_type=ArmType.NEVER_LEARNED),
        ]
        s = compute_s(store, never, 10.0, "crisis", 200, 0.25)
        assert s.s_value < 0

    def test_s_sign_matches_cost_form(self) -> None:
        """S > 0 iff store recovers faster (lower c) → same sign as c_never − c_store."""
        store = [
            _make_result(c_episodes=4.0, seed=0),
            _make_result(c_episodes=6.0, seed=1),
            _make_result(c_episodes=8.0, seed=2),
        ]
        never = [
            _make_result(c_episodes=12.0, seed=0, arm_type=ArmType.NEVER_LEARNED),
            _make_result(c_episodes=15.0, seed=1, arm_type=ArmType.NEVER_LEARNED),
            _make_result(c_episodes=18.0, seed=2, arm_type=ArmType.NEVER_LEARNED),
        ]
        s = compute_s(store, never, 10.0, "earnings", 200, 0.5)
        assert s.s_value > 0
        avg_c_store = sum(r.c_episodes for r in store) / len(store)
        avg_c_never = sum(r.c_episodes for r in never) / len(never)
        assert (avg_c_never - avg_c_store) > 0


class TestCensorRate:
    def test_no_censored(self) -> None:
        results = [_make_result(censored=False)] * 5
        assert censor_rate(results) == 0.0

    def test_all_censored(self) -> None:
        results = [_make_result(censored=True)] * 5
        assert censor_rate(results) == 1.0

    def test_mixed(self) -> None:
        results = [_make_result(censored=True)] * 2 + [_make_result(censored=False)] * 3
        assert abs(censor_rate(results) - 0.4) < 1e-9

    def test_empty(self) -> None:
        assert censor_rate([]) == 0.0


class TestFailLadder:
    def test_proceed_when_gd_holds_and_flat(self) -> None:
        results_by_d = {
            0: [
                _make_result(d=0, kappa=0.25, pr_restore=0.9),
                _make_result(d=0, kappa=0.50, pr_restore=0.9),
                _make_result(d=0, kappa=1.0, pr_restore=0.8),
            ],
            200: [
                _make_result(d=200, kappa=0.25, pr_restore=0.5),
                _make_result(d=200, kappa=0.50, pr_restore=0.5),
                _make_result(d=200, kappa=1.0, pr_restore=0.78),
            ],
        }
        decision = evaluate_fail_ladder(results_by_d)
        assert decision.label == LadderOutcome.PROCEED.value

    def test_h2_kill_when_gd_fails(self) -> None:
        results_by_d = {
            0: [
                _make_result(d=0, kappa=0.25, pr_restore=0.8),
                _make_result(d=0, kappa=0.50, pr_restore=0.8),
                _make_result(d=0, kappa=1.0, pr_restore=0.8),
            ],
            200: [
                _make_result(d=200, kappa=0.25, pr_restore=0.7),
                _make_result(d=200, kappa=0.50, pr_restore=0.7),
                _make_result(d=200, kappa=1.0, pr_restore=0.78),
            ],
        }
        decision = evaluate_fail_ladder(results_by_d)
        assert decision.label == LadderOutcome.H2_KILL.value

    def test_harmfulness_when_flat_and_s_negative(self) -> None:
        """Rung-3 only fires when G-D fails (H2 flat/kill) AND S<0 in
        withheld-era — NOT when G-D holds."""
        results_by_d = {
            0: [
                _make_result(d=0, kappa=0.25, pr_restore=0.8),
                _make_result(d=0, kappa=0.50, pr_restore=0.8),
                _make_result(d=0, kappa=1.0, pr_restore=0.8),
            ],
            200: [
                _make_result(d=200, kappa=0.25, pr_restore=0.7),
                _make_result(d=200, kappa=0.50, pr_restore=0.7),
                _make_result(d=200, kappa=1.0, pr_restore=0.78),
            ],
        }
        s_negative = [SMetric(
            family="earnings", d=200, kappa=0.25,
            pr_store=0.3, pr_never=0.6, s_value=-0.3, b_frozen=10.0,
            in_withheld_era=True,
        )]
        decision = evaluate_fail_ladder(results_by_d, s_metrics=s_negative)
        assert decision.label == LadderOutcome.HARMFULNESS.value

    def test_harness_bug_when_kappa1_rises(self) -> None:
        results_by_d = {
            0: [
                _make_result(d=0, kappa=0.25, pr_restore=0.9),
                _make_result(d=0, kappa=0.50, pr_restore=0.9),
                _make_result(d=0, kappa=1.0, pr_restore=0.5),
            ],
            200: [
                _make_result(d=200, kappa=0.25, pr_restore=0.5),
                _make_result(d=200, kappa=0.50, pr_restore=0.5),
                _make_result(d=200, kappa=1.0, pr_restore=0.8),
            ],
        }
        decision = evaluate_fail_ladder(results_by_d)
        assert decision.label == LadderOutcome.HARNESS_BUG.value
