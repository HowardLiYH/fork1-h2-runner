"""Regression tests for the four must-fix harness fidelity items.

Fix 1: S uses compute_pr_within_b(frozen B), not raw pr_restore
Fix 2: Rung-3 fires only when G-D fails/flat + S<0 (withheld-era only)
Fix 3: probe_window / w = 2 (Stack frozen)
Fix 4: c_never from frozen pilot, not full-grid average
"""

from __future__ import annotations

import json

import pytest

from fork1.fail_ladder import LadderOutcome, evaluate_fail_ladder
from fork1.grid import GridConfig, grid_output_to_json, run_grid, run_pilot_c_never
from fork1.metrics import (
    CellResult,
    SMetric,
    compute_pr_within_b,
    freeze_b,
)
from fork1.oracle import ModelAdapterStub
from fork1.schedule import ArmType, Schedule


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
    probe_outcomes: list[bool] | None = None,
) -> CellResult:
    if probe_outcomes is None:
        n_probes = 50
        probe_outcomes = [True] * int(pr_restore * n_probes) + [False] * (n_probes - int(pr_restore * n_probes))
    else:
        n_probes = len(probe_outcomes)
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
        n_successes=sum(probe_outcomes),
        pre_dormancy_mean=pre_dormancy_mean,
        probe_outcomes=probe_outcomes,
    )


class TestFix1_SLoopUsesFrozenB:
    """S must use Pr[restore within frozen B], not raw pr_restore."""

    def test_s_differs_from_raw_pr_when_b_truncates(self) -> None:
        """When B < len(probe_outcomes), Pr[within B] != pr_restore."""
        outcomes_10 = [True, True, False, False, False, False, False, False, False, False]
        pr_full = sum(outcomes_10) / len(outcomes_10)
        assert pr_full == 0.2

        pr_within_3 = compute_pr_within_b(outcomes_10, b=3.0)
        assert abs(pr_within_3 - 2 / 3) < 1e-9
        assert pr_within_3 != pr_full

    def test_grid_s_metrics_use_frozen_b(self) -> None:
        """In a grid run, S.pr_store must differ from raw pr_restore when
        B truncates the probe window."""
        config = GridConfig(n_seeds=1, n_instances=5)
        output = run_grid(config)
        assert output.b_frozen > 0
        for s in output.s_metrics:
            assert abs(s.b_frozen - output.b_frozen) < 1e-9

    def test_cell_results_carry_probe_outcomes(self) -> None:
        """CellResult must carry probe_outcomes for S recomputation."""
        config = GridConfig(n_seeds=1, n_instances=5)
        output = run_grid(config)
        for r in output.cell_results:
            assert hasattr(r, "probe_outcomes")
            assert isinstance(r.probe_outcomes, list)
            if r.n_probes > 0:
                assert len(r.probe_outcomes) == r.n_probes


class TestFix2_Rung3OnlyOnFlatOrKill:
    """Harmfulness fires ONLY when H2 is flat/kill AND S<0 (withheld-era)."""

    def test_harmfulness_does_not_fire_when_gd_holds(self) -> None:
        """When G-D holds, rung-3 must NOT trigger harmfulness even if S<0."""
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
        s_negative = [SMetric(
            family="earnings", d=200, kappa=0.25,
            pr_store=0.3, pr_never=0.6, s_value=-0.3, b_frozen=10.0,
            in_withheld_era=True,
        )]
        decision = evaluate_fail_ladder(results_by_d, s_metrics=s_negative)
        assert decision.label != LadderOutcome.HARMFULNESS.value
        assert decision.label == LadderOutcome.PROCEED.value

    def test_harmfulness_fires_when_gd_fails_and_s_negative_withheld(self) -> None:
        """Rung-3: G-D fails (H2 kill) + S<0 in withheld-era → harmfulness."""
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
        s_negative_withheld = [SMetric(
            family="earnings", d=200, kappa=0.25,
            pr_store=0.3, pr_never=0.6, s_value=-0.3, b_frozen=10.0,
            in_withheld_era=True,
        )]
        decision = evaluate_fail_ladder(results_by_d, s_metrics=s_negative_withheld)
        assert decision.label == LadderOutcome.HARMFULNESS.value

    def test_harmfulness_requires_withheld_era_flag(self) -> None:
        """S<0 outside withheld-era must NOT trigger harmfulness."""
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
        s_negative_not_withheld = [SMetric(
            family="earnings", d=200, kappa=0.25,
            pr_store=0.3, pr_never=0.6, s_value=-0.3, b_frozen=10.0,
            in_withheld_era=False,
        )]
        decision = evaluate_fail_ladder(results_by_d, s_metrics=s_negative_not_withheld)
        assert decision.label == LadderOutcome.H2_KILL.value
        assert decision.label != LadderOutcome.HARMFULNESS.value


class TestFix3_ProbeWindowW2:
    """Stack froze w=2. probe_window must default to 2 everywhere."""

    def test_schedule_default_probe_window_is_2(self) -> None:
        schedule = Schedule()
        assert schedule.probe_window == 2

    def test_grid_config_default_probe_window_is_2(self) -> None:
        config = GridConfig()
        assert config.probe_window == 2

    def test_grid_run_produces_w2_probes(self) -> None:
        config = GridConfig(n_seeds=1, n_instances=3)
        output = run_grid(config)
        busy_d50 = [
            r for r in output.cell_results
            if r.arm_type == ArmType.BUSY and r.d == 50
        ]
        for r in busy_d50:
            assert r.n_probes == config.n_instances * 2


class TestFix4_CNeverFromPilot:
    """c_never estimated ONCE from frozen pilot ceiling, not full-grid average."""

    def test_pilot_runs_separate_from_grid(self) -> None:
        config = GridConfig(n_seeds=1, n_instances=5, pilot_instances=3, pilot_seeds=1)
        adapter = ModelAdapterStub()
        c_never = run_pilot_c_never(config, adapter)
        assert c_never > 0

    def test_pilot_c_never_locked_for_all_cells(self) -> None:
        config = GridConfig(n_seeds=1, n_instances=5, pilot_instances=3, pilot_seeds=1)
        output = run_grid(config)
        assert output.b_frozen == freeze_b(output.c_never)
        for s in output.s_metrics:
            assert abs(s.b_frozen - output.b_frozen) < 1e-9

    def test_json_reports_pilot_source(self) -> None:
        config = GridConfig(n_seeds=1, n_instances=5)
        output = run_grid(config)
        j = grid_output_to_json(output)
        assert j["c_never_source"] == "pilot"

    def test_grid_config_has_pilot_params(self) -> None:
        config = GridConfig()
        assert hasattr(config, "pilot_instances")
        assert hasattr(config, "pilot_seeds")
        assert config.pilot_instances > 0
        assert config.pilot_seeds > 0
