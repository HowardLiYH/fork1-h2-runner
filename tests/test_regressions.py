"""Regression tests for harness fidelity fixes.

Round 1 fixes (3 and 4 passed review):
  Fix 3: probe_window / w = 2 (Stack frozen)
  Fix 4: c_never from frozen pilot, not full-grid average

Round 2 still-fails:
  SF1: S = P(c ≤ B | store) − P(c ≤ B | never-learned) using per-cell
       c_episodes, NOT truncated probe-window success rate
  SF2: withheld-era partition wired into run_grid; in_withheld_era set
"""

from __future__ import annotations

import pytest

from fork1.fail_ladder import LadderOutcome, evaluate_fail_ladder
from fork1.grid import GridConfig, grid_output_to_json, run_grid, run_pilot_c_never
from fork1.metrics import (
    CellResult,
    SMetric,
    compute_p_recovery_within_b,
    compute_s,
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
) -> CellResult:
    n_probes = 2
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
        probe_outcomes=[True] * int(pr_restore * n_probes) + [False] * (n_probes - int(pr_restore * n_probes)),
    )


# ── SF1: S definition uses P(c ≤ B) from c_episodes ──────────────────────

class TestSF1_SUsesRecoveryCost:
    """S = P(c ≤ B | store) − P(c ≤ B | never-learned) via c_episodes."""

    def test_p_recovery_within_b_basic(self) -> None:
        """P(c ≤ B) is fraction of cells where c_episodes ≤ B."""
        cells = [
            _make_result(c_episodes=5.0),
            _make_result(c_episodes=8.0),
            _make_result(c_episodes=15.0),
            _make_result(c_episodes=20.0),
        ]
        assert compute_p_recovery_within_b(cells, b_frozen=10.0) == 0.5
        assert compute_p_recovery_within_b(cells, b_frozen=20.0) == 1.0
        assert compute_p_recovery_within_b(cells, b_frozen=4.0) == 0.0

    def test_s_same_sign_as_cost_difference(self) -> None:
        """S must have the same sign as (c_never − c_store).

        If store recovers faster (lower c), more store cells have c ≤ B,
        so P(c≤B|store) > P(c≤B|never) → S > 0 and c_never > c_store.
        """
        store_cells = [
            _make_result(c_episodes=3.0, seed=0),
            _make_result(c_episodes=5.0, seed=1),
            _make_result(c_episodes=7.0, seed=2),
        ]
        never_cells = [
            _make_result(c_episodes=12.0, seed=0, arm_type=ArmType.NEVER_LEARNED),
            _make_result(c_episodes=15.0, seed=1, arm_type=ArmType.NEVER_LEARNED),
            _make_result(c_episodes=18.0, seed=2, arm_type=ArmType.NEVER_LEARNED),
        ]
        b = 10.0
        s = compute_s(store_cells, never_cells, b, "earnings", 200, 0.5)
        avg_c_store = sum(r.c_episodes for r in store_cells) / len(store_cells)
        avg_c_never = sum(r.c_episodes for r in never_cells) / len(never_cells)
        cost_diff = avg_c_never - avg_c_store
        assert cost_diff > 0
        assert s.s_value > 0

    def test_s_negative_when_store_worse(self) -> None:
        """If store has higher c (slower recovery), S < 0."""
        store_cells = [
            _make_result(c_episodes=15.0, seed=0),
            _make_result(c_episodes=20.0, seed=1),
        ]
        never_cells = [
            _make_result(c_episodes=5.0, seed=0, arm_type=ArmType.NEVER_LEARNED),
            _make_result(c_episodes=3.0, seed=1, arm_type=ArmType.NEVER_LEARNED),
        ]
        b = 10.0
        s = compute_s(store_cells, never_cells, b, "crisis", 200, 0.25)
        assert s.s_value < 0

    def test_s_is_not_probe_window_success_rate(self) -> None:
        """S must NOT be computed from probe_outcomes truncated to B.
        It must use c_episodes (recovery cost)."""
        store_cells = [
            _make_result(c_episodes=5.0, seed=0),
            _make_result(c_episodes=50.0, seed=1),
        ]
        never_cells = [
            _make_result(c_episodes=100.0, seed=0, arm_type=ArmType.NEVER_LEARNED),
            _make_result(c_episodes=100.0, seed=1, arm_type=ArmType.NEVER_LEARNED),
        ]
        b = 10.0
        s = compute_s(store_cells, never_cells, b, "earnings", 200, 0.5)
        assert s.pr_store == 0.5
        assert s.pr_never == 0.0
        assert s.s_value == 0.5

    def test_grid_s_uses_c_episodes(self) -> None:
        """Grid run S metrics must reflect P(c ≤ B), not probe success rate."""
        config = GridConfig(n_seeds=2, n_instances=5)
        output = run_grid(config)
        for s in output.s_metrics:
            assert 0.0 <= s.pr_store <= 1.0
            assert 0.0 <= s.pr_never <= 1.0
            assert abs(s.s_value - (s.pr_store - s.pr_never)) < 1e-9


# ── SF2: withheld-era partition wired in run_grid ─────────────────────────

class TestSF2_WithheldEraInGrid:
    """run_grid must produce S metrics with in_withheld_era=True."""

    def test_grid_produces_withheld_era_s_metrics(self) -> None:
        """run_grid must emit at least one S metric with in_withheld_era=True
        when the grid has enough seeds to cover withheld_era_seeds."""
        config = GridConfig(n_seeds=8, n_instances=5, withheld_era_seeds=(6, 7))
        output = run_grid(config)
        withheld_s = [s for s in output.s_metrics if s.in_withheld_era]
        primary_s = [s for s in output.s_metrics if not s.in_withheld_era]
        assert len(withheld_s) > 0, "No withheld-era S metrics produced"
        assert len(primary_s) > 0, "No primary-era S metrics produced"

    def test_withheld_era_partition_is_frozen_in_config(self) -> None:
        config = GridConfig()
        assert hasattr(config, "withheld_era_seeds")
        assert len(config.withheld_era_seeds) > 0

    def test_withheld_era_documented_in_json(self) -> None:
        config = GridConfig(n_seeds=2, n_instances=5)
        output = run_grid(config)
        j = grid_output_to_json(output)
        assert "withheld_era_seeds" in j
        assert j["withheld_era_seeds"] == list(config.withheld_era_seeds)

    def test_rung3_only_fires_on_withheld_s(self) -> None:
        """Fail ladder harmfulness requires in_withheld_era=True S metrics."""
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
        s_primary_only = [SMetric(
            family="earnings", d=200, kappa=0.25,
            pr_store=0.2, pr_never=0.8, s_value=-0.6, b_frozen=10.0,
            in_withheld_era=False,
        )]
        decision = evaluate_fail_ladder(results_by_d, s_metrics=s_primary_only)
        assert decision.label != LadderOutcome.HARMFULNESS.value

        s_withheld = [SMetric(
            family="earnings", d=200, kappa=0.25,
            pr_store=0.2, pr_never=0.8, s_value=-0.6, b_frozen=10.0,
            in_withheld_era=True,
        )]
        decision2 = evaluate_fail_ladder(results_by_d, s_metrics=s_withheld)
        assert decision2.label == LadderOutcome.HARMFULNESS.value

    def test_no_withheld_s_when_seeds_below_partition(self) -> None:
        """With only 1 seed and withheld_era_seeds=(6,7), no withheld S."""
        config = GridConfig(n_seeds=1, n_instances=5, withheld_era_seeds=(6, 7))
        output = run_grid(config)
        withheld_s = [s for s in output.s_metrics if s.in_withheld_era]
        assert len(withheld_s) == 0


# ── Fix 3 (passed review): probe_window = 2 ──────────────────────────────

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


# ── Fix 4 (passed review): c_never from pilot ────────────────────────────

class TestFix4_CNeverFromPilot:
    """c_never estimated ONCE from frozen pilot ceiling, not full-grid average.
    Pilot sizes locked at 30 instances × 4 seeds."""

    def test_pilot_sizes_locked(self) -> None:
        """Experiment Design locked pilot at 30 instances × 4 seeds."""
        config = GridConfig()
        assert config.pilot_instances == 30
        assert config.pilot_seeds == 4

    def test_pilot_runs_separate_from_grid(self) -> None:
        config = GridConfig(n_seeds=1, n_instances=5)
        adapter = ModelAdapterStub()
        c_never = run_pilot_c_never(config, adapter)
        assert c_never > 0

    def test_pilot_c_never_locked_for_all_cells(self) -> None:
        config = GridConfig(n_seeds=1, n_instances=5)
        output = run_grid(config)
        assert output.b_frozen == freeze_b(output.c_never)
        for s in output.s_metrics:
            assert abs(s.b_frozen - output.b_frozen) < 1e-9

    def test_json_reports_pilot_source_and_sizes(self) -> None:
        config = GridConfig(n_seeds=1, n_instances=5)
        output = run_grid(config)
        j = grid_output_to_json(output)
        assert j["c_never_source"] == "pilot"
        assert j["pilot"]["instances"] == config.pilot_instances
        assert j["pilot"]["seeds"] == config.pilot_seeds
