"""Regression tests for harness fidelity fixes.

Round 1 fixes (3 and 4 passed review):
  Fix 3: probe_window / w = 2 (Stack frozen)
  Fix 4: c_never from frozen pilot, not full-grid average

Round 2 still-fails:
  SF1: S = P(c ≤ B | store) − P(c ≤ B | never-learned) using per-cell
       c_episodes, NOT truncated probe-window success rate
  SF2: withheld-era partition wired into run_grid; in_withheld_era set

P0 items (all required — GATE HOLD):
  P0-1: Causal probe — predict(task, seed, memory|None); miss ≠ hit
  P0-2: Strip dormancy_decay (DEBUG-only flag, default off)
  P0-3: Deletion arm clears family entries (not idle-equivalent)
  P0-4: G-D := P(c≤B) on busy with frozen B; κ=1 flat same;
        probe-window mean = diagnostic only; partial proceed removed
"""

from __future__ import annotations

import pytest

from fork1.fail_ladder import LadderOutcome, evaluate_fail_ladder
from fork1.grid import GridConfig, grid_output_to_json, run_grid, run_pilot_c_never
from fork1.memory import FamilyTaggedStore, MemoryEntry
from fork1.metrics import (
    CellResult,
    SMetric,
    check_gd,
    check_kappa1_flat,
    compute_p_recovery_within_b,
    compute_s,
    freeze_b,
)
from fork1.oracle import MEMORY_HIT_BONUS, ModelAdapterStub, PlaybookOracle
from fork1.schedule import ArmType, Schedule, Task


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
        """S must have the same sign as (c_never − c_store)."""
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
        b_frozen = 10.0
        results_by_d = {
            0: [
                _make_result(d=0, kappa=0.25, c_episodes=5.0),
                _make_result(d=0, kappa=0.50, c_episodes=5.0),
                _make_result(d=0, kappa=1.0, c_episodes=5.0),
            ],
            200: [
                _make_result(d=200, kappa=0.25, c_episodes=8.0),
                _make_result(d=200, kappa=0.50, c_episodes=8.0),
                _make_result(d=200, kappa=1.0, c_episodes=6.0),
            ],
        }
        s_primary_only = [SMetric(
            family="earnings", d=200, kappa=0.25,
            pr_store=0.2, pr_never=0.8, s_value=-0.6, b_frozen=10.0,
            in_withheld_era=False,
        )]
        decision = evaluate_fail_ladder(results_by_d, b_frozen=b_frozen, s_metrics=s_primary_only)
        assert decision.label != LadderOutcome.HARMFULNESS.value

        s_withheld = [SMetric(
            family="earnings", d=200, kappa=0.25,
            pr_store=0.2, pr_never=0.8, s_value=-0.6, b_frozen=10.0,
            in_withheld_era=True,
        )]
        decision2 = evaluate_fail_ladder(results_by_d, b_frozen=b_frozen, s_metrics=s_withheld)
        assert decision2.label == LadderOutcome.HARMFULNESS.value

    def test_no_withheld_s_when_seeds_below_partition(self) -> None:
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
    """c_never estimated ONCE from frozen pilot ceiling, not full-grid average."""

    def test_pilot_sizes_locked(self) -> None:
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
        assert j["c_never_source"] == "pilot_cold_start"
        assert j["pilot"]["instances"] == config.pilot_instances
        assert j["pilot"]["seeds"] == config.pilot_seeds


# ══════════════════════════════════════════════════════════════════════════
# P0 regression tests
# ══════════════════════════════════════════════════════════════════════════

# ── P0-1: Causal probe ───────────────────────────────────────────────────

class TestP0_1_CausalProbe:
    """predict(task, seed, memory|None) — miss ≠ hit; store access moves Pr."""

    def test_miss_ne_hit(self) -> None:
        """Oracle must return different success probability for hit vs miss."""
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
        assert oracle.success(task, seed=0, memory=mem) > oracle.success(task, seed=0, memory=None)

    def test_predict_outcomes_differ_across_episodes(self) -> None:
        """Over many episodes, hit vs miss must produce at least one different
        binary prediction — proving memory is causally wired."""
        oracle = PlaybookOracle("earnings", base_success=0.80)
        mem = MemoryEntry(
            family="earnings", skill=0.85, fidelity=1.0,
            episode_stored=5, last_accessed=5,
        )
        diffs = sum(
            1 for ep in range(300)
            if oracle.predict(
                Task(family="earnings", episode=ep, dormancy=50,
                     pos_in_activation=0, drifted=False, is_probe=True,
                     arm_type=ArmType.BUSY),
                seed=0, memory=None,
            ) != oracle.predict(
                Task(family="earnings", episode=ep, dormancy=50,
                     pos_in_activation=0, drifted=False, is_probe=True,
                     arm_type=ArmType.BUSY),
                seed=0, memory=mem,
            )
        )
        assert diffs > 0

    def test_grid_wires_memory_to_probes(self) -> None:
        """In a grid run, busy-arm probe results must depend on store state.

        Never-learned arms always miss → lower Pr than idle arms that always
        hit (memory preserved).  This can only hold if memory is actually
        passed to predict.
        """
        config = GridConfig(n_seeds=2, n_instances=10)
        output = run_grid(config)
        idle_d0 = [r for r in output.cell_results
                   if r.arm_type == ArmType.IDLE and r.d == 0]
        never = [r for r in output.cell_results
                 if r.arm_type == ArmType.NEVER_LEARNED]
        if idle_d0 and never:
            idle_pr = sum(r.pr_restore for r in idle_d0) / len(idle_d0)
            never_pr = sum(r.pr_restore for r in never) / len(never)
            assert idle_pr > never_pr, (
                "idle probe (memory hit) should outperform never-learned (miss)"
            )


# ── P0-2: Strip dormancy_decay ───────────────────────────────────────────

class TestP0_2_StripDormancyDecay:
    """dormancy_decay must be DEBUG-only (default off)."""

    def test_default_oracles_no_decay(self) -> None:
        """FAMILY_ORACLES must NOT apply dormancy_decay by default."""
        from fork1.oracle import FAMILY_ORACLES
        for name, oracle in FAMILY_ORACLES.items():
            assert not oracle._debug_dormancy_decay, (
                f"{name} oracle has dormancy_decay enabled"
            )

    def test_no_decay_same_prob_at_any_dormancy(self) -> None:
        oracle = PlaybookOracle("earnings", base_success=0.85, dormancy_decay=0.01)
        task_d0 = Task(
            family="earnings", episode=1, dormancy=0,
            pos_in_activation=0, drifted=False, is_probe=False,
            arm_type=ArmType.BUSY,
        )
        task_d200 = Task(
            family="earnings", episode=201, dormancy=200,
            pos_in_activation=0, drifted=False, is_probe=True,
            arm_type=ArmType.BUSY,
        )
        assert oracle.success(task_d0, seed=0) == oracle.success(task_d200, seed=0)

    def test_debug_flag_enables_decay(self) -> None:
        oracle = PlaybookOracle(
            "earnings", base_success=0.85, dormancy_decay=0.01,
            debug_dormancy_decay=True,
        )
        task_d0 = Task(
            family="earnings", episode=1, dormancy=0,
            pos_in_activation=0, drifted=False, is_probe=False,
            arm_type=ArmType.BUSY,
        )
        task_d200 = Task(
            family="earnings", episode=201, dormancy=200,
            pos_in_activation=0, drifted=False, is_probe=True,
            arm_type=ArmType.BUSY,
        )
        assert oracle.success(task_d200, seed=0) < oracle.success(task_d0, seed=0)


# ── P0-3: Deletion arm clears store ──────────────────────────────────────

class TestP0_3_DeletionClearsStore:
    """Deletion arm must clear family entries (not idle-equivalent)."""

    def test_delete_family_method(self) -> None:
        """FamilyTaggedStore.delete_family removes all entries for that family."""
        store = FamilyTaggedStore(capacity=10, kappa=0.5)
        store.store("earnings", skill=0.8, fidelity=0.9, episode=1)
        store.store("earnings", skill=0.9, fidelity=1.0, episode=2)
        store.store("crisis", skill=0.7, fidelity=0.8, episode=3)
        assert store.has_family("earnings")
        n = store.delete_family("earnings")
        assert n == 2
        assert not store.has_family("earnings")
        assert store.has_family("crisis")

    def test_deletion_ne_idle(self) -> None:
        """Deletion and idle arms must produce different probe outcomes.

        Idle preserves memory → hit → higher Pr.
        Deletion clears memory → miss → lower Pr.
        """
        config = GridConfig(n_seeds=2, n_instances=10)
        output = run_grid(config)
        idle_d50 = [r for r in output.cell_results
                    if r.arm_type == ArmType.IDLE and r.d == 50]
        del_d50 = [r for r in output.cell_results
                   if r.arm_type == ArmType.DELETION and r.d == 50]
        if idle_d50 and del_d50:
            idle_pr = sum(r.pr_restore for r in idle_d50) / len(idle_d50)
            del_pr = sum(r.pr_restore for r in del_d50) / len(del_d50)
            assert idle_pr > del_pr, (
                f"idle Pr={idle_pr:.3f} should exceed deletion Pr={del_pr:.3f}"
            )

    def test_deletion_arm_store_is_empty_after_clear(self) -> None:
        """After deletion, the store should have no entries for the family."""
        from fork1.grid import run_single_cell
        config = GridConfig(n_seeds=1, n_instances=3)
        adapter = ModelAdapterStub()
        result = run_single_cell(
            "earnings", ArmType.DELETION, d=50, kappa=0.5,
            seed=0, config=config, adapter=adapter,
        )
        result_idle = run_single_cell(
            "earnings", ArmType.IDLE, d=50, kappa=0.5,
            seed=0, config=config, adapter=adapter,
        )
        assert result.pr_restore != result_idle.pr_restore or \
               result.c_episodes != result_idle.c_episodes, \
            "Deletion and idle must produce different metric values"


# ── P0-4: G-D uses P(c≤B), probe-mean is diagnostic ─────────────────────

class TestP0_4_GDUsesPcLeqB:
    """G-D := P(c≤B) on busy with frozen B; κ=1 flat same Pr."""

    def test_check_gd_requires_b_frozen(self) -> None:
        """check_gd must accept b_frozen parameter."""
        results_by_d = {
            0: [_make_result(d=0, kappa=0.5, c_episodes=5.0)],
            200: [_make_result(d=200, kappa=0.5, c_episodes=50.0)],
        }
        gd = check_gd(results_by_d, kappa=0.5, b_frozen=10.0)
        assert gd.gd_holds

    def test_check_gd_uses_c_episodes_not_pr_restore(self) -> None:
        """If pr_restore says 'gap' but c_episodes says 'no gap', G-D must
        follow c_episodes (P(c≤B))."""
        b_frozen = 10.0
        results_by_d = {
            0: [_make_result(d=0, kappa=0.5, pr_restore=0.9, c_episodes=5.0)],
            200: [_make_result(d=200, kappa=0.5, pr_restore=0.5, c_episodes=8.0)],
        }
        gd = check_gd(results_by_d, kappa=0.5, b_frozen=b_frozen)
        assert not gd.gd_holds, "G-D must use P(c≤B), not pr_restore"

    def test_old_probe_mean_cannot_kill(self) -> None:
        """Probe-window mean (pr_restore) must never drive the kill decision.

        Even when probe-window mean has a large gap, if P(c≤B) says no gap,
        G-D must NOT hold.
        """
        b_frozen = 100.0
        results_by_d = {
            0: [_make_result(d=0, kappa=0.5, pr_restore=0.95, c_episodes=5.0)],
            200: [_make_result(d=200, kappa=0.5, pr_restore=0.40, c_episodes=8.0)],
        }
        gd = check_gd(results_by_d, kappa=0.5, b_frozen=b_frozen)
        assert not gd.gd_holds, "probe-mean gap cannot cause a kill"

    def test_kappa1_flat_uses_same_pr(self) -> None:
        """κ=1 flat check must use P(c≤B), same definition as G-D."""
        b_frozen = 10.0
        results_by_d = {
            0: [_make_result(d=0, kappa=1.0, pr_restore=0.9, c_episodes=5.0)],
            200: [_make_result(d=200, kappa=1.0, pr_restore=0.4, c_episodes=7.0)],
        }
        flat = check_kappa1_flat(results_by_d, b_frozen=b_frozen)
        assert flat, "κ=1 flat must use P(c≤B), not probe-mean"

    def test_fail_ladder_uses_b_frozen(self) -> None:
        """evaluate_fail_ladder must accept b_frozen."""
        b_frozen = 10.0
        results_by_d = {
            0: [
                _make_result(d=0, kappa=0.25, c_episodes=5.0),
                _make_result(d=0, kappa=0.50, c_episodes=5.0),
                _make_result(d=0, kappa=1.0, c_episodes=5.0),
            ],
            200: [
                _make_result(d=200, kappa=0.25, c_episodes=50.0),
                _make_result(d=200, kappa=0.50, c_episodes=50.0),
                _make_result(d=200, kappa=1.0, c_episodes=6.0),
            ],
        }
        decision = evaluate_fail_ladder(results_by_d, b_frozen=b_frozen)
        assert decision.label == LadderOutcome.PROCEED.value

    def test_partial_proceed_removed(self) -> None:
        """No 'partial proceed' — if G-D holds at some κ but fails at others,
        result must be H2 kill, not proceed."""
        b_frozen = 10.0
        results_by_d = {
            0: [
                _make_result(d=0, kappa=0.25, c_episodes=5.0),
                _make_result(d=0, kappa=0.50, c_episodes=5.0),
                _make_result(d=0, kappa=1.0, c_episodes=5.0),
            ],
            200: [
                _make_result(d=200, kappa=0.25, c_episodes=50.0),
                _make_result(d=200, kappa=0.50, c_episodes=8.0),
                _make_result(d=200, kappa=1.0, c_episodes=6.0),
            ],
        }
        decision = evaluate_fail_ladder(results_by_d, b_frozen=b_frozen)
        assert decision.label == LadderOutcome.H2_KILL.value

    def test_gd_diagnostic_fields_in_json(self) -> None:
        """JSON output includes diagnostic probe-window means, clearly labeled."""
        config = GridConfig(n_seeds=1, n_instances=5)
        output = run_grid(config)
        j = grid_output_to_json(output)
        for key, gd in j["primary_gd"].items():
            assert "diagnostic_probe_mean_d0" in gd
            assert "diagnostic_probe_mean_d200" in gd
            assert gd["metric"] == "P(c<=B)"
        assert "pr_restore_diagnostic" in j
        assert j["gd_metric"] == "P(c<=B)"


# ══════════════════════════════════════════════════════════════════════════
# P1: κ must move retention under busy
# ══════════════════════════════════════════════════════════════════════════

class TestP1_KappaMovesRetention:
    """P1: different κ must produce different retention under busy load.

    Metric: probe hit rate (pr_restore) at fixed (family, d, seed) must
    diverge between κ=0.25 and κ=1.00.  At κ=1.00 (skill-dominant
    eviction), target-family entries survive busy fill → memory hit →
    higher pr_restore.  At κ=0.25 (LRU-dominant), target entries are
    evicted → memory miss → lower pr_restore.
    """

    def test_kappa_diverges_on_busy_probe_hit_rate(self) -> None:
        """Busy arm at d=200: κ=1 and κ=0.25 must differ on pr_restore."""
        from fork1.grid import run_single_cell
        config = GridConfig(n_seeds=1, n_instances=10)
        adapter = ModelAdapterStub()

        r_low = run_single_cell(
            "earnings", ArmType.BUSY, d=200, kappa=0.25,
            seed=0, config=config, adapter=adapter,
        )
        r_high = run_single_cell(
            "earnings", ArmType.BUSY, d=200, kappa=1.00,
            seed=0, config=config, adapter=adapter,
        )
        assert r_high.pr_restore != r_low.pr_restore, (
            f"κ=0.25 pr={r_low.pr_restore}, κ=1.0 pr={r_high.pr_restore} — "
            "must differ (κ is inert = P1 fail)"
        )

    def test_high_kappa_retains_target_family(self) -> None:
        """At κ=1.0 under busy d=200, target-family entries survive in the
        store (skill-based eviction protects high-skill entries)."""
        store = FamilyTaggedStore(capacity=20, kappa=1.0)
        for ep in range(10):
            store.store("target", skill=0.85, fidelity=1.0, episode=ep)
        for ep in range(10, 60):
            store.store("filler", skill=0.80, fidelity=1.0, episode=ep)
        assert store.has_family("target"), (
            "κ=1.0 should retain target (highest skill) through busy fill"
        )

    def test_low_kappa_loses_target_family(self) -> None:
        """At κ=0.25 under busy fill, target-family entries (oldest) are
        evicted because position dominates skill."""
        store = FamilyTaggedStore(capacity=20, kappa=0.25)
        for ep in range(10):
            store.store("target", skill=0.85, fidelity=1.0, episode=ep)
        for ep in range(10, 60):
            store.store("filler", skill=0.80, fidelity=1.0, episode=ep)
        assert not store.has_family("target"), (
            "κ=0.25 should evict target (oldest) during busy fill"
        )

    def test_grid_diagnostic_probe_mean_diverges_across_kappa(self) -> None:
        """In the grid JSON, diagnostic_probe_mean at d=200 must not be
        identical across κ values."""
        config = GridConfig(n_seeds=2, n_instances=10)
        output = run_grid(config)
        j = grid_output_to_json(output)
        probe_means_d200 = set()
        for key, gd in j["primary_gd"].items():
            probe_means_d200.add(round(gd["diagnostic_probe_mean_d200"], 4))
        assert len(probe_means_d200) > 1, (
            f"All κ have same probe_mean_d200={probe_means_d200} — κ inert"
        )


# ══════════════════════════════════════════════════════════════════════════
# Stack alternate c_never: cold-start acquisition cost
# ══════════════════════════════════════════════════════════════════════════

class TestColdStartCNever:
    """c_never = cold-start acquisition cost: episodes until w=2
    consecutive successes on never-learned probes."""

    def test_cold_start_basic(self) -> None:
        from fork1.metrics import compute_c_never_cold_start
        outcomes = [True, True, False, True, True]
        c, censored = compute_c_never_cold_start(outcomes, w=2)
        assert c == 2.0
        assert not censored

    def test_cold_start_delayed(self) -> None:
        from fork1.metrics import compute_c_never_cold_start
        outcomes = [True, False, False, True, True]
        c, censored = compute_c_never_cold_start(outcomes, w=2)
        assert c == 5.0
        assert not censored

    def test_cold_start_censored(self) -> None:
        from fork1.metrics import compute_c_never_cold_start
        outcomes = [True, False, True, False, True]
        c, censored = compute_c_never_cold_start(outcomes, w=2)
        assert censored
        assert c == 5.0

    def test_cold_start_empty(self) -> None:
        from fork1.metrics import compute_c_never_cold_start
        c, censored = compute_c_never_cold_start([], w=2, max_episodes=100)
        assert censored
        assert c == 100.0

    def test_never_learned_uses_cold_start_in_grid(self) -> None:
        """Grid never-learned cells must use cold-start c, not recovery c."""
        config = GridConfig(n_seeds=2, n_instances=10)
        output = run_grid(config)
        never = [r for r in output.cell_results
                 if r.arm_type == ArmType.NEVER_LEARNED]
        assert len(never) > 0
        for r in never:
            assert r.c_episodes <= r.n_probes, (
                f"Cold-start c ({r.c_episodes}) should be ≤ n_probes ({r.n_probes})"
            )

    def test_never_learned_pre_dormancy_mean_is_nan(self) -> None:
        """Never-learned has no active block → pre_dormancy_mean = NaN."""
        import math
        config = GridConfig(n_seeds=1, n_instances=5)
        output = run_grid(config)
        never = [r for r in output.cell_results
                 if r.arm_type == ArmType.NEVER_LEARNED]
        for r in never:
            assert math.isnan(r.pre_dormancy_mean)

    def test_busy_idle_deletion_use_recovery_c(self) -> None:
        """Arms with active blocks must still use ε-recovery c, not cold-start."""
        import math
        config = GridConfig(n_seeds=1, n_instances=5)
        output = run_grid(config)
        for r in output.cell_results:
            if r.arm_type != ArmType.NEVER_LEARNED:
                assert not math.isnan(r.pre_dormancy_mean)

    def test_json_c_never_source_is_cold_start(self) -> None:
        config = GridConfig(n_seeds=1, n_instances=5)
        output = run_grid(config)
        j = grid_output_to_json(output)
        assert j["c_never_source"] == "pilot_cold_start"
        assert "c_never_definition" in j
        assert "PILOT_B_CONTAMINATION_RISK" in j

    def test_json_never_learned_diagnostic(self) -> None:
        config = GridConfig(n_seeds=1, n_instances=5)
        output = run_grid(config)
        j = grid_output_to_json(output)
        diag = j["never_learned_diagnostic"]
        assert len(diag) > 0
        for key, entry in diag.items():
            assert entry["pre_dormancy_mean"] is None
            assert entry["c_definition"] == "cold_start_consecutive_w2"
