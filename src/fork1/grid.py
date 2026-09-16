"""Frozen grid runner for the H2 experiment.

Frozen grid: d ∈ {0, 50, 200}; κ ∈ {0.25, 0.50, 1.00}; ≥70 instances/family;
8 seeds; 3 families (earnings, crisis, filings).

Seed patterns inherited from NicheMem: env 1000+seed, policy 50000+seed*97.
"""

from __future__ import annotations

import dataclasses
import json
from typing import Any, Optional

from fork1.fail_ladder import evaluate_fail_ladder
from fork1.memory import FamilyTaggedStore
from fork1.metrics import (
    CellResult,
    SMetric,
    censor_rate,
    check_gd,
    check_kappa1_flat,
    compute_c_episodes,
    compute_p_recovery_within_b,
    compute_pr_restore,
    compute_pre_dormancy_mean,
    compute_s,
    estimate_c_never,
    evaluate_cell,
    freeze_b,
)
from fork1.oracle import ModelAdapterStub
from fork1.schedule import FAMILIES, ArmType, Schedule, Task


DORMANCY_VALUES = (0, 50, 200)
KAPPA_VALUES = (0.25, 0.50, 1.00)
N_INSTANCES = 70
N_SEEDS = 8
MEMORY_CAPACITY = 100
PILOT_INSTANCES = 30
PILOT_SEEDS = 4
WITHHELD_ERA_SEEDS = (6, 7)


@dataclasses.dataclass
class GridConfig:
    """Frozen grid configuration.

    Pilot sizes (30 instances × 4 seeds) are locked by Experiment Design.
    """
    dormancy_values: tuple[int, ...] = DORMANCY_VALUES
    kappa_values: tuple[float, ...] = KAPPA_VALUES
    families: tuple[str, ...] = FAMILIES
    n_instances: int = N_INSTANCES
    n_seeds: int = N_SEEDS
    memory_capacity: int = MEMORY_CAPACITY
    active_length: int = 20
    probe_window: int = 2
    min_dormancy_before_probe: int = 10
    pilot_instances: int = PILOT_INSTANCES
    pilot_seeds: int = PILOT_SEEDS
    withheld_era_seeds: tuple[int, ...] = WITHHELD_ERA_SEEDS


@dataclasses.dataclass
class GridOutput:
    """Full output of a grid run."""
    config: GridConfig
    cell_results: list[CellResult]
    gd_results: dict[str, Any]
    kappa1_flat: bool
    c_never: float
    b_frozen: float
    s_metrics: list[SMetric]
    censor_rates: dict[str, float]
    fail_ladder: Any


def _env_seed(base_seed: int) -> int:
    """NicheMem pattern: env seed = 1000 + seed."""
    return 1000 + base_seed


def _policy_seed(base_seed: int) -> int:
    """NicheMem pattern: policy seed = 50000 + seed * 97."""
    return 50000 + base_seed * 97


def run_single_cell(
    family: str,
    arm_type: ArmType,
    d: int,
    kappa: float,
    seed: int,
    config: GridConfig,
    adapter: ModelAdapterStub,
) -> CellResult:
    """Run a single experimental cell and return its result.

    Memory is wired into predictions causally (P0-1):
    - Probe tasks: retrieve memory for the target family; pass the entry
      (hit) or None (miss) to adapter.predict().
    - Training / busy-fill tasks: predict with memory=None.

    Deletion arm (P0-3): family entries are cleared from the store before
    each probe window, so probes always get a memory miss.
    """
    env_s = _env_seed(seed)

    schedule = Schedule(
        families=config.families,
        active_length=config.active_length,
        probe_window=config.probe_window,
        min_dormancy_before_probe=config.min_dormancy_before_probe,
    )

    tasks = schedule.build_stream(
        arm_type=arm_type,
        target_family=family,
        dormancy_d=d,
        n_instances=config.n_instances,
    )

    store = FamilyTaggedStore(capacity=config.memory_capacity, kappa=kappa)

    pre_dormancy_outcomes: list[bool] = []
    post_dormancy_outcomes: list[bool] = []
    probe_outcomes: list[bool] = []

    for task in tasks:
        # P0-3: deletion arm clears family entries before probes
        if arm_type == ArmType.DELETION and task.is_probe and task.family == family:
            store.delete_family(family)

        # P0-1: look up memory for probe tasks on the target family
        memory = None
        if task.family == family and task.is_probe:
            memory = store.access(family, task.episode)

        outcome = adapter.predict(task, env_s, memory=memory)

        if task.family == family:
            if task.dormancy == 0 and not task.is_probe:
                pre_dormancy_outcomes.append(outcome)
                if outcome:
                    store.store(
                        family=task.family,
                        skill=adapter.success(task, env_s),
                        fidelity=1.0,
                        episode=task.episode,
                    )
            elif task.is_probe:
                probe_outcomes.append(outcome)
                post_dormancy_outcomes.append(outcome)
            elif task.dormancy > 0 and not task.is_probe:
                post_dormancy_outcomes.append(outcome)
        else:
            if outcome:
                store.store(
                    family=task.family,
                    skill=adapter.success(task, env_s),
                    fidelity=1.0,
                    episode=task.episode,
                )

    max_episodes = len(tasks)
    return evaluate_cell(
        probe_outcomes=probe_outcomes,
        pre_dormancy_outcomes=pre_dormancy_outcomes,
        post_dormancy_outcomes=post_dormancy_outcomes,
        family=family,
        arm_type=arm_type,
        d=d,
        kappa=kappa,
        seed=seed,
        max_episodes=max_episodes,
    )


def run_pilot_c_never(config: GridConfig, adapter: ModelAdapterStub) -> float:
    """Estimate c_never ONCE from a frozen pilot ceiling.

    Runs never-learned arms on a reduced grid (pilot_seeds × pilot_instances)
    across all families and κ values. The result is locked — no per-cell or
    post-hoc re-fit of B.
    """
    pilot_config = GridConfig(
        dormancy_values=config.dormancy_values,
        kappa_values=config.kappa_values,
        families=config.families,
        n_instances=config.pilot_instances,
        n_seeds=config.pilot_seeds,
        memory_capacity=config.memory_capacity,
        active_length=config.active_length,
        probe_window=config.probe_window,
        min_dormancy_before_probe=config.min_dormancy_before_probe,
        pilot_instances=config.pilot_instances,
        pilot_seeds=config.pilot_seeds,
    )
    pilot_results: list[CellResult] = []
    for seed in range(pilot_config.n_seeds):
        for family in pilot_config.families:
            for kappa in pilot_config.kappa_values:
                result = run_single_cell(
                    family=family,
                    arm_type=ArmType.NEVER_LEARNED,
                    d=0,
                    kappa=kappa,
                    seed=seed,
                    config=pilot_config,
                    adapter=adapter,
                )
                pilot_results.append(result)

    return estimate_c_never(pilot_results)


def run_grid(config: Optional[GridConfig] = None) -> GridOutput:
    """Run the full frozen grid experiment."""
    if config is None:
        config = GridConfig()

    adapter = ModelAdapterStub()
    all_results: list[CellResult] = []

    c_never = run_pilot_c_never(config, adapter)
    b_frozen = freeze_b(c_never)

    arm_types_to_run = [ArmType.BUSY, ArmType.IDLE, ArmType.NEVER_LEARNED, ArmType.DELETION]

    for seed in range(config.n_seeds):
        for family in config.families:
            for kappa in config.kappa_values:
                for d in config.dormancy_values:
                    for arm_type in arm_types_to_run:
                        if arm_type == ArmType.NEVER_LEARNED and d != 0:
                            continue

                        result = run_single_cell(
                            family=family,
                            arm_type=arm_type,
                            d=d,
                            kappa=kappa,
                            seed=seed,
                            config=config,
                            adapter=adapter,
                        )
                        all_results.append(result)

    never_learned_results = [
        r for r in all_results if r.arm_type == ArmType.NEVER_LEARNED
    ]

    results_by_d: dict[int, list[CellResult]] = {}
    for r in all_results:
        results_by_d.setdefault(r.d, []).append(r)

    gd_results: dict[str, Any] = {}
    for kappa in config.kappa_values:
        kappa_cells: dict[int, list[CellResult]] = {}
        for d, cells in results_by_d.items():
            matching = [
                c for c in cells
                if abs(c.kappa - kappa) < 1e-9 and c.arm_type == ArmType.BUSY
            ]
            if matching:
                kappa_cells[d] = matching
        if kappa_cells:
            gd = check_gd(kappa_cells, kappa, b_frozen)
            gd_results[f"kappa_{kappa}"] = {
                "kappa": gd.kappa,
                "p_c_leq_b_d0": gd.pr_d0,
                "p_c_leq_b_d200": gd.pr_d200,
                "gap": gd.gap,
                "gd_holds": gd.gd_holds,
                "metric": "P(c<=B)",
                "diagnostic_probe_mean_d0": round(gd.probe_mean_d0, 4),
                "diagnostic_probe_mean_d200": round(gd.probe_mean_d200, 4),
            }

    kappa1_flat = check_kappa1_flat(results_by_d, b_frozen)

    s_metrics: list[SMetric] = []
    withheld_seeds = set(config.withheld_era_seeds)
    for family in config.families:
        for kappa in (0.25, 0.50):
            for d in config.dormancy_values:
                if d == 0:
                    continue
                store_cells = [
                    r for r in all_results
                    if r.family == family
                    and r.arm_type == ArmType.BUSY
                    and r.d == d
                    and abs(r.kappa - kappa) < 1e-9
                ]
                never_cells = [
                    r for r in never_learned_results
                    if r.family == family
                    and abs(r.kappa - kappa) < 1e-9
                ]
                if store_cells and never_cells:
                    s_all = compute_s(
                        store_cells, never_cells, b_frozen,
                        family, d, kappa,
                        in_withheld_era=False,
                    )
                    s_metrics.append(s_all)

                    store_wh = [r for r in store_cells if r.seed in withheld_seeds]
                    never_wh = [r for r in never_cells if r.seed in withheld_seeds]
                    if store_wh and never_wh:
                        s_wh = compute_s(
                            store_wh, never_wh, b_frozen,
                            family, d, kappa,
                            in_withheld_era=True,
                        )
                        s_metrics.append(s_wh)

    cell_censor_rates: dict[str, float] = {}
    for family in config.families:
        for d in config.dormancy_values:
            for kappa in config.kappa_values:
                key = f"{family}_d{d}_k{kappa}"
                matching = [
                    r for r in all_results
                    if r.family == family and r.d == d and abs(r.kappa - kappa) < 1e-9
                ]
                cell_censor_rates[key] = censor_rate(matching)

    fail_decision = evaluate_fail_ladder(
        results_by_d=results_by_d,
        b_frozen=b_frozen,
        kappa_values=(0.25, 0.50),
        s_metrics=s_metrics,
    )

    return GridOutput(
        config=config,
        cell_results=all_results,
        gd_results=gd_results,
        kappa1_flat=kappa1_flat,
        c_never=c_never,
        b_frozen=b_frozen,
        s_metrics=s_metrics,
        censor_rates=cell_censor_rates,
        fail_ladder=fail_decision,
    )


def grid_output_to_json(output: GridOutput) -> dict[str, Any]:
    """Convert GridOutput to a JSON-serializable dict."""
    pr_by_cell: dict[str, float] = {}
    c_by_cell: dict[str, float] = {}
    for r in output.cell_results:
        key = f"{r.family}_{r.arm_type.value}_d{r.d}_k{r.kappa}_s{r.seed}"
        pr_by_cell[key] = round(r.pr_restore, 4)
        c_by_cell[key] = round(r.c_episodes, 2)

    s_list = []
    for s in output.s_metrics:
        s_list.append({
            "family": s.family,
            "d": s.d,
            "kappa": s.kappa,
            "pr_store": round(s.pr_store, 4),
            "pr_never": round(s.pr_never, 4),
            "s_value": round(s.s_value, 4),
            "b_frozen": round(s.b_frozen, 2),
            "in_withheld_era": s.in_withheld_era,
        })

    return {
        "status_note": (
            "SMOKE / HARNESS ONLY — P0 in progress. "
            "Not Stack-countable. Do not cite as H2 evidence."
        ),
        "grid": {
            "dormancy_values": list(output.config.dormancy_values),
            "kappa_values": list(output.config.kappa_values),
            "families": list(output.config.families),
            "n_instances": output.config.n_instances,
            "n_seeds": output.config.n_seeds,
        },
        "pilot": {
            "instances": output.config.pilot_instances,
            "seeds": output.config.pilot_seeds,
        },
        "withheld_era_seeds": list(output.config.withheld_era_seeds),
        "primary_gd": output.gd_results,
        "gd_metric": "P(c<=B)",
        "kappa1_flat": output.kappa1_flat,
        "c_never": round(output.c_never, 2),
        "c_never_source": "pilot",
        "b_frozen": round(output.b_frozen, 2),
        "s_metrics": s_list,
        "censor_rates": {k: round(v, 4) for k, v in output.censor_rates.items()},
        "fail_ladder": {
            "step": output.fail_ladder.step,
            "label": output.fail_ladder.label,
            "detail": output.fail_ladder.detail,
        },
        "pr_restore_diagnostic": pr_by_cell,
        "c_episodes": c_by_cell,
    }
