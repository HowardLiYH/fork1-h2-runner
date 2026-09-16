"""Determinism harness: same seed → 100/100 identical traces.

Seed pattern inherited from NicheMem: env 1000+seed, policy 50000+seed*97.
"""

from __future__ import annotations

from fork1.grid import GridConfig, run_single_cell
from fork1.oracle import ModelAdapterStub
from fork1.schedule import ArmType


def check_determinism(
    family: str = "earnings",
    arm_type: ArmType = ArmType.BUSY,
    d: int = 50,
    kappa: float = 0.50,
    seed: int = 0,
    n_repeats: int = 100,
    config: GridConfig | None = None,
) -> tuple[bool, int]:
    """Run the same cell n_repeats times and verify identical results.

    Returns (all_identical, n_repeats).
    """
    if config is None:
        config = GridConfig(n_instances=10)

    adapter = ModelAdapterStub()

    reference = run_single_cell(
        family=family,
        arm_type=arm_type,
        d=d,
        kappa=kappa,
        seed=seed,
        config=config,
        adapter=adapter,
    )

    for i in range(1, n_repeats):
        result = run_single_cell(
            family=family,
            arm_type=arm_type,
            d=d,
            kappa=kappa,
            seed=seed,
            config=config,
            adapter=adapter,
        )
        if (
            result.pr_restore != reference.pr_restore
            or result.c_episodes != reference.c_episodes
            or result.censored != reference.censored
            or result.n_probes != reference.n_probes
            or result.n_successes != reference.n_successes
            or result.pre_dormancy_mean != reference.pre_dormancy_mean
        ):
            return False, i

    return True, n_repeats
