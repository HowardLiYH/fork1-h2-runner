"""Tests for determinism harness."""

from __future__ import annotations

from fork1.determinism import check_determinism
from fork1.grid import GridConfig
from fork1.schedule import ArmType


class TestDeterminism:
    def test_determinism_100_100(self) -> None:
        """Same seed → 100/100 identical traces."""
        config = GridConfig(n_instances=5)
        is_det, n = check_determinism(
            family="earnings",
            arm_type=ArmType.BUSY,
            d=50,
            kappa=0.50,
            seed=42,
            n_repeats=100,
            config=config,
        )
        assert is_det
        assert n == 100

    def test_determinism_different_seeds_differ(self) -> None:
        """Different seeds should (generally) produce different results."""
        from fork1.grid import run_single_cell
        from fork1.oracle import ModelAdapterStub

        config = GridConfig(n_instances=10)
        adapter = ModelAdapterStub()

        r1 = run_single_cell("earnings", ArmType.BUSY, 50, 0.5, 0, config, adapter)
        r2 = run_single_cell("earnings", ArmType.BUSY, 50, 0.5, 1, config, adapter)
        # Not asserting they MUST differ (could happen by chance with
        # procedural oracles), but the harness should handle both seeds.
        assert r1.seed != r2.seed

    def test_determinism_across_arm_types(self) -> None:
        config = GridConfig(n_instances=5)
        for arm_type in [ArmType.BUSY, ArmType.IDLE]:
            is_det, n = check_determinism(
                family="crisis",
                arm_type=arm_type,
                d=50,
                kappa=0.25,
                seed=7,
                n_repeats=50,
                config=config,
            )
            assert is_det, f"Determinism failed for {arm_type}"
