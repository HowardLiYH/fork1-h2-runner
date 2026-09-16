"""Tests for CLI dry-run and JSON output."""

from __future__ import annotations

import json
import os
import tempfile

import pytest

from fork1.cli import main


class TestCLI:
    def test_dry_run_emits_json(self, capsys: pytest.CaptureFixture[str]) -> None:
        main(["--dry-run"])
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert "primary_gd" in data
        assert "kappa1_flat" in data
        assert "c_never" in data
        assert "b_frozen" in data
        assert "s_metrics" in data
        assert "censor_rates" in data
        assert "fail_ladder" in data
        assert "pr_restore_diagnostic" in data
        assert "c_episodes" in data
        assert data["c_never_source"] == "pilot_cold_start"
        assert data["gd_metric"] == "P(c<=B)"
        assert "status_note" in data
        assert "PILOT_B_CONTAMINATION_RISK" in data

    def test_dry_run_output_file(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            tmppath = f.name
        try:
            main(["--dry-run", "--output", tmppath])
            with open(tmppath) as f:
                data = json.load(f)
            assert "fail_ladder" in data
            assert data["fail_ladder"]["step"] in (1, 2, 3, 4)
            assert data["fail_ladder"]["label"] in (
                "proceed", "h2_kill", "harmfulness", "harness_bug"
            )
        finally:
            os.unlink(tmppath)

    def test_dry_run_has_grid_config(self, capsys: pytest.CaptureFixture[str]) -> None:
        main(["--dry-run", "--seeds", "1", "--instances", "5"])
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["grid"]["n_seeds"] == 1
        assert data["grid"]["n_instances"] == 5
        assert data["grid"]["families"] == ["earnings", "crisis", "filings"]
        assert data["grid"]["dormancy_values"] == [0, 50, 200]
        assert data["grid"]["kappa_values"] == [0.25, 0.50, 1.00]
        assert "pilot" in data
        assert "withheld_era_seeds" in data

    def test_dry_run_b_frozen_is_half_c_never(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        main(["--dry-run"])
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert abs(data["b_frozen"] - 0.5 * data["c_never"]) < 0.01

    def test_dry_run_s_metrics_have_frozen_b(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        main(["--dry-run"])
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        b_frozen = data["b_frozen"]
        for s in data["s_metrics"]:
            assert abs(s["b_frozen"] - b_frozen) < 0.01

    def test_dry_run_gd_metric_is_p_c_leq_b(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """P0-4: primary_gd entries report metric=P(c<=B)."""
        main(["--dry-run"])
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["gd_metric"] == "P(c<=B)"
        for key, gd in data["primary_gd"].items():
            assert gd["metric"] == "P(c<=B)"
            assert "p_c_leq_b_d0" in gd
            assert "p_c_leq_b_d200" in gd
            assert "diagnostic_probe_mean_d0" in gd
            assert "diagnostic_probe_mean_d200" in gd

    def test_determinism_check(self, capsys: pytest.CaptureFixture[str]) -> None:
        main(["--determinism-check"])
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["all_pass"]
        assert "determinism_check" in data
