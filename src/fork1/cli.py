"""CLI entry point for Fork 1 / H2 experiment runner.

Usage:
    fork1-run --dry-run              # Quick dry-run with reduced grid
    fork1-run --full                 # Full frozen grid (8 seeds, 70 instances)
    fork1-run --determinism-check    # Run determinism harness
    fork1-run --output results.json  # Write results to file
"""

from __future__ import annotations

import argparse
import json
import sys

from fork1.determinism import check_determinism
from fork1.grid import GridConfig, GridOutput, grid_output_to_json, run_grid


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fork1-run",
        description="Fork 1 / H2 experiment runner: family-tagged store + κ + LRU",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Quick dry-run with reduced grid (2 seeds, 10 instances)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        default=False,
        help="Full frozen grid (8 seeds, 70 instances/family)",
    )
    parser.add_argument(
        "--determinism-check",
        action="store_true",
        default=False,
        help="Run determinism harness (100 repeats, same seed)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Write JSON results to file (default: stdout)",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=None,
        help="Override number of seeds",
    )
    parser.add_argument(
        "--instances",
        type=int,
        default=None,
        help="Override number of instances per family",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.determinism_check:
        _run_determinism_check(args)
        return

    config = _build_config(args)
    output = run_grid(config)
    result_json = grid_output_to_json(output)
    _emit_json(result_json, args.output)


def _build_config(args: argparse.Namespace) -> GridConfig:
    if args.dry_run:
        n_seeds = args.seeds if args.seeds is not None else 2
        n_instances = args.instances if args.instances is not None else 10
    elif args.full:
        n_seeds = args.seeds if args.seeds is not None else 8
        n_instances = args.instances if args.instances is not None else 70
    else:
        n_seeds = args.seeds if args.seeds is not None else 2
        n_instances = args.instances if args.instances is not None else 10

    return GridConfig(
        n_seeds=n_seeds,
        n_instances=n_instances,
    )


def _run_determinism_check(args: argparse.Namespace) -> None:
    n_repeats = 100
    config = GridConfig(n_instances=10)

    from fork1.schedule import ArmType
    cells_to_check = [
        ("earnings", ArmType.BUSY, 50, 0.50, 0),
        ("crisis", ArmType.IDLE, 200, 0.25, 3),
        ("filings", ArmType.BUSY, 0, 1.00, 7),
    ]

    results = {}
    all_pass = True
    for family, arm_type, d, kappa, seed in cells_to_check:
        is_det, n = check_determinism(
            family=family,
            arm_type=arm_type,
            d=d,
            kappa=kappa,
            seed=seed,
            n_repeats=n_repeats,
            config=config,
        )
        key = f"{family}_{arm_type.value}_d{d}_k{kappa}_s{seed}"
        results[key] = {"deterministic": is_det, "repeats": n}
        if not is_det:
            all_pass = False

    output = {
        "determinism_check": results,
        "all_pass": all_pass,
        "target": f"{n_repeats}/{n_repeats}",
    }
    _emit_json(output, args.output)

    if not all_pass:
        sys.exit(1)


def _emit_json(data: dict, output_path: str | None) -> None:
    json_str = json.dumps(data, indent=2)
    if output_path:
        with open(output_path, "w") as f:
            f.write(json_str)
            f.write("\n")
        print(f"Results written to {output_path}", file=sys.stderr)
    else:
        print(json_str)


if __name__ == "__main__":
    main()
