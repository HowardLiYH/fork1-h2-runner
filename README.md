# Fork 1 / H2 Experiment Runner

Experiment harness for testing the H2 hypothesis under family-tagged store + κ + LRU memory. Procedural oracles only — no real model results yet.

## Falsifiable Claim (Primary)

> Under family-tagged store + κ + LRU, busy arms fail Stack G-D:
> Pr(d=200) ≤ Pr(d=0) − 0.25 at κ ∈ {0.25, 0.50} with κ=1 flat → H2 dies.

## Frozen Grid

| Parameter | Values |
|-----------|--------|
| Families | earnings, crisis, filings (≥3) |
| Arms | busy, idle, never-learned, deletion |
| d (dormancy) | 0, 50, 200 |
| κ (retention) | 0.25, 0.50, 1.00 |
| Instances/family | ≥70 |
| Seeds | 8 |
| Determinism | 100/100 identical traces |

**Busy arms never reuse a dormant family** — this is a hard invariant enforced by the schedule builder and tested.

## Frozen Parameters

- **θ = 0.80** (train-only)
- **w = 2** (probe window / Stack frozen)
- **B = 0.5 × c_never** — estimated ONCE from never-learned reference under a frozen pilot ceiling (30 instances × 4 seeds), then locked for all P(c ≤ B), reacquisition budget, and S computations. No per-cell re-fit.
- **Pilot sizes**: 30 instances × 4 seeds (locked by Experiment Design)
- ε = 0.05 absolute (secondary c(d) recovery criterion)
- **Withheld-era partition**: seeds 6, 7 of 8 (frozen before first S measurement; open lab knob for PIT cut dates on real data)

## Metrics

### Primary: G-D

Restoration probability `Pr(d)` on busy at κ∈{0.25, 0.50}. Requires κ=1 flat as control.

**Kill condition:** Pr(d=200) ≤ Pr(d=0) − 0.25 AND κ=1 flat fails → H2 dies.

### Secondary: c(d) (KM-style)

Episodes to recover within ε=0.05 absolute of pre-dormancy family mean success (≥70/family eval set). Censored runs count as max. Per-cell censor rate reported. High censor → inconclusive (not a pass). Target: c(200)/c(0) ≥ 2.

### Rung 3: S (harmfulness branch only)

```
S = P(c ≤ B | store) − P(c ≤ B | never-learned)
```

Where `c` is recovery cost in episodes (from c_episodes). P(c ≤ B) is the fraction of runs where recovery cost ≤ frozen B. Same sign as (c_never − c_store). Withheld-era / regime-shift partition fixed BEFORE first S measurement. Rung-3 fires ONLY when H2 is flat/kill AND S<0 under drift in the withheld-era partition.

## Fail Ladder

1. G-D holds + κ=1 flat → proceed
2. G-D fails → H2 kill, bounce Stack
3. H2 flat/kill AND S<0 under drift (withheld-era only) → harmfulness
4. rise at κ=1 → harness bug, stop

## Inheritance Statement

This harness inherits **schedule patterns only** from:

- **NicheMem** ([HowardLiYH/NicheMem](https://github.com/HowardLiYH/NicheMem)): cyclic sliding-window `Schedule`, `Environment.stream()` task fields, probe conventions, `evict_lru()` with skill/fidelity tie-break, E3 dormancy sweep seed pattern (env `1000+seed`, policy `50000+seed*97`).
- **RISP** ([HowardLiYH/RISP](https://github.com/HowardLiYH/RISP)): `label_L1`/`label_L2` causal shift regimes, `RealMarket.schedule()` dormancy between regime blocks, `StitchedMarket` stitching, walk-forward / withheld-era / honest-null discipline.

**NicheMem and RISP results are NOT H2 evidence.** Their policies (`compete→pin`, `Γ̂`) and result JSONs are not copied. Only harness/schedule structure is inherited.

## Out of v1

dormancy ladder, ρ(d), keyed paging, ĉ, fleet R3, baseline bake-off, PopAgent, E8 drift as treatment, extractive foils, RecMem arms

## Open Knobs (placeholders — freeze before real data)

- PIT cut dates
- Single frozen open-weight model id
- batch-1

## Quick Start

```bash
pip install -e ".[dev]"

# Dry-run (2 seeds, 10 instances — fast)
fork1-run --dry-run

# Full frozen grid (8 seeds, 70 instances)
fork1-run --full

# Determinism check (100/100 target)
fork1-run --determinism-check

# Write results to file
fork1-run --dry-run --output results.json
```

## Running Tests

```bash
pytest tests/ -v
```

## Package Structure

```
src/fork1/
├── __init__.py          # Version
├── memory.py            # Family-tagged store + κ + LRU
├── schedule.py          # Schedule, arms, families, busy≠reuse invariant
├── oracle.py            # Procedural playbook oracles + model adapter stub
├── metrics.py           # G-D checker, c(d) secondary, S metric, frozen B
├── fail_ladder.py       # 4-step fail ladder
├── grid.py              # Frozen grid runner
├── determinism.py       # 100/100 determinism harness
└── cli.py               # CLI entry point (dry-run / full / determinism)

tests/
├── test_memory.py       # Store, eviction, κ weighting
├── test_schedule.py     # Arm types, busy≠reuse invariant
├── test_oracle.py       # Deterministic hash, playbook oracles
├── test_metrics.py      # G-D, c(d), S, fail ladder, censor rate
├── test_determinism.py  # 100/100 identical traces
└── test_cli.py          # JSON output structure, B=0.5×c_never
```
