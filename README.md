# orc_bo — Bayesian optimization for ORC working fluids

Constrained Bayesian optimization for selecting Organic Rankine Cycle (ORC) working
fluids — both **pure fluids** and **binary mixtures** — to maximize cycle thermal
efficiency subject to pressure and pinch constraints.

The optimizer searches a relaxed one-hot fluid space. Continuous suggestions are *snapped*
to a concrete fluid: to the nearest vertex (a pure fluid) or to a point on an edge between
two vertices (a binary mixture, with its mole fraction set by the projection). For each
realized fluid, an inner SCBO (TuRBO) loop optimizes the operating pressures.

---

## Architecture

The package is a small core of single-responsibility modules with two pipelines on top.
Data flows: **config → thermo/geometry → orc_model → scbo → pipelines**.

```
orc_bo/
  config.py          Frozen dataclasses for all constants (ORC, BO, mixture, two-stage,
                     thermo, paths). Loads defaults, an optional TOML file, and env overrides.
  logging_setup.py   configure_logging(); modules log via logging, never print.
  thermo.py          The ONLY module that calls CoolProp/REFPROP. Pure & mixture
                     properties, isentropic enthalpy, with a logged + counted REFPROP ->
                     mixing-rule fallback.
  geometry.py        snap_to_vertex (pure), snap_to_mixture (edge), composition snapping,
                     canonical keys.
  orc_model.py       ORCSimulator: one ORC evaluation -> (eta, sink_pinch, source_pinch),
                     using saturation-temperature pinch logic (handles mixture glide).
  constraints.py     ORC SCBO constraints + MixtureConstraintManager (composition,
                     miscibility, property ranges, weighted GWP/ODP/cost).
  scbo.py            SCBO/TuRBO trust-region constrained optimization of operating
                     conditions for a fixed fluid.
  targeting.py       Two-stage stage-1 blocks: PropNormalizer, target GP, variational GP
                     classifier (GPC), maximin space-filling.
  seeding.py         Reproducible seed derivation from PYTHONHASHSEED / JOBACK_SEED.
  cli.py             python -m orc_bo.cli onestage|twostage ...
  pipelines/
    common.py        Fluid loading, candidate realization, incremental result writing.
    onestage.py      One-stage qEI BO directly in one-hot space (pure or mixture).
    twostage.py      Two-stage property targeting -> GPC space-filling -> SCBO -> cEI.
  data/Joback_Refrigerants.csv

benchmarks/
  run_benchmark.py     Multi-seed, parallel runner (pure/mixture x onestage/twostage),
                       with an optional hyperparameter sweep.
  aggregate_results.py Per-seed best-efficiency stats (mean, std, bootstrap CI) -> CSV/LaTeX.

tests/                 pytest suite; REFPROP-only tests auto-skip when REFPROP is absent.
```

### How the modules interact

1. `config.load_config()` produces an `AppConfig` bundle.
2. A pipeline loads fluids (`pipelines.common.load_fluids`) and builds the one-hot basis.
3. BO proposes a continuous point; `geometry` snaps it to a pure fluid or mixture.
4. `pipelines.common.realize_candidate` turns the selection into a working-fluid string and
   its critical properties via `thermo`.
5. `scbo.optimize_operating_conditions_robust` runs the inner TuRBO loop, calling
   `orc_model.ORCSimulator.simulate` for each operating point.
6. Results are written incrementally; `benchmarks/aggregate_results.py` summarizes them.

---

## Installation

```bash
python -m venv venv
venv/Scripts/python -m pip install -r requirements.txt        # Windows
# source venv/bin/activate && pip install -r requirements.txt # Unix
```

* **Pure-fluid** optimization works with the CoolProp **HEOS** backend (no extra license).
* **Mixture** optimization is most accurate with **REFPROP** (a licensed product reachable
  by CoolProp). Without REFPROP, mixture critical properties fall back to analytic mixing
  rules (logged and counted), and HEOS may not support every mixture.

Select the backend with `--backend {HEOS,REFPROP}` or the `ORC_BO_BACKEND` env variable.

---

## Usage

Both pipelines share the same flags: `--mode {pure,mixture}`, `--n-init`, `--scbo-budget`,
`--outdir`, `--backend`, `--csv`, `--config`, `--log-level`. The seed comes from
`PYTHONHASHSEED`, so runs are reproducible.

### Pure fluids

Snap to one-hot vertices; each candidate is a single fluid. Runs on the HEOS backend, so no
REFPROP is required.

```bash
# One-stage, pure fluids
PYTHONHASHSEED=0 python -m orc_bo.cli onestage \
    --mode pure --backend HEOS \
    --n-init 3 --scbo-budget 10 \
    --outdir runs/pure_onestage
```

Outputs (under `runs/pure_onestage/seed_000/`):

| file | contents |
|------|----------|
| `scbo_results.csv` | one row per evaluation: phase, fluid, Tc, Pc, eta, pressures |
| `sequence.csv`     | order of evaluation |
| `summary.txt`      | list of evaluated fluids |

### Mixture fluids

Snap to edges between vertices; each candidate is a binary mixture with a mole fraction.
Use the REFPROP backend for accurate mixture thermodynamics.

```bash
# One-stage, binary mixtures (REFPROP backend)
PYTHONHASHSEED=0 python -m orc_bo.cli onestage \
    --mode mixture --backend REFPROP \
    --n-init 3 --scbo-budget 10 \
    --outdir runs/mixture_onestage

# Two-stage property targeting (mixtures)
PYTHONHASHSEED=0 python -m orc_bo.cli twostage \
    --mode mixture --backend REFPROP \
    --n-init 5 --scbo-budget 3 \
    --outdir runs/mixture_twostage
```

The two-stage pipeline first searches `(Tc, Pc)` property space for mixtures near desirable
targets (with a GP classifier proposing space-filling targets), then runs SCBO on the
satisfied targets, followed by a bounded cEI exploitation loop.

### Benchmarks

```bash
# 10 seeds of pure one-stage, 4 workers, then aggregate
python -m benchmarks.run_benchmark \
    --stages onestage --modes pure --backend HEOS \
    --seeds 10 --n-init 3 --scbo-budget 10 \
    --outdir bench/pure --workers 4

# Aggregate any results tree into a summary table (CSV + LaTeX)
python benchmarks/aggregate_results.py bench/pure --latex
```

A hyperparameter sweep adds extra config directories:

```bash
python -m benchmarks.run_benchmark --stages onestage --modes mixture \
    --seeds 20 --sweep-n-init 3,5 --sweep-budget 10,20 --outdir bench/sweep
```

### Configuration via TOML

Any constant can be overridden in a TOML file passed with `--config`:

```toml
[orc]
t_in_source_c = 150.0
turbine_eff = 0.65

[twostage]
required_valid_init = 5
radius_norm = 0.05
```

---

## Testing

```bash
python -m pytest tests/ -v
```

The geometry, thermo (mixing rules), ORC-model (golden values), constraints, and targeting
tests run on any machine. Tests marked `@pytest.mark.refprop` exercise the REFPROP mixture
path and are skipped automatically when REFPROP is unavailable — run them on a
REFPROP-licensed machine to validate the mixture pipeline end-to-end.

The ORC golden values in `tests/golden/` were captured from the simulation via the HEOS
backend; `test_orc_model.py` asserts the refactored code reproduces them.

---

## Notes and scope

* The ORC model is a saturated subcritical cycle at a single operating point (no superheat
  or recuperator), with source/sink conditions and efficiencies in `config.ORCConfig`.
* Mixture flammability/azeotropy are not modeled; `MixtureConstraintManager` offers
  composition, miscibility, property-range, and weighted GWP/ODP/cost screening for
  multi-criteria extensions (e.g. efficiency vs GWP).
* Visualization is intentionally out of scope for this package; the benchmark aggregator
  emits tables only.
