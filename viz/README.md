# Visualisation suite

Static figures for the ORC-BO benchmark study. Every script reads the per-seed
`scbo_results.csv` files under `bench/full3/<stage>_<mode>/seed_*/` and writes PNGs into
**one** folder: [`figures/`](figures/).

## Run

```bash
python viz/make_all.py                 # render everything into viz/figures/
python viz/make_all.py --figdir out/   # or elsewhere
```

The results tree defaults to `bench/full3`; point elsewhere with the `ORC_BO_BENCH`
environment variable. Each suite is also runnable on its own: `python viz/convergence.py`, etc.

## Figures

| File | What it shows |
|------|---------------|
| `convergence.png` | Best valid η found so far vs SCBO-evaluation count; median + 25–75th-pct band across seeds. Panels: pure \| mixture; colour = stage. |
| `final_performance.png` | Mean best η per configuration with 95 % bootstrap CI (same methodology as the LaTeX table). |
| `seed_distribution.png` | Per-seed best η (box = median/IQR, points = seeds). Reveals spread and outliers the mean hides. |
| `property_space.png` | Every evaluated fluid in (Tc, Pc) space coloured by η on the blue ramp; `×` = infeasible. 2×2 by config. |
| `feasibility_rate.png` | Share of SCBO evaluations that were operable (η > 0), per configuration. |
| `coverage_ablation.png` | Two-stage best η vs pre-SCBO coverage (`n_property_targets` 8/20/40) with the one-stage baseline. **Standalone** (`python viz/coverage_ablation.py`); needs the sweep trees `bench/cov08`, `bench/full3`, `bench/cov40`. |

## Design

Colour = **stage** (one-stage = blue, two-stage = aqua), validated colourblind-safe; mode is
carried by panel/x-group, never colour, so the stage comparison stays consistent across every
figure. Efficiency magnitude uses a single-hue blue sequential ramp. Palette and matplotlib
defaults live in [`style.py`](style.py).

## Findings (bench/full3 — budget-matched, no domain prior, 20 seeds/config)

`full3` is the fair comparison: budgets matched at 25 SCBO evaluations each (5 init + 20
loop), the operability Tc band **off**, and early stopping **off**.

| Configuration | Mean best η | 95% CI | Max | Feasible |
|---|---|---|---|---|
| two-stage · pure | 0.1310 | [0.1284, 0.1332] | 0.1371 | 81% |
| two-stage · mixture | 0.1289 | [0.1254, 0.1322] | 0.1473 | 64% |
| one-stage · pure | 0.1266 | [0.1229, 0.1301] | 0.1407 | 97% |
| one-stage · mixture | 0.1240 | [0.1203, 0.1276] | 0.1417 | 81% |

- **Two-stage shows a positive but *non-significant* trend** in both fluid spaces (paired
  Wilcoxon: pure Δη +0.0044, p = 0.076; mixture +0.0049, p = 0.064). Neither crosses α = 0.05.

### Ablation 1 — the operability prior (Tc band)

Re-running with the band **on** (`bench/full2`) lifts two-stage · pure to 0.1356 and makes the
pure advantage highly significant (p = 8×10⁻⁵). So **most of two-stage's apparent pure-fluid
win is the operability prior, not the search algorithm** — removing it drops the effect to
non-significance. One-stage is band-invariant (bit-identical across `full2`/`full3`).

### Ablation 2 — pre-SCBO coverage (`coverage_ablation.png`)

Sweeping `n_property_targets ∈ {8, 20, 40}` shows best η is **flat-to-declining** with
coverage (pure: 0.1327 → 0.1310 → 0.1284), never rising. More cheap screening does **not**
help — the ~25 SCBO evaluations (the only source of efficiency information) are the bottleneck,
and with the band off, denser coverage even mildly misleads the reachability GPC toward
reachable-but-inoperable fluids.

### Caveats

- Two-stage · pure under-spends slightly (mean 23.7 evals; 3/20 seeds stop on candidate
  exhaustion in the finite 61-vertex space) — conservative, since it cuts *against* two-stage.
- Mixtures hold the single highest max (0.1473) but the lowest feasibility (64%).

## History

This suite earned its keep on first run: it caught two data-quality bugs that a mean-only
table had hidden — the two-stage pipeline silently ignoring `--mode pure`, and an isentropic
solver fabricating above-Carnot efficiencies. Both are now fixed in the package (mode-aware
snapping; the solver raises on failure and the simulator enforces a Carnot ceiling), and the
numbers above are from a clean re-run.
