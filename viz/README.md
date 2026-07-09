# Visualisation suite

Static figures for the ORC-BO benchmark study. Every script reads the per-seed
`scbo_results.csv` files under `bench/full2/<stage>_<mode>/seed_*/` and writes PNGs into
**one** folder: [`figures/`](figures/).

## Run

```bash
python viz/make_all.py                 # render everything into viz/figures/
python viz/make_all.py --figdir out/   # or elsewhere
```

The results tree defaults to `bench/full2`; point elsewhere with the `ORC_BO_BENCH`
environment variable. Each suite is also runnable on its own: `python viz/convergence.py`, etc.

## Figures

| File | What it shows |
|------|---------------|
| `convergence.png` | Best valid η found so far vs SCBO-evaluation count; median + 25–75th-pct band across seeds. Panels: pure \| mixture; colour = stage. |
| `final_performance.png` | Mean best η per configuration with 95 % bootstrap CI (same methodology as the LaTeX table). |
| `seed_distribution.png` | Per-seed best η (box = median/IQR, points = seeds). Reveals spread and outliers the mean hides. |
| `property_space.png` | Every evaluated fluid in (Tc, Pc) space coloured by η on the blue ramp; `×` = infeasible. 2×2 by config. |
| `feasibility_rate.png` | Share of SCBO evaluations that were operable (η > 0), per configuration. |

## Design

Colour = **stage** (one-stage = blue, two-stage = aqua), validated colourblind-safe; mode is
carried by panel/x-group, never colour, so the stage comparison stays consistent across every
figure. Efficiency magnitude uses a single-hue blue sequential ramp. Palette and matplotlib
defaults live in [`style.py`](style.py).

## Findings (bench/full2, 20 seeds/config)

| Configuration | Mean best η | 95% CI | std | Max | Feasible |
|---|---|---|---|---|---|
| **two-stage · pure** | **0.1356** | [0.1347, 0.1363] | ±0.0019 | 0.1377 | 84% |
| two-stage · mixture | 0.1278 | [0.1249, 0.1310] | ±0.0071 | 0.1454 | 73% |
| one-stage · pure | 0.1266 | [0.1229, 0.1301] | ±0.0084 | 0.1407 | 97% |
| one-stage · mixture | 0.1240 | [0.1203, 0.1276] | ±0.0086 | 0.1417 | 81% |

- **Two-stage beats one-stage in both fluid spaces** — property-targeting helps.
- **`two-stage · pure` is significantly best and remarkably consistent** (non-overlapping CI,
  std ±0.0019). It concentrates on the top pure fluids — Methanol won the best-fluid slot in
  10/20 seeds, Ethanol in 4. One-stage · pure instead scatters across 14 different winners,
  which is why its mean is lower and its variance ~4× larger.
- **Consistency, not a higher ceiling.** Two-stage · pure's *max* (0.1377) is actually below
  one-stage · pure's (0.1407, a lucky Water seed); the gap is SCBO pressure-optimisation noise
  on the same top fluids. Mixtures hold the single highest max of all (0.1454) but with more
  variance and the lowest feasibility (73%).
- **Caveat for write-up:** two-stage · pure's tiny variance partly reflects the small candidate
  pool (~61 discrete vertices) — targeting over a handful of points converges hard. State this
  so it is not over-read as a general BO property.

## History

This suite earned its keep on first run: it caught two data-quality bugs that a mean-only
table had hidden — the two-stage pipeline silently ignoring `--mode pure`, and an isentropic
solver fabricating above-Carnot efficiencies. Both are now fixed in the package (mode-aware
snapping; the solver raises on failure and the simulator enforces a Carnot ceiling), and the
numbers above are from a clean re-run.
