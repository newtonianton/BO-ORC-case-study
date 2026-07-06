# Visualisation suite

Static figures for the ORC-BO benchmark study. Every script reads the per-seed
`scbo_results.csv` files under `bench/full/<stage>_<mode>/seed_*/` and writes PNGs into
**one** folder: [`figures/`](figures/).

## Run

```bash
python viz/make_all.py                 # render everything into viz/figures/
python viz/make_all.py --figdir out/   # or elsewhere
```

Each suite is also runnable on its own: `python viz/convergence.py`, etc.

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

## Caveats these figures surfaced (both now fixed in code)

These two data-quality bugs were found *by* the figures below and have since been fixed in
the package. **The `bench/full/` tree was generated before the fixes, so the current figures
still show the buggy data — regenerate the benchmark and re-render to get clean results:**

```bash
python -m benchmarks.run_benchmark --stages onestage,twostage --modes pure,mixture \
    --backend REFPROP --seeds 20 --n-init 5 --scbo-budget 20 --outdir bench/full --workers 4
python viz/make_all.py
```

- **`twostage_pure` was identical to `twostage_mixture`.** The two-stage pipeline ignored
  `--mode pure` and always snapped to mixtures. Fixed: it now snaps to pure vertices in pure
  mode, so the two trees will differ after re-running.
- **A few two-stage seeds reported η ≈ 0.33** (above the ~0.295 Carnot bound). The
  isentropic-enthalpy solver fabricated a bracket midpoint on flash failure. Fixed: the solver
  now raises on genuine failure and the simulator rejects any η above the Carnot ceiling, so
  no unphysical efficiencies can be recorded.
