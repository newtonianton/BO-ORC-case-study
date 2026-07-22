# Data & Figure Catalog

What every `bench/` results directory and `viz/figures/` image represents, so runs can be
told apart and the right dataset backs each paper claim. **CURRENT** = use this;
**SUPERSEDED** = kept for the audit trail, do not use for new analysis.

Common setup unless a row says otherwise: 61-fluid database, **REFPROP** backend, **20 seeds**
(0–19), ORC source 150 °C → 135 °C / sink 25 °C → 35 °C, `n_init = 5`. Each config lives in
`<dir>/<stage>_<mode>/seed_XXX/`.

---

## The three files in every `seed_XXX/`

| File | Contents |
|---|---|
| `scbo_results.csv` | One row per SCBO evaluation. Columns: `phase, order, mixture, mode, comp1, comp2, x1, Tc_K, Pc_Pa, eta, p_evap_bar, p_cond_bar, cost`. `eta` is the fluid's best efficiency (or the −0.05 penalty if infeasible); `cost` is cumulative SCBO-equivalent cost at that row. |
| `sequence.csv` | `order, mixture` — the evaluation order (redundant with the CSV, kept for quick reads). |
| `summary.txt` | `#`-commented parameter block (the full resolved config — `[orc] [bo] [mixture] [twostage] [thermo]`), then the list of evaluated fluid names, then footer notes: `# total cost spent`, `# carnot_guard: r/n …`, `# backend could not evaluate f/n fluids`. **The parameter block is the ground truth for what a run's settings were.** |

`phase` values: `INIT` = initial batch (one-stage LHS init / two-stage Step-7 realization);
`OPT` = surrogate-guided loop (one-stage qLogEI / two-stage Step-8 cEI); `TARGET` = two-stage
Stage-1 targeting (logs only, writes no result rows).

---

## Data directories (`bench/`)

### Head-to-head (the main result)

| Dir | Configs | Budget | Distinguishing settings | Status / purpose |
|---|---|---|---|---|
| `cost01/` | 1-stage & 2-stage × pure & mixture | cost-weighted, `cost_budget=20`, `lab_cost=0.1` | `step8=screen`, neutral prior, `min_mole_frac=0.05`; one-stage novelty bug **fixed**, one-stage-mixture budget leak **fixed** | **CURRENT — primary cost-matched comparison.** Backs the "no significant one- vs two-stage difference" result. |
| `cost01_leaky_onestage_mixture/` | 1-stage mixture (flat `seed_XXX/`) | as `cost01` | pre-leak-fix one-stage mixture | **SUPERSEDED** by `cost01/onestage_mixture` (kept to show the budget-leak audit). |

### Ablations off the `cost01` baseline

| Dir | Configs | Distinguishing setting | Purpose |
|---|---|---|---|
| `cost01_inverse/` | 2-stage pure & mixture | `step8_proposal=inverse` | Inverse-design vs Monte-Carlo screen (`cost01`). Result: statistically ties screen. |
| `cost01_vprior/` | 2-stage mixture | `validity_prior_mean=-1.0` | Pessimistic validity prior (GPC2). **Best mixture arm.** |
| `cost000/`, `cost002/`, `cost005/` | 2-stage pure & mixture | `lab_cost = 0.0 / 0.02 / 0.05` | Lab-to-process **cost-ratio sweep** (with `cost01` = 0.1). Only two-stage varies with `lab_cost`; one-stage baseline is in `cost01`. |
| `cov08/`, `cov40/` | 2-stage pure & mixture | `n_property_targets = 8 / 40` (count-based budget) | Pre-SCBO **coverage ablation** (with `cost01` ≈ 20 as the mid point). |
| `relax01/` | 1-stage & 2-stage mixture | `min_mole_frac=0.01` (vs 0.05) | **Composition-clamp ablation** — does relaxing the clamp let mixtures climb toward pure water? Pairs against `cost01` mixtures. *(newly added)* |
| `relax01_vprior/` | 2-stage mixture | `min_mole_frac=0.01` + `validity_prior_mean=-1.0` | Relaxed clamp on the best arm; pairs against `cost01_vprior`. *(newly added)* |

### Historical (do not use for new analysis)

| Dir | Configs | Why superseded |
|---|---|---|
| `full2/` | all 4 | Earliest run: `required_valid_init=8`, `failure_allowance=3` (early stopping ON), operability-band era, count-based unequal budgets, **and** the one-stage novelty bug. **SUPERSEDED.** |
| `full3/` | all 4 | Band-off, budget-match *attempt* (`failure_allowance=0`), but count-based (no cost model) and **still carries the one-stage novelty bug** (~5 distinct fluids/run). **SUPERSEDED by `cost01`.** Useful only as the before-picture for the one-stage-bug fix. |

---

## Figures (`viz/figures/`)

Rendered per-dataset into subfolders. Regenerate with `ORC_BO_BENCH=bench/<dir> python viz/make_all.py`
(and `viz/cost_curve.py`, `viz/coverage_ablation.py`, `viz/ratio_sweep` as applicable).

### `viz/figures/cost01/` — CURRENT figures for the main result

| File | Shows |
|---|---|
| `convergence.png` | Best-η-so-far vs SCBO evaluation index (median + IQR), per config. |
| `cost_curve.png` | Best-η-so-far vs **cumulative cost** — the cost-matched view; two-stage starts right-shifted by its screening cost. |
| `final_performance.png` | Final best-η per config with bootstrap 95% CIs. |
| `feasibility_rate.png` | Fraction of feasible/valid evaluations per config. |
| `property_space.png` | (Tc, Pc) scatter of evaluated fluids (the property landscape the search explores). |
| `seed_distribution.png` | Per-seed best-η spread per config. |
| `coverage_ablation.png` | Two-stage best-η vs `n_property_targets` {8, 20, 40} with one-stage baseline (uses `cov08/`, `cost01/`, `cov40/`). |

### `viz/figures/full3/` — **SUPERSEDED**
Same figure set (no `cost_curve`, since `full3` is count-based) from the historical `full3` run.

### `viz/figures/ratio_sweep/`
| File | Shows |
|---|---|
| `ratio_sweep.png` | Best-valid-η vs cumulative cost, one line per cost ratio `lab_cost` {0, 0.02, 0.05, 0.1} — **per-ratio median** across seeds + dashed one-stage baseline (uses `cost000/002/005/` + `cost01/`). |
| `ratio_sweep_mean.png` | Same, but each line is the **per-ratio mean** across seeds (sensitive to slow/not-yet-valid seeds; rises lower/smoother in the ramp-up). |

---

## Quick reference — which data backs which claim

- **One- vs two-stage parity (headline):** `cost01/` (all four configs).
- **The advantage was artifacts:** `full3` → `cost01` (band off, bug fixed, budgets matched).
- **Inverse ≈ screen:** `cost01_inverse/` vs `cost01/` (two-stage).
- **Pessimistic prior helps operability:** `cost01_vprior/` vs `cost01/twostage_mixture`.
- **Coverage is flat-to-declining:** `cov08/`, `cost01/`, `cov40/`.
- **Cost-ratio sensitivity:** `cost000/002/005/` + `cost01/`.
- **Composition clamp does NOT cap mixtures (null result):** `relax01/` ≈ `cost01/` mixtures (relaxing `min_mole_frac` 0.05→0.01 changes nothing; the mixture gap is real, not a clamp artifact).
- **Mixtures < pure fluids (~10%):** `cost01/` pure vs mixture configs.
