# Results (draft)

Placeholder figure/table numbers (`Fig. N`, `Table N`) map to `viz/figures/`. Statistics use
paired Wilcoxon signed-rank tests over 20 seeds unless stated; confidence intervals are 95 %
percentile bootstrap (10 000 resamples) of the mean.

## 4.1 Experimental setup

We compared four configurations — the one-stage and two-stage Bayesian-optimisation pipelines,
each applied to pure working fluids and to binary mixtures — on maximising ORC thermal
efficiency $\eta$ for a fixed heat source (150 °C, cooled to 135 °C) and sink (25 °C, warmed to
35 °C). Candidate fluids were drawn from a database of 61 species. Each configuration was run
for 20 independent seeds; seed $k$ fixes all pseudo-random state, so a given seed index drives
comparable initialisations across configurations, giving a **paired** design.

Both pipelines were budgeted on a common resource, the *SCBO evaluation*: for one candidate
fluid, a full constrained trust-region (TuRBO) optimisation of the operating pressures
$(p_\mathrm{evap}, p_\mathrm{cond})$ subject to the pinch and pressure-ordering constraints,
returning that fluid's best feasible efficiency. This is the dominant cost — each evaluation
drives hundreds to thousands of REFPROP equation-of-state calls — and every other component is
cheaper by two to three orders of magnitude. The **primary comparison budgets both pipelines
at 25 SCBO evaluations** (5 initial + 20 acquisition-loop). For the two-stage pipeline this
means 5 realisation evaluations (Step 7) plus 20 exploitation evaluations (Step 8); early
stopping of the exploitation loop is disabled so the full budget is spent. To isolate the
**search algorithm** rather than injected domain knowledge, the two-stage pipeline's optional
operability prior (a critical-temperature band on its property targets) is **disabled** in the
primary comparison; its effect is quantified separately as an ablation (§4.3).

Performance is the best-of-run efficiency (maximum $\eta$ over a run; infeasible designs carry
a penalty). We report the mean, 95 % CI, single best, and the fraction of evaluations that were
feasible. All four configurations share the fluid database, ORC boundary conditions, the
REFPROP backend, and a Carnot efficiency ceiling ($\eta \le 0.295$ for this source/sink);
across all runs no configuration produced a super-Carnot efficiency.

## 4.2 End-to-end performance

Table 1 summarises the budget-matched, prior-free comparison. The two-stage pipeline attained
a higher mean efficiency than the one-stage pipeline in both fluid spaces, but in neither case
was the difference statistically significant: pure fluids $\Delta\eta = +0.0044$ ($p = 0.076$),
mixtures $\Delta\eta = +0.0049$ ($p = 0.064$). Both trends point the same way and sit just
outside the conventional $\alpha = 0.05$ threshold. Under a fair budget and without a domain
prior, therefore, **property-space targeting yields a positive but non-significant improvement
over direct Bayesian optimisation.**

**Table 1.** Best-of-run efficiency over 20 seeds (budget-matched, no domain prior).

| Configuration | Mean $\eta$ | 95 % CI | Best | Feasible |
|---|---|---|---|---|
| Two-stage · pure | 0.1310 | [0.1284, 0.1332] | 0.1371 | 81 % |
| Two-stage · mixture | 0.1289 | [0.1254, 0.1322] | 0.1473 | 64 % |
| One-stage · pure | 0.1266 | [0.1229, 0.1301] | 0.1407 | 97 % |
| One-stage · mixture | 0.1240 | [0.1203, 0.1276] | 0.1417 | 81 % |

Convergence traces (Fig. 1) show the two-stage pipeline climbing slightly earlier on pure
fluids — consistent with its property-screening front-end — but the pure trajectories converge
to overlapping bands, and the mixture trajectories are indistinguishable throughout.

## 4.3 Ablation 1 — the operability prior

The two-stage pipeline can restrict its property targets to an *operable* critical-temperature
band $[\,T_\text{source}, T_\text{source}+200\,\text{K}]$, excluding fluids that cannot run the
cycle. This is domain knowledge the one-stage pipeline does not receive, so it is disabled in
the primary comparison. Re-enabling it (Table 2) raises two-stage · pure from 0.1310 to 0.1356
and turns the pure-fluid advantage from non-significant into highly significant
($p = 8\times10^{-5}$). The one-stage pipeline is unaffected by this setting (its results are
bit-identical with and without the prior).

**Table 2.** Effect of the operability prior on the two- vs one-stage comparison (pure fluids).

| | Two-stage · pure mean $\eta$ | Paired $\Delta\eta$ vs one-stage | $p$ |
|---|---|---|---|
| Prior **off** (primary) | 0.1310 | $+0.0044$ | 0.076 |
| Prior **on** (ablation) | 0.1356 | $+0.0090$ | $8\times10^{-5}$ |

We conclude that **the significant pure-fluid advantage reported under the operability prior is
substantially attributable to that prior, not to the search strategy.** The prior is a valid
and effective ingredient — but it is domain knowledge, and crediting it to the algorithm would
overstate the latter's contribution.

## 4.4 Ablation 2 — pre-SCBO property coverage

Because the two-stage property-screening phase is cheap (critical-point lookups, no cycle
simulation), one might expect denser screening — a better-trained reachability model — to
improve results. We swept the number of property targets, `n_property_targets`
$\in \{8, 20, 40\}$ (≈ 13 %, 33 %, 66 % of the fluid set), holding the SCBO budget fixed
(Fig. 5). Best efficiency was **flat-to-declining** in coverage: two-stage · pure gave
$0.1327 \to 0.1310 \to 0.1284$ and two-stage · mixture $0.1311 \to 0.1289 \to 0.1287$, with
feasibility likewise not improving. All differences lie within overlapping confidence intervals.

More cheap screening therefore does **not** improve performance, and the leanest level was if
anything the best. Two mechanisms explain this. First, property coordinates carry no efficiency
information — that enters only through the ~25 SCBO evaluations, which remain the binding
constraint regardless of coverage. Second, with the operability prior off, denser coverage maps
more of the *inoperable* property region as reachable, mildly biasing the reachability model
toward reachable-but-inoperable fluids. This also shows the pure-fluid advantage lost in §4.3
cannot be recovered by more screening: the prior's value was the operability restriction itself,
not the quantity of reconnaissance.

## 4.5 Feasibility and budget notes

Feasibility (Table 1, Fig. 4) was highest for one-stage · pure (97 %) and lowest for
two-stage · mixture (64 %): mixtures are harder to operate under the pinch constraints, and the
two-stage pipeline expends budget exploring property regions whose operability is only tested at
the SCBO stage. Mixtures held the single highest efficiency of any run (0.1473) but with greater
seed-to-seed variance. Finally, two-stage · pure used slightly under its budget (mean 23.7 of 25
evaluations; 3/20 seeds terminated on candidate exhaustion in the finite 61-fluid space) — a
minor asymmetry that works against, not for, the two-stage pipeline.

---

### Summary of the honest claim

> Under matched evaluation budgets and no domain prior, two-stage property-targeting shows a
> positive but statistically non-significant improvement over one-stage in both fluid spaces
> ($\Delta\eta \approx +0.004$–$0.005$, $p \approx 0.06$–$0.08$). The significant advantage
> observed when the two-stage pipeline is augmented with an operability prior
> ($p < 10^{-4}$, pure fluids) is substantially attributable to that prior. Additional cheap
> pre-SCBO property screening does not improve performance, confirming that the expensive
> operating-condition optimisation — not property-space coverage — is the binding constraint.
