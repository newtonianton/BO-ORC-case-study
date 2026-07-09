"""Two-stage property-targeting Bayesian optimization pipeline.

The two-stage method separates *where to look* (property space) from *how to operate*
(operating conditions):

Stage 1 - Targeting (Steps 1-6)
    Latin-hypercube initial mixtures are realized and their ``(Tc, Pc)`` recorded. Property
    targets are proposed and :func:`orc_bo.targeting.run_targeting` drives mixtures toward
    them. A **reachability** GP classifier then proposes additional space-filling targets
    until ``required_valid_init`` targets have been *reached*.

Stage 2 - Realization (Step 7) and exploitation (Step 8)
    The earliest reached targets are handed to SCBO, which optimizes operating conditions
    for each realized mixture. A bounded cEI loop then continues proposing mixtures, weighted
    by **reachability**, until the system budget or failure allowance is exhausted.

Terminology - "feasibility" is deliberately split into three precise notions; the codebase
uses these words consistently:

* **Reachability** (property-space feasibility): whether some realizable fluid/mixture lies
  within ``radius_norm`` of a target point in normalized ``(Tc, Pc)`` space. Modelled by the
  *reachability* GP classifier (``P_prop``); its labels come from
  :func:`orc_bo.targeting.success_mask`. NOTE: the historical field ``required_valid_init``
  and log phrase "valid targets" both refer to *reachability* (a target was reached), NOT to
  validity below.
* **Validity** (operability feasibility): whether a fluid/mixture admits at least one ORC
  operating point that satisfies the constraints, i.e. SCBO returns a positive efficiency.
  Modelled by the *validity* GP classifier (``P_sys``).
* **Constraint feasibility** (operating point): whether a specific ``(p_evap, p_cond)``
  satisfies the pressure-ordering and pinch constraints. This is the SCBO inner-loop notion
  (see :mod:`orc_bo.scbo`); a fluid is *valid* iff at least one operating point is
  constraint-feasible.

This pipeline targets mixtures (the geometric projection produces binary mixtures). It
reuses the validated :mod:`orc_bo.targeting`, :mod:`orc_bo.scbo` and pipeline ``common``
building blocks. End-to-end mixture runs require the REFPROP backend; the plotting and
console-tee machinery of the original script is intentionally omitted.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Set

import torch
from scipy.stats import qmc

from ..config import AppConfig
from ..geometry import MixtureKey, format_mixture_name, snap_selection
from ..logging_setup import get_logger
from ..orc_model import ORCSimulator
from ..scbo import optimize_operating_conditions_robust
from ..seeding import base_seed
from ..targeting import (
    PropNormalizer,
    fit_target_gp,
    gpc_predict_proba,
    greedy_maximin,
    run_targeting,
    success_mask,
    train_gpc,
)
from .. import thermo
from .common import (
    PHASE_INIT,
    PHASE_OPT,
    PHASE_TARGET,
    Candidate,
    Fluid,
    RunWriter,
    TKWARGS,
    format_run_header,
    load_fluids,
    realize_candidate,
)

logger = get_logger(__name__)

PROP_NAMES = ("Tc", "Pc")


@dataclass
class TwoStageResult:
    """Summary of a two-stage run."""

    best_name: str
    best_eta: float
    n_scbo: int
    n_targets_satisfied: int
    outdir: Path


def _earliest_success_ids(labels: torch.Tensor, k: int) -> List[int]:
    """Indices of the first ``k`` satisfied targets (label == 1), in order."""
    ids = [i for i, v in enumerate(labels.reshape(-1).tolist()) if v >= 0.5]
    return ids[:k]


def _sample_band(
    n: int,
    tc_band: tuple[float, float],
    pc_bounds: tuple[float, float],
    *,
    lhs: bool = False,
    seed: int = 0,
) -> torch.Tensor:
    """Sample ``n`` property targets as real ``(Tc, Pc)`` within the operable band.

    ``Tc`` is drawn from ``tc_band`` and ``Pc`` from ``pc_bounds`` (the observed pressure
    range). Latin-hypercube stratification (``lhs=True``) is used for the few initial
    targets; plain uniform sampling for the many GPC-refill candidates.
    """
    if lhs:
        u = torch.tensor(qmc.LatinHypercube(d=2, seed=seed).random(n=n), **TKWARGS)
    else:
        u = torch.rand(n, 2, **TKWARGS)
    tc = tc_band[0] + u[:, 0] * (tc_band[1] - tc_band[0])
    pc = pc_bounds[0] + u[:, 1] * (pc_bounds[1] - pc_bounds[0])
    return torch.stack([tc, pc], dim=1)


def run_twostage(
    csv_path: Path,
    mode: str = "mixture",
    n_init: int = 5,
    scbo_budget: Optional[int] = None,
    outdir: Path = Path("runs/twostage"),
    config: Optional[AppConfig] = None,
) -> TwoStageResult:
    """Run the two-stage pipeline and return a result summary.

    Parameters
    ----------
    csv_path:
        Dataset CSV listing candidate fluids.
    mode:
        Retained for CLI symmetry; the two-stage pipeline operates on mixtures.
    n_init:
        Number of Latin-hypercube initial mixtures.
    scbo_budget:
        Step-8 cEI budget (number of exploitation proposals). This is the ``--scbo-budget``
        CLI flag; when ``None`` it falls back to ``config.twostage.system_budget``. Note the
        total number of SCBO evaluations is roughly ``required_valid_init`` (Step 7) plus this.
    outdir:
        Output directory; a ``seed_XXX`` subdirectory is created within it.
    config:
        Application configuration; defaults to :class:`AppConfig` defaults.

    Returns
    -------
    TwoStageResult
        Best mixture found, its efficiency, and run statistics.
    """
    config = config or AppConfig()
    ts = config.twostage
    if scbo_budget is None:
        scbo_budget = ts.system_budget
    threshold = config.mixture.composition_threshold
    seed = base_seed()
    subdir = Path(outdir) / f"seed_{seed:03d}"

    fluids = load_fluids(Path(csv_path))
    t_dim = len(fluids)
    onehot = torch.eye(t_dim, **TKWARGS)
    simulator = ORCSimulator(orc=config.orc, backend=config.thermo.backend)
    logger.info(
        "Two-stage: %d fluids | n_init=%d n_targets=%d required_valid_init=%d radius_norm=%.3f "
        "gpc_max_rounds=%d scbo_budget=%d failure_allowance=%d scbo_max_retries=%d | backend=%s seed=%d",
        t_dim, n_init, ts.n_property_targets, ts.required_valid_init, ts.radius_norm,
        ts.gpc_max_rounds, scbo_budget, ts.failure_allowance, config.bo.scbo_max_retries,
        config.thermo.backend, seed,
    )

    evaluated_mixtures: Set[MixtureKey] = set()
    metadata: List[MixtureKey] = []
    p_rows: List[List[float]] = []

    # ---- Stage 1, Step 1-5: initial LHS selections and their properties ----
    lhs = torch.tensor(qmc.LatinHypercube(d=t_dim, seed=seed).random(n=n_init), **TKWARGS)
    for k in range(lhs.shape[0]):
        j1, j2, x1 = snap_selection(mode, lhs[k], onehot, evaluated_mixtures, threshold)
        evaluated_mixtures.add((j1, j2, x1))
        f1 = fluids[j1]
        f2 = fluids[j2] if j2 is not None else None
        tc, pc = thermo.critical_properties(
            f1.name, f2.name if f2 else None, x1, config.thermo,
            refprop1=f1.refprop, refprop2=f2.refprop if f2 else None,
        )
        metadata.append((j1, j2, x1))
        p_rows.append([tc, pc])
    p_real = torch.tensor(p_rows, **TKWARGS)

    normalizer = PropNormalizer(PROP_NAMES)
    normalizer.fit_from_real_points(p_real)

    # Tc range for property targets. When use_tc_band is set, restrict targets to the operable
    # band (a domain prior that excludes inoperable high-Tc fluids); otherwise targets span the
    # full observed range, matching the one-stage pipeline's unrestricted fluid access. Pc
    # always spans the observed range.
    source_k = config.orc.t_in_source_c + 273.15
    obs_tc_lo, obs_tc_hi = float(normalizer.bounds["lower"][0]), float(normalizer.bounds["upper"][0])
    if ts.use_tc_band:
        tc_lo = max(ts.tc_min_k if ts.tc_min_k is not None else source_k, obs_tc_lo)
        tc_hi = min(ts.tc_max_k if ts.tc_max_k is not None else source_k + 200.0, obs_tc_hi)
        if tc_hi <= tc_lo:
            logger.warning("Operable Tc band collapsed; falling back to full observed Tc range")
            tc_lo, tc_hi = obs_tc_lo, obs_tc_hi
    else:
        tc_lo, tc_hi = obs_tc_lo, obs_tc_hi
    tc_band = (tc_lo, tc_hi)
    pc_bounds = (float(normalizer.bounds["lower"][1]), float(normalizer.bounds["upper"][1]))
    logger.info("Property-target Tc range: [%.1f, %.1f] K (band=%s, source %.1f K)",
                tc_lo, tc_hi, "on" if ts.use_tc_band else "off", source_k)

    # Initial property targets: LHS within the (Tc, Pc) box.
    asked_targets_real = _sample_band(ts.n_property_targets, tc_band, pc_bounds, lhs=True, seed=seed)

    metadata, p_real, evaluated_mixtures, _ = run_targeting(
        asked_targets_real, metadata, p_real, normalizer, onehot, evaluated_mixtures,
        fluids, config, radius=ts.radius_norm, budget_per_target=ts.target_budget, mode=mode,
    )

    # ---- Step 6 (phase TARGET): reachability-GPC space-filling until enough targets reached ----
    normalizer.maybe_expand(p_real)
    flags, _, _ = success_mask(
        normalizer.to_norm(p_real, clip=False),
        normalizer.to_norm(asked_targets_real, clip=False), ts.radius_norm,
    )
    # labels/num_valid track REACHABILITY (a target was reached), not validity (operability).
    labels = torch.tensor([[1.0 if f else 0.0] for f in flags], **TKWARGS)
    num_reached = int(sum(flags))
    logger.info("[%s] initially reached %d of %d targets (need >= %d to skip space-filling)",
                PHASE_TARGET, num_reached, len(asked_targets_real), ts.required_valid_init)

    gpc_round = 0
    while num_reached < ts.required_valid_init and gpc_round < ts.gpc_max_rounds:
        gpc_round += 1
        x_targets = normalizer.to_norm(asked_targets_real, clip=False)
        # Reachability GPC: P(a target here is reachable by some realizable mixture).
        model, likelihood = train_gpc(x_targets, labels, steps=ts.gpc_steps, lr=ts.gpc_lr)
        # Refill candidates are drawn only from the operable Tc band, then normalized for the GPC.
        candidates = normalizer.to_norm(
            _sample_band(ts.gpc_candidates, tc_band, pc_bounds), clip=False
        )
        proba = gpc_predict_proba(model, likelihood, candidates)
        reachable = candidates[proba >= ts.gpc_feasibility_threshold]
        if reachable.numel() == 0:
            logger.info("[%s] reachability GPC found no reachable region in round %d; stopping",
                        PHASE_TARGET, gpc_round)
            break
        to_add = ts.required_valid_init - num_reached
        new_targets_norm = greedy_maximin(x_targets, reachable, k=to_add)
        new_targets_real = normalizer.to_real(new_targets_norm)

        metadata, p_real, evaluated_mixtures, _ = run_targeting(
            new_targets_real, metadata, p_real, normalizer, onehot, evaluated_mixtures,
            fluids, config, radius=ts.radius_norm, budget_per_target=ts.target_budget, mode=mode,
        )
        asked_targets_real = torch.cat([asked_targets_real, new_targets_real], dim=0)
        normalizer.maybe_expand(p_real)
        flags, _, _ = success_mask(
            normalizer.to_norm(p_real, clip=False),
            normalizer.to_norm(asked_targets_real, clip=False), ts.radius_norm,
        )
        labels = torch.tensor([[1.0 if f else 0.0] for f in flags], **TKWARGS)
        num_reached = int(sum(flags))
        logger.info("[%s] after round %d: reached %d of %d targets (need >= %d)",
                    PHASE_TARGET, gpc_round, num_reached, len(asked_targets_real), ts.required_valid_init)

    # ---- Step 7 (phase INIT): SCBO on the earliest satisfied targets' realized mixtures ----
    _, success_rows, _ = success_mask(
        normalizer.to_norm(p_real, clip=False),
        normalizer.to_norm(asked_targets_real, clip=False), ts.radius_norm,
    )
    realized_ids = sorted({row for ok, row in zip(flags, success_rows) if ok and row >= 0})

    best_eta = -float("inf")
    best_name = "(none)"
    n_scbo = 0
    scbo_props: List[List[float]] = []
    scbo_eta: List[float] = []
    header = format_run_header(config, stage="twostage", mode=mode, seed=seed,
                               n_init=n_init, scbo_budget=scbo_budget)
    with RunWriter(subdir, header=header) as writer:
        order = 0
        for idx in realized_ids[: ts.required_valid_init]:
            j1, j2, x1 = metadata[idx]
            cand = realize_candidate(mode, j1, j2, x1, fluids, onehot, config)
            if cand is None:
                continue
            eta, p_evap, p_cond = optimize_operating_conditions_robust(
                cand.wf, cand.pc, cand.ptriple, simulator, config.bo
            )
            order += 1
            n_scbo += 1
            writer.record(PHASE_INIT, order, mode, cand, eta, p_evap, p_cond)
            scbo_props.append([cand.tc, cand.pc])
            scbo_eta.append(eta)
            logger.info("[%s %d] %s: eta=%.5f", PHASE_INIT, order, cand.name, eta)
            if eta > best_eta:
                best_eta, best_name = eta, cand.name

        # ---- Step 8 (phase OPT): bounded cEI exploitation (EI x reachability x validity) ----
        best_eta, best_name, order = _exploitation_loop(
            writer, order, fluids, onehot, evaluated_mixtures, normalizer,
            asked_targets_real, labels, simulator, config, best_eta, best_name,
            scbo_props, scbo_eta, system_budget=scbo_budget, mode=mode,
        )

    result = TwoStageResult(
        best_name=best_name,
        best_eta=best_eta if best_eta > -float("inf") else simulator.orc.infeasible_penalty,
        n_scbo=n_scbo,
        n_targets_satisfied=num_reached,
        outdir=subdir,
    )
    logger.info("Two-stage complete: best %s eta=%.5f -> %s",
                result.best_name, result.best_eta, subdir)
    logger.info("Backend coverage: %s", simulator.backend_failure_report())
    return result


def _exploitation_loop(
    writer: RunWriter,
    order: int,
    fluids: List[Fluid],
    onehot: torch.Tensor,
    evaluated_mixtures: Set[MixtureKey],
    normalizer: PropNormalizer,
    asked_targets_real: torch.Tensor,
    labels: torch.Tensor,
    simulator: ORCSimulator,
    config: AppConfig,
    best_eta: float,
    best_name: str,
    scbo_props: List[List[float]],
    scbo_eta: List[float],
    system_budget: int,
    mode: str = "mixture",
    n_screen: int = 64,
) -> tuple[float, str, int]:
    """Step-8 probability-weighted cEI exploitation loop.

    Scores screened candidate mixtures by ``EI x P_prop x P_sys`` and SCBOs the best:

    * ``P_prop`` - **reachability** GPC over property-target coordinates: P(a target here is
      reachable by some realizable mixture).
    * ``P_sys``  - **validity** GPC over SCBO'd mixtures: P(a constraint-feasible operating
      point exists, i.e. eta > 0).
    * ``EI``     - expected efficiency improvement from a GP of (property -> eta), fit on
      *valid* outcomes only.

    Stops after ``system_budget`` proposals or ``failure_allowance`` consecutive invalid
    (no feasible operating point) evaluations; ``failure_allowance <= 0`` disables early
    stopping so the full budget is always spent (budget-matched comparison).
    """
    ts = config.twostage
    if labels.sum() <= 0:
        logger.info("[%s] no reached targets to train the reachability GPC; skipping", PHASE_OPT)
        return best_eta, best_name, order

    # P_prop: reachability GPC over property-target coordinates.
    x_targets = normalizer.to_norm(asked_targets_real, clip=False)
    prop_model, prop_lik = train_gpc(x_targets, labels, steps=ts.gpc_steps, lr=ts.gpc_lr)

    budget = system_budget
    failures = 0
    while budget > 0 and (ts.failure_allowance <= 0 or failures < ts.failure_allowance):
        # (Re)train the validity GPC (P_sys) and efficiency GP from accumulated outcomes.
        orc_gpc = orc_lik = eff_gp = None
        if scbo_props:
            props_t = normalizer.to_norm(torch.tensor(scbo_props, **TKWARGS), clip=True)
            valid = [e > 0 for e in scbo_eta]  # validity: a feasible operating point exists
            # P_sys (validity) is always applied when trainable — it needs both a valid and
            # an invalid example present (a one-class classifier is undefined).
            if any(valid) and not all(valid):
                y_valid = torch.tensor([[1.0 if v else 0.0] for v in valid], **TKWARGS)
                orc_gpc, orc_lik = train_gpc(props_t, y_valid, steps=ts.gpc_steps, lr=ts.gpc_lr)
            # Efficiency GP for the EI base — VALID outcomes only (eta > 0). The infeasible
            # penalty is a sentinel, not an efficiency, and would corrupt the regression.
            valid_pts = [(p, e) for p, e in zip(scbo_props, scbo_eta) if e > 0]
            if len(valid_pts) >= 3:
                P_valid = normalizer.to_norm(
                    torch.tensor([p for p, _ in valid_pts], **TKWARGS), clip=True)
                y_eta = torch.tensor([e for _, e in valid_pts], **TKWARGS).reshape(-1, 1)
                eff_gp = fit_target_gp(P_valid, y_eta)
        chosen = _propose_by_cei(
            prop_model, prop_lik, orc_gpc, orc_lik, eff_gp, best_eta, fluids, onehot,
            evaluated_mixtures, normalizer, config, mode, n_screen,
        )
        if chosen is None:
            logger.info("[%s] no novel candidate could be realized; stopping", PHASE_OPT)
            break

        eta, p_evap, p_cond = optimize_operating_conditions_robust(
            chosen.wf, chosen.pc, chosen.ptriple, simulator, config.bo
        )
        order += 1
        budget -= 1
        writer.record(PHASE_OPT, order, mode, chosen, eta, p_evap, p_cond)
        scbo_props.append([chosen.tc, chosen.pc]) # outcome goes back into next iteration
        scbo_eta.append(eta)
        logger.info("[%s %d] %s: eta=%.5f", PHASE_OPT, order, chosen.name, eta)
        if eta > 0:
            failures = 0
            if eta > best_eta:
                best_eta, best_name = eta, chosen.name
        else:
            failures += 1

    return best_eta, best_name, order


def _propose_by_cei(
    prop_model,
    prop_lik,
    orc_gpc, 
    orc_lik,
    eff_gp,
    best_eta: float,
    fluids: List[Fluid],
    onehot: torch.Tensor,
    evaluated_mixtures: Set[MixtureKey],
    normalizer: PropNormalizer,
    config: AppConfig,
    mode: str,
    n_screen: int,
) -> Optional[Candidate]:
    """Screen novel candidates; pick max of EI x P_prop x P_sys."""
    realized: List[Candidate] = []
    seen: Set[MixtureKey] = set()
    suggestions = torch.rand(n_screen, onehot.shape[0], **TKWARGS)
    for i in range(n_screen):
        j1, j2, x1 = snap_selection(
            mode, suggestions[i], onehot, evaluated_mixtures,
            config.mixture.composition_threshold,
        )
        if (j1, j2, x1) in evaluated_mixtures or (j1, j2, x1) in seen:
            continue
        seen.add((j1, j2, x1))
        cand = realize_candidate(mode, j1, j2, x1, fluids, onehot, config)
        if cand is not None:
            realized.append(cand)

    if not realized:
        return None

    props_norm = normalizer.to_norm(
        torch.tensor([[c.tc, c.pc] for c in realized], **TKWARGS), clip=True
    )

    # P_prop: reachability (target reachable by some realizable mixture).
    score = gpc_predict_proba(prop_model, prop_lik, props_norm)
    # P_sys: validity (a constraint-feasible ORC operating point exists).
    if orc_gpc is not None:
        score = score * gpc_predict_proba(orc_gpc, orc_lik, props_norm)
    # EI: expected efficiency improvement (cEI base) — only with a valid incumbent (eta > 0).
    if eff_gp is not None and best_eta > 0.0:
        try:
            from botorch.acquisition.analytic import LogExpectedImprovement
            ei = LogExpectedImprovement(eff_gp, best_f=best_eta)
            with torch.no_grad():
                score = score * ei(props_norm.unsqueeze(1)).exp()
        except Exception as exc:  # analytic EI can fail on degenerate posteriors
            logger.debug("EI term skipped: %s", exc)

    best = realized[int(torch.argmax(score).item())]
    evaluated_mixtures.add((best.j1, best.j2, best.x1))
    return best
