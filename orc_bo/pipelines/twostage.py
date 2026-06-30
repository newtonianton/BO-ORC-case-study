"""Two-stage property-targeting Bayesian optimization pipeline.

The two-stage method separates *where to look* (property space) from *how to operate*
(operating conditions):

Stage 1 - Targeting (Steps 1-6)
    Latin-hypercube initial mixtures are realized and their ``(Tc, Pc)`` recorded. Property
    targets are proposed and :func:`orc_bo.targeting.run_targeting` drives mixtures toward
    them. A variational GP classifier (GPC) over feasibility then proposes additional
    space-filling targets until ``required_valid_init`` targets are satisfied.

Stage 2 - Realization (Step 7) and exploitation (Step 8)
    The earliest satisfied targets are handed to SCBO, which optimizes operating conditions
    for each realized mixture. A bounded cEI loop then continues proposing mixtures, weighted
    by GPC feasibility, until the system budget or failure allowance is exhausted.

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
from ..geometry import MixtureKey, format_mixture_name, snap_to_mixture
from ..logging_setup import get_logger
from ..orc_model import ORCSimulator
from ..scbo import optimize_operating_conditions_robust
from ..seeding import base_seed
from ..targeting import (
    PropNormalizer,
    gpc_predict_proba,
    greedy_maximin,
    run_targeting,
    success_mask,
    train_gpc,
)
from .. import thermo
from .common import Candidate, RunWriter, TKWARGS, load_fluids, realize_candidate

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


def run_twostage(
    csv_path: Path,
    mode: str = "mixture",
    n_init: int = 5,
    scbo_budget: int = 3,
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
        Step-8 cEI system budget (number of exploitation proposals).
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
    threshold = config.mixture.composition_threshold
    seed = base_seed()
    subdir = Path(outdir) / f"seed_{seed:03d}"

    fluids = load_fluids(Path(csv_path))
    t_dim = len(fluids)
    onehot = torch.eye(t_dim, **TKWARGS)
    simulator = ORCSimulator(orc=config.orc, backend=config.thermo.backend)
    logger.info("Two-stage: %d fluids, n_init=%d, seed=%d", t_dim, n_init, seed)

    evaluated_mixtures: Set[MixtureKey] = set()
    metadata: List[MixtureKey] = []
    p_rows: List[List[float]] = []

    # ---- Stage 1, Step 1-5: initial LHS mixtures and their properties ----
    lhs = torch.tensor(qmc.LatinHypercube(d=t_dim, seed=seed).random(n=n_init), **TKWARGS)
    for k in range(lhs.shape[0]):
        j1, j2, x1 = snap_to_mixture(lhs[k], onehot, evaluated_mixtures, composition_threshold=threshold)
        evaluated_mixtures.add((j1, j2, x1))
        tc, pc = thermo.critical_properties(fluids[j1], fluids[j2], x1, config.thermo)
        metadata.append((j1, j2, x1))
        p_rows.append([tc, pc])
    p_real = torch.tensor(p_rows, **TKWARGS)

    normalizer = PropNormalizer(PROP_NAMES)
    normalizer.fit_from_real_points(p_real)

    # Initial property targets via LHS in normalized property space.
    targets_norm = torch.tensor(
        qmc.LatinHypercube(d=len(PROP_NAMES), seed=seed).random(n=ts.n_property_targets), **TKWARGS
    )
    asked_targets_real = normalizer.to_real(targets_norm)

    metadata, p_real, evaluated_mixtures, _ = run_targeting(
        asked_targets_real, metadata, p_real, normalizer, onehot, evaluated_mixtures,
        fluids, config, radius=ts.radius_norm, budget_per_target=ts.target_budget,
    )

    # ---- Step 6: GPC space-filling rounds until enough targets are satisfied ----
    normalizer.maybe_expand(p_real)
    flags, _, _ = success_mask(
        normalizer.to_norm(p_real, clip=False),
        normalizer.to_norm(asked_targets_real, clip=False), ts.radius_norm,
    )
    labels = torch.tensor([[1.0 if f else 0.0] for f in flags], **TKWARGS)
    num_valid = int(sum(flags))
    logger.info("[Step 6] initially valid targets: %d/%d", num_valid, ts.required_valid_init)

    gpc_round = 0
    while num_valid < ts.required_valid_init and gpc_round < ts.gpc_max_rounds:
        gpc_round += 1
        x_targets = normalizer.to_norm(asked_targets_real, clip=False)
        model, likelihood = train_gpc(x_targets, labels, steps=ts.gpc_steps, lr=ts.gpc_lr)
        candidates = torch.rand(ts.gpc_candidates, len(PROP_NAMES), **TKWARGS)
        proba = gpc_predict_proba(model, likelihood, candidates)
        feasible = candidates[proba >= ts.gpc_feasibility_threshold]
        if feasible.numel() == 0:
            logger.info("[Step 6] GPC found no feasible region in round %d; stopping", gpc_round)
            break
        to_add = ts.required_valid_init - num_valid
        new_targets_norm = greedy_maximin(x_targets, feasible, k=to_add)
        new_targets_real = normalizer.to_real(new_targets_norm)

        metadata, p_real, evaluated_mixtures, _ = run_targeting(
            new_targets_real, metadata, p_real, normalizer, onehot, evaluated_mixtures,
            fluids, config, radius=ts.radius_norm, budget_per_target=ts.target_budget,
        )
        asked_targets_real = torch.cat([asked_targets_real, new_targets_real], dim=0)
        normalizer.maybe_expand(p_real)
        flags, _, _ = success_mask(
            normalizer.to_norm(p_real, clip=False),
            normalizer.to_norm(asked_targets_real, clip=False), ts.radius_norm,
        )
        labels = torch.tensor([[1.0 if f else 0.0] for f in flags], **TKWARGS)
        num_valid = int(sum(flags))
        logger.info("[Step 6] valid targets after round %d: %d/%d",
                    gpc_round, num_valid, ts.required_valid_init)

    # ---- Step 7: SCBO on the earliest satisfied targets' realized mixtures ----
    _, success_rows, _ = success_mask(
        normalizer.to_norm(p_real, clip=False),
        normalizer.to_norm(asked_targets_real, clip=False), ts.radius_norm,
    )
    realized_ids = sorted({row for ok, row in zip(flags, success_rows) if ok and row >= 0})

    best_eta = -float("inf")
    best_name = "(none)"
    n_scbo = 0
    with RunWriter(subdir) as writer:
        order = 0
        for idx in realized_ids[: ts.required_valid_init]:
            j1, j2, x1 = metadata[idx]
            cand = realize_candidate("mixture", j1, j2, x1, fluids, onehot, config)
            if cand is None:
                continue
            eta, p_evap, p_cond = optimize_operating_conditions_robust(
                cand.wf, cand.pc, cand.ptriple, simulator, config.bo
            )
            order += 1
            n_scbo += 1
            writer.record("SCBO", order, "mixture", cand, eta, p_evap, p_cond)
            logger.info("[Step 7] %s: eta=%.5f", cand.name, eta)
            if eta > best_eta:
                best_eta, best_name = eta, cand.name

        # ---- Step 8: bounded cEI exploitation loop (GPC-feasibility weighted) ----
        best_eta, best_name, order = _exploitation_loop(
            writer, order, fluids, onehot, evaluated_mixtures, normalizer,
            asked_targets_real, labels, simulator, config, best_eta, best_name,
        )

    result = TwoStageResult(
        best_name=best_name,
        best_eta=best_eta if best_eta > -float("inf") else simulator.orc.infeasible_penalty,
        n_scbo=n_scbo,
        n_targets_satisfied=num_valid,
        outdir=subdir,
    )
    logger.info("Two-stage complete: best %s eta=%.5f -> %s",
                result.best_name, result.best_eta, subdir)
    return result


def _exploitation_loop(
    writer: RunWriter,
    order: int,
    fluids: List[str],
    onehot: torch.Tensor,
    evaluated_mixtures: Set[MixtureKey],
    normalizer: PropNormalizer,
    asked_targets_real: torch.Tensor,
    labels: torch.Tensor,
    simulator: ORCSimulator,
    config: AppConfig,
    best_eta: float,
    best_name: str,
    n_screen: int = 64,
) -> tuple[float, str, int]:
    """Step-8 exploitation: realize the most property-feasible novel mixture, SCBO it.

    A GP classifier trained on (normalized property, satisfied?) labels scores screened
    candidate mixtures by feasibility probability; the most feasible novel candidate is
    optimized with SCBO. The loop stops after ``system_budget`` proposals or
    ``failure_allowance`` consecutive infeasible evaluations.
    """
    ts = config.twostage
    t_dim = onehot.shape[0]

    if labels.sum() <= 0:
        logger.info("[Step 8] no satisfied targets to train the GPC; skipping exploitation")
        return best_eta, best_name, order

    x_targets = normalizer.to_norm(asked_targets_real, clip=False)
    model, likelihood = train_gpc(x_targets, labels, steps=ts.gpc_steps, lr=ts.gpc_lr)

    budget = ts.system_budget
    failures = 0
    while budget > 0 and failures < ts.failure_allowance:
        chosen = _propose_feasible_candidate(
            model, likelihood, fluids, onehot, evaluated_mixtures, normalizer, config, n_screen
        )
        if chosen is None:
            logger.info("[Step 8] no novel feasible candidate found; stopping")
            break

        eta, p_evap, p_cond = optimize_operating_conditions_robust(
            chosen.wf, chosen.pc, chosen.ptriple, simulator, config.bo
        )
        order += 1
        budget -= 1
        writer.record("STEP8", order, "mixture", chosen, eta, p_evap, p_cond)
        logger.info("[Step 8] %s: eta=%.5f", chosen.name, eta)
        if eta > 0:
            failures = 0
            if eta > best_eta:
                best_eta, best_name = eta, chosen.name
        else:
            failures += 1

    return best_eta, best_name, order


def _propose_feasible_candidate(
    model,
    likelihood,
    fluids: List[str],
    onehot: torch.Tensor,
    evaluated_mixtures: Set[MixtureKey],
    normalizer: PropNormalizer,
    config: AppConfig,
    n_screen: int,
) -> Optional[Candidate]:
    """Screen random one-hot candidates and return the most property-feasible novel one."""
    t_dim = onehot.shape[0]
    realized: List[Candidate] = []
    seen: Set[MixtureKey] = set()
    suggestions = torch.rand(n_screen, t_dim, **TKWARGS)
    for i in range(n_screen):
        j1, j2, x1 = snap_to_mixture(
            suggestions[i], onehot, evaluated_mixtures,
            composition_threshold=config.mixture.composition_threshold,
        )
        if (j1, j2, x1) in evaluated_mixtures or (j1, j2, x1) in seen:
            continue
        seen.add((j1, j2, x1))
        cand = realize_candidate("mixture", j1, j2, x1, fluids, onehot, config)
        if cand is not None:
            realized.append(cand)

    if not realized:
        return None

    props = torch.tensor([[c.tc, c.pc] for c in realized], **TKWARGS)
    proba = gpc_predict_proba(model, likelihood, normalizer.to_norm(props, clip=True))
    best = realized[int(torch.argmax(proba).item())]
    evaluated_mixtures.add((best.j1, best.j2, best.x1))
    return best
