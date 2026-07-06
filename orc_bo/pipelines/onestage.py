"""One-stage Bayesian optimization pipeline.

The pipeline optimizes directly in the relaxed one-hot fluid space:

1. Latin-hypercube initial selections, each snapped to a fluid (pure) or mixture edge.
2. For each realization, SCBO optimizes the operating pressures and records efficiency.
3. A qEI BO loop proposes new one-hot points; each is snapped, realized, and evaluated.

It supports ``mode="pure"`` (snap to vertices) and ``mode="mixture"`` (snap to edges).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Set

import torch
from scipy.stats import qmc

from botorch.acquisition import qLogExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.optim import optimize_acqf
from botorch.sampling.normal import SobolQMCNormalSampler
from gpytorch.mlls import ExactMarginalLogLikelihood

from ..config import AppConfig
from ..geometry import (
    MixtureKey,
    mixture_key_canonical,
    snap_to_mixture,
    snap_to_vertex,
)
from ..logging_setup import get_logger
from ..orc_model import ORCSimulator
from ..scbo import optimize_operating_conditions_robust
from ..seeding import base_seed
from .common import (
    PHASE_INIT,
    PHASE_OPT,
    Candidate,
    RunWriter,
    TKWARGS,
    format_run_header,
    load_fluids,
    realize_candidate,
)

logger = get_logger(__name__)


@dataclass
class OneStageResult:
    """Summary of a one-stage run."""

    mode: str
    best_name: str
    best_eta: float
    n_evaluations: int
    outdir: Path
    sequence: List[str]


def _select(
    mode: str,
    x_suggest: torch.Tensor,
    onehot: torch.Tensor,
    evaluated_mixtures: Set[MixtureKey],
    composition_threshold: float,
) -> tuple[int, Optional[int], float]:
    """Snap a continuous suggestion to a (j1, j2, x1) selection for the given mode."""
    if mode == "pure":
        return snap_to_vertex(x_suggest, onehot), None, 1.0
    return snap_to_mixture(x_suggest, onehot, evaluated_mixtures,
                           composition_threshold=composition_threshold)


def run_onestage(
    csv_path: Path,
    mode: str = "mixture",
    n_init: int = 3,
    scbo_budget: int = 4,
    outdir: Path = Path("runs/onestage"),
    config: Optional[AppConfig] = None,
) -> OneStageResult:
    """Run the one-stage pipeline and return a result summary.

    Parameters
    ----------
    csv_path:
        Dataset CSV listing candidate fluids.
    mode:
        ``"pure"`` or ``"mixture"``.
    n_init:
        Number of Latin-hypercube initial selections.
    scbo_budget:
        Number of qEI BO-loop iterations after initialization.
    outdir:
        Output directory; a ``seed_XXX`` subdirectory is created within it.
    config:
        Application configuration; defaults to :class:`AppConfig` defaults.

    Returns
    -------
    OneStageResult
        Best fluid/mixture, its efficiency, and the evaluation sequence.
    """
    config = config or AppConfig()
    threshold = config.mixture.composition_threshold
    seed = base_seed()
    subdir = Path(outdir) / f"seed_{seed:03d}"

    fluids = load_fluids(Path(csv_path))
    t_dim = len(fluids)
    onehot = torch.eye(t_dim, **TKWARGS)
    simulator = ORCSimulator(orc=config.orc, backend=config.thermo.backend)
    logger.info("One-stage (%s): %d fluids, n_init=%d, budget=%d, seed=%d",
                mode, t_dim, n_init, scbo_budget, seed)

    evaluated_mixtures: Set[MixtureKey] = set()
    x_train_rows: List[torch.Tensor] = []
    y_train_vals: List[float] = []
    sequence: List[str] = []

    def evaluate_and_record(writer: RunWriter, phase: str, order: int, cand: Candidate) -> None:
        eta, p_evap, p_cond = optimize_operating_conditions_robust(
            cand.wf, cand.pc, cand.ptriple, simulator, config.bo
        )
        logger.info("[%s %d] %s: eta=%.5f", phase, order, cand.name, eta)
        writer.record(phase, order, mode, cand, eta, p_evap, p_cond)
        x_train_rows.append(cand.x_onehot)
        y_train_vals.append(eta)
        sequence.append(cand.name)

    header = format_run_header(config, stage="onestage", mode=mode, seed=seed,
                               n_init=n_init, scbo_budget=scbo_budget)
    with RunWriter(subdir, header=header) as writer:
        # ---- Initialization: Latin-hypercube selections ----
        lhs = torch.tensor(qmc.LatinHypercube(d=t_dim, seed=seed).random(n_init), **TKWARGS)
        order = 0
        for k in range(lhs.shape[0]):
            j1, j2, x1 = _select(mode, lhs[k], onehot, evaluated_mixtures, threshold)
            evaluated_mixtures.add(mixture_key_canonical(j1, j2, x1))
            cand = realize_candidate(mode, j1, j2, x1, fluids, onehot, config)
            if cand is None:
                continue
            order += 1
            evaluate_and_record(writer, PHASE_INIT, order, cand)

        if not y_train_vals:
            raise RuntimeError("No initial candidates could be realized")

        x_train = torch.stack(x_train_rows, dim=0)
        y_train = torch.tensor(y_train_vals, **TKWARGS).reshape(-1, 1)

        # ---- qEI BO loop in one-hot space ----
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([config.bo.mc_samples]))
        bounds = torch.stack([torch.zeros(t_dim, **TKWARGS), torch.ones(t_dim, **TKWARGS)])
        for iteration in range(1, max(0, scbo_budget) + 1):
            gp = SingleTaskGP(x_train, y_train, outcome_transform=Standardize(m=1))
            fit_gpytorch_mll(ExactMarginalLogLikelihood(gp.likelihood, gp))
            acqf = qLogExpectedImprovement(model=gp, best_f=y_train.max(), sampler=sampler)
            x_cand, _ = optimize_acqf(
                acq_function=acqf, bounds=bounds, q=1,
                num_restarts=min(config.bo.num_restarts, 2 * t_dim),
                raw_samples=config.bo.raw_samples,
                options={"batch_limit": 5, "maxiter": 200},
            )
            x_next = x_cand.squeeze(0)
            j1, j2, x1 = _select(mode, x_next, onehot, evaluated_mixtures, threshold)
            evaluated_mixtures.add(mixture_key_canonical(j1, j2, x1))
            cand = realize_candidate(mode, j1, j2, x1, fluids, onehot, config)
            if cand is None:
                continue
            order += 1
            evaluate_and_record(writer, PHASE_OPT, iteration, cand)
            x_train = torch.cat([x_train, cand.x_onehot.unsqueeze(0)], dim=0)
            y_train = torch.cat([y_train, torch.tensor([[y_train_vals[-1]]], **TKWARGS)], dim=0)

    best_idx = int(torch.tensor(y_train_vals).argmax().item())
    result = OneStageResult(
        mode=mode,
        best_name=sequence[best_idx],
        best_eta=y_train_vals[best_idx],
        n_evaluations=len(sequence),
        outdir=subdir,
        sequence=sequence,
    )
    logger.info("One-stage complete: best %s eta=%.5f (%d evals) -> %s",
                result.best_name, result.best_eta, result.n_evaluations, subdir)
    logger.info("Backend coverage: %s", simulator.backend_failure_report())
    return result
