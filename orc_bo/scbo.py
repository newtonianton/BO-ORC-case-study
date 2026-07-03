"""SCBO (TuRBO) trust-region constrained optimization of ORC operating conditions.

For a fixed working fluid, the inner optimization searches the two operating pressures
``(p_evap, p_cond)`` to maximize efficiency subject to the pressure-ordering and pinch
constraints. It uses a single-trust-region TuRBO loop with constrained Thompson sampling
(``ConstrainedMaxPosteriorSampling``).

The public entry points are :func:`optimize_operating_conditions` and the retry wrapper
:func:`optimize_operating_conditions_robust`, both returning
``(eta, p_evap_bar, p_cond_bar)``.

Terminology: "feasible" in this module always means **constraint feasibility of an operating
point** - a specific ``(p_evap, p_cond)`` satisfies the pressure-ordering and pinch
constraints. This is distinct from the pipeline-level notions of *reachability* (property
space) and *validity* (whether any feasible operating point exists at all); a fluid is
*valid* exactly when this optimizer returns a constraint-feasible point (positive efficiency).
See :mod:`orc_bo.pipelines.twostage` for the full glossary.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Tuple

import torch
from torch import Tensor
from torch.quasirandom import SobolEngine

from botorch.exceptions import ModelFittingError
from botorch.fit import fit_gpytorch_mll
from botorch.generation.sampling import ConstrainedMaxPosteriorSampling
from botorch.models import ModelListGP, SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.utils.transforms import unnormalize
from gpytorch.constraints import Interval
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood

from .config import BOConfig
from .logging_setup import get_logger
from .orc_model import ORCSimulator
from .seeding import derive_seed

logger = get_logger(__name__)

TKWARGS = {"device": "cpu", "dtype": torch.double}

# Optimization result: (eta, p_evap_bar, p_cond_bar).
OperatingPoint = Tuple[float, float, float]


@dataclass
class ScboState:
    """Single trust-region state for the SCBO loop (TuRBO-style)."""

    dim: int
    batch_size: int
    length: float = 0.8
    length_min: float = 0.5 ** 6
    length_max: float = 1.6
    failure_counter: int = 0
    failure_tolerance: int = field(default=0)
    success_counter: int = 0
    success_tolerance: int = 3
    best_value: float = -float("inf")
    best_constraint_values: Tensor = field(
        default_factory=lambda: torch.ones(2, **TKWARGS) * torch.inf
    )
    restart_triggered: bool = False

    def __post_init__(self) -> None:
        self.failure_tolerance = math.ceil(
            max(4.0 / self.batch_size, self.dim / self.batch_size)
        )


def _update_trust_region(state: ScboState) -> ScboState:
    """Expand/shrink the trust region based on success/failure streaks."""
    if state.success_counter == state.success_tolerance:
        state.length = min(2.0 * state.length, state.length_max)
        state.success_counter = 0
    elif state.failure_counter == state.failure_tolerance:
        state.length /= 2.0
        state.failure_counter = 0
    if state.length < state.length_min:
        state.restart_triggered = True
    return state


def best_feasible_index(y: Tensor, c: Tensor) -> Tensor:
    """Index of the best point: max objective among constraint-feasible operating points
    (all constraints ``<= 0``), else the least-violating point."""
    feasible = (c <= 0).all(dim=-1)
    if feasible.any():
        scored = y.clone()
        scored[~feasible] = -float("inf")
        return scored.argmax()
    return c.clamp(min=0).sum(dim=-1).argmin()


def _update_state(state: ScboState, y_new: Tensor, c_new: Tensor) -> ScboState:
    """Update incumbent and counters with a new batch of observations."""
    idx = best_feasible_index(y_new, c_new)
    y, c = y_new[idx], c_new[idx]
    if (c <= 0).all():
        threshold = state.best_value + 1e-3 * abs(state.best_value)
        improved = y > threshold or (state.best_constraint_values > 0).any()
    else:
        v_next = c.clamp(min=0).sum(dim=-1)
        v_incumbent = state.best_constraint_values.clamp(min=0).sum(dim=-1)
        improved = bool(v_next < v_incumbent)

    if improved:
        state.success_counter += 1
        state.failure_counter = 0
        state.best_value = float(y.item())
        state.best_constraint_values = c
    else:
        state.success_counter = 0
        state.failure_counter += 1
    return _update_trust_region(state)


def _fit_gp(
    x: Tensor, y: Tensor, dim: int, noise: float, max_noise: float, standardize: bool
) -> SingleTaskGP:
    """Fit a SingleTaskGP, escalating the noise floor on fitting failure."""
    covar = ScaleKernel(
        MaternKernel(nu=2.5, ard_num_dims=dim, lengthscale_constraint=Interval(0.005, 4.0))
    )
    try:
        outcome_transform = Standardize(m=1) if standardize else None
        model = SingleTaskGP(
            x, y, torch.full_like(y, noise), covar_module=covar, outcome_transform=outcome_transform
        )
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)
    except ModelFittingError:
        if noise >= max_noise:
            raise
        logger.debug("GP fit failed at noise=%g; retrying at %g", noise, noise * 10)
        return _fit_gp(x, y, dim, noise * 10, max_noise, standardize)
    return model


def optimize_operating_conditions(
    wf: str,
    pcrit: float,
    ptriple: float,
    simulator: ORCSimulator,
    bo: BOConfig | None = None,
    seed_offset: int = 0,
) -> OperatingPoint:
    """Optimize ``(p_evap, p_cond)`` for a fixed working fluid via SCBO.

    Parameters
    ----------
    wf:
        Working-fluid string passed to ``simulator``.
    pcrit, ptriple:
        Critical and triple-point pressures [Pa], used to bound the pressure search.
    simulator:
        ORC simulator providing ``simulate(wf, p_evap_bar, p_cond_bar)``.
    bo:
        BO hyperparameters; defaults to :class:`BOConfig` defaults.
    seed_offset:
        Added to the derived seeds so retries explore different initializations.

    Returns
    -------
    OperatingPoint
        ``(eta, p_evap_bar, p_cond_bar)``. Returns the infeasible penalty with
        ``(-1, -1)`` pressures when no feasible point is found.
    """
    bo = bo or BOConfig()
    dim = bo.scbo_dim
    penalty = simulator.orc.infeasible_penalty

    pcrit_bar = 0.99 * pcrit / 1e5
    ptr_bar = 1.01 * ptriple / 1e5
    bounds = torch.tensor([[ptr_bar, ptr_bar], [pcrit_bar, pcrit_bar]], **TKWARGS)

    def evaluate(x_unit: Tensor) -> Tuple[float, float, float]:
        real = unnormalize(x_unit, bounds)
        p_evap_bar, p_cond_bar = float(real[0]), float(real[1])
        result = simulator.simulate(wf, p_evap_bar, p_cond_bar)
        # Constraints (feasible when <= 0): pressure ordering, sink pinch, source pinch.
        return result.eta, result.sink_pinch, result.source_pinch, p_cond_bar - p_evap_bar

    def observe(xs: Tensor) -> Tuple[Tensor, Tensor]:
        objectives, c1, c2, c3 = [], [], [], []
        for x in xs:
            eta, snk, src, dp = evaluate(x)
            objectives.append(eta)
            c1.append(dp)
            c2.append(snk)
            c3.append(src)
        y = torch.tensor(objectives, **TKWARGS).unsqueeze(-1)
        c = torch.tensor(list(zip(c1, c2, c3)), **TKWARGS)
        return y, c

    n_cand = min(bo.n_candidates, 200 * dim)
    sobol_init = SobolEngine(dim, scramble=True, seed=derive_seed(7001) + seed_offset)
    x = sobol_init.draw(bo.scbo_n_init).to(**TKWARGS)
    y, c = observe(x)

    state = ScboState(dim, batch_size=bo.batch_size, length=bo.tr_length_init,
                      length_min=bo.tr_length_min, length_max=bo.tr_length_max,
                      success_tolerance=bo.success_tolerance)
    sobol_batch = SobolEngine(dim, scramble=True, seed=derive_seed(7002) + seed_offset)

    while not state.restart_triggered:
        model = _fit_gp(x, y, dim, bo.gp_noise, bo.gp_max_noise, standardize=True)
        constraint_models = [
            _fit_gp(x, c[:, j:j + 1], dim, bo.gp_noise, bo.gp_max_noise, standardize=False)
            for j in range(c.shape[-1])
        ]

        idx = best_feasible_index(y, c)
        x0 = x[idx].clone()
        half = state.length / 2
        lb = torch.clamp(x0 - half, 0, 1)
        ub = torch.clamp(x0 + half, 0, 1)
        candidates = lb + (ub - lb) * sobol_batch.draw(n_cand).to(**TKWARGS)

        sampler = ConstrainedMaxPosteriorSampling(
            model=model, constraint_model=ModelListGP(*constraint_models), replacement=False
        )
        with torch.no_grad():
            x_next = sampler(candidates, num_samples=bo.batch_size)

        y_next, c_next = observe(x_next)
        state = _update_state(state, y_next, c_next)
        x = torch.cat([x, x_next], dim=0)
        y = torch.cat([y, y_next], dim=0)
        c = torch.cat([c, c_next], dim=0)

    if (state.best_constraint_values <= 0).all():
        idx = torch.where(y == state.best_value)
        eta = float(y[idx].item())
        real = unnormalize(x[idx[0]], bounds)
        return eta, float(real[:, 0].item()), float(real[:, 1].item())
    return penalty, -1.0, -1.0


def optimize_operating_conditions_robust(
    wf: str,
    pcrit: float,
    ptriple: float,
    simulator: ORCSimulator,
    bo: BOConfig | None = None,
    max_retries: int | None = None,
) -> OperatingPoint:
    """Retry :func:`optimize_operating_conditions` with fresh seeds until feasible.

    Handles cases where an unlucky initialization fails to find a feasible operating
    point. Returns the first result with positive efficiency, or the penalty if all
    attempts are exhausted.

    ``max_retries`` defaults to ``bo.scbo_max_retries``; pass an explicit value to
    override the configured number of attempts.
    """
    bo = bo or BOConfig()
    if max_retries is None:
        max_retries = bo.scbo_max_retries

    # Skip fluids the backend cannot even build (e.g. a mixture pair with no interaction
    # parameters): this avoids wasting every retry and records it as a backend failure
    # rather than a merely-infeasible fluid.
    if not simulator.can_evaluate(wf):
        logger.debug("Backend cannot build %s; skipping SCBO", wf)
        return simulator.orc.infeasible_penalty, -1.0, -1.0

    for attempt in range(max_retries):
        try:
            eta, p_evap, p_cond = optimize_operating_conditions(
                wf, pcrit, ptriple, simulator, bo, seed_offset=attempt
            )
            if eta > 0:
                if attempt > 0:
                    logger.info("SCBO retry %d succeeded (eta=%.5f)", attempt + 1, eta)
                return eta, p_evap, p_cond
            logger.debug("SCBO attempt %d infeasible (eta=%.5f)", attempt + 1, eta)
        except Exception as exc:  # noqa: BLE001 - SCBO/GP failures are retried
            logger.warning("SCBO attempt %d raised %s; retrying", attempt + 1, exc)

    logger.warning("SCBO exhausted %d attempts without a valid point", max_retries)
    penalty = (simulator.orc.infeasible_penalty if simulator else -0.05)
    return penalty, -1.0, -1.0
