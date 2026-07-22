"""Stage-1 property targeting for the two-stage pipeline.

The two-stage approach first searches in normalized ``(Tc, Pc)`` property space for fluids
near desirable property targets, then optimizes operating conditions (stage 2, via
:mod:`orc_bo.scbo`). This module provides the reusable, plotting-free building blocks:

* :class:`PropNormalizer` - adaptive normalization of property coordinates to ``[0, 1]``.
* :func:`success_mask` - which targets have been *reached* (a realized fluid within a radius).
* :func:`fit_property_gp` / :func:`run_targeting` - composite GP-driven search toward
  targets: a multi-output GP models chemical space -> properties, and a Monte-Carlo
  qLogEI of the negative L2 distance to each target drives proposals.
* :func:`fit_target_gp` - single-output GP helper (used for the Step-8 efficiency GP).
* :class:`ApproxGPClassifier` / :func:`train_gpc` / :func:`gpc_predict_proba` - a generic
  variational GP binary classifier. It is trained on **reachability** labels here (a target
  is reachable by some realizable mixture) and reused for **validity** labels in the Step-8
  cEI (a fluid has a feasible operating point). See :mod:`orc_bo.pipelines.twostage` for the
  reachability / validity / constraint-feasibility glossary.
* :func:`greedy_maximin` - maximin space-filling selection.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

if TYPE_CHECKING:  # avoid a runtime import cycle; used only for type hints
    from .pipelines.common import Fluid

import gpytorch
from botorch.acquisition import qLogExpectedImprovement
from botorch.acquisition.objective import GenericMCObjective
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.optim import optimize_acqf
from botorch.sampling.normal import SobolQMCNormalSampler
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch.quasirandom import SobolEngine

from .config import AppConfig
from .geometry import MixtureKey, format_mixture_name, simplex_row, snap_selection
from .logging_setup import get_logger
from .seeding import base_seed
from . import thermo

logger = get_logger(__name__)

TKWARGS = {"device": "cpu", "dtype": torch.double}


class PropNormalizer:
    """Adaptive min/max normalizer for property coordinates (e.g. Tc, Pc).

    Bounds are fit from observed points with an asymmetric margin (shrink the lower edge,
    expand the upper edge) and can grow as new points arrive via :meth:`maybe_expand`.
    """

    def __init__(
        self,
        prop_names: Sequence[str],
        lower_shrink: float = 0.5,
        upper_expand: float = 1.5,
    ) -> None:
        self.prop_names = list(prop_names)
        self.lower_shrink = lower_shrink
        self.upper_expand = upper_expand
        self.bounds: Optional[Dict[str, torch.Tensor]] = None

    def _bounds_from_observed(self, vmin: float, vmax: float) -> Tuple[float, float]:
        if vmin >= 0 and vmax >= 0:
            lb = 0.0 if vmin == 0 else vmin * self.lower_shrink
            ub = vmax * self.upper_expand
        elif vmax <= 0 and vmin <= 0:
            lb = vmin * self.upper_expand
            ub = 0.0 if vmax == 0 else vmax * self.lower_shrink
        else:
            lb, ub = vmin * self.upper_expand, vmax * self.upper_expand
        if not (np.isfinite(lb) and np.isfinite(ub)):
            lb, ub = vmin, vmax
        if lb == ub:
            span = max(1.0, abs(vmax - vmin))
            lb -= 0.01 * span
            ub += 0.01 * span
        if lb > ub:
            lb, ub = min(lb, ub), max(lb, ub)
        return lb, ub

    def fit_from_real_points(self, p_real: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Fit normalization bounds from a ``(n, d)`` tensor of real property points."""
        obs = p_real.detach().cpu().numpy()
        vmin, vmax = np.nanmin(obs, axis=0), np.nanmax(obs, axis=0)
        lbs, ubs = [], []
        for j in range(obs.shape[1]):
            lb, ub = self._bounds_from_observed(float(vmin[j]), float(vmax[j]))
            lbs.append(lb)
            ubs.append(ub)
        self.bounds = {
            "lower": torch.tensor(lbs, **TKWARGS),
            "upper": torch.tensor(ubs, **TKWARGS),
        }
        return self.bounds

    def maybe_expand(self, p_real: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], bool]:
        """Refit bounds; return ``(bounds, changed)``."""
        old = self.bounds
        new = self.fit_from_real_points(p_real)
        changed = old is None or not (
            torch.allclose(old["lower"], new["lower"]) and torch.allclose(old["upper"], new["upper"])
        )
        return self.bounds, changed

    def to_real(self, z_norm: torch.Tensor) -> torch.Tensor:
        """Map normalized coordinates back to real units."""
        lb, ub = self.bounds["lower"], self.bounds["upper"]
        return lb + z_norm * (ub - lb)

    def to_norm(self, p_real: torch.Tensor, clip: bool = True) -> torch.Tensor:
        """Map real property coordinates to normalized ``[0, 1]`` coordinates."""
        lb, ub = self.bounds["lower"], self.bounds["upper"]
        z = (p_real - lb) / (ub - lb)
        return z.clamp(0, 1) if clip else z


def success_mask(
    p_norm: torch.Tensor, targets_norm: torch.Tensor, radius: float
) -> Tuple[List[bool], List[int], List[float]]:
    """For each target, whether a realized point lies within ``radius`` (and which/how far)."""
    flags, rows, dists = [], [], []
    for k in range(targets_norm.shape[0]):
        d = torch.norm(p_norm - targets_norm[k].unsqueeze(0), dim=-1)
        dmin, arg = torch.min(d, dim=0)
        ok = float(dmin) <= radius
        flags.append(ok)
        rows.append(int(arg.item()) if ok else -1)
        dists.append(float(dmin.item()))
    return flags, rows, dists


def fit_target_gp(x: torch.Tensor, y: torch.Tensor) -> SingleTaskGP:
    """Fit a standardized single-output SingleTaskGP (the Step-8 efficiency regressor GPR2)."""
    model = SingleTaskGP(x, y, outcome_transform=Standardize(m=1))
    fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
    return model


def fit_property_gp(x: torch.Tensor, p_norm: torch.Tensor) -> SingleTaskGP:
    """Fit the multi-output targeting surrogate GPR1: chemical space -> normalized properties.

    This is the composite-BO formulation: rather than modelling the scalar distance to one
    target, a single multi-output GP models the property vector itself, and each target's
    acquisition applies its own objective ``g_k(p) = -||p - p_k*||_2`` to posterior property
    samples. One model per round serves every active target.
    """
    y = p_norm + 1e-8 * torch.randn_like(p_norm)  # jitter for numerical stability
    model = SingleTaskGP(x, y, outcome_transform=Standardize(m=y.shape[-1]))
    fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
    return model


class ApproxGPClassifier(gpytorch.models.ApproximateGP):
    """Generic variational sparse GP binary classifier.

    Trained on reachability labels (property-space) in stage 1, or validity labels
    (operability) in the Step-8 cEI. It models a probability, not a specific notion of
    "feasibility" — the caller decides what the labels mean.

    ``prior_mean`` sets the latent constant mean. With the probit link, the far-from-data
    class probability is ``Phi(prior_mean / sqrt(1 + s^2))``: 0.5 for the neutral default,
    ~0.24 at -1.0. A negative prior makes unlabeled regions "negative until demonstrated"
    (pessimistic), which suits validity/operability. When nonzero, the constant is FROZEN:
    labels are typically majority-positive, so a learnable constant would drift back up
    during training and silently erase the requested pessimism.
    """

    def __init__(self, inducing_points: torch.Tensor, prior_mean: float = 0.0) -> None:
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(0)
        )
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        if prior_mean != 0.0:
            self.mean_module.initialize(constant=prior_mean)
            self.mean_module.constant.requires_grad_(False)
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=inducing_points.shape[-1])
        )

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x)
        )


def train_gpc(
    x_cls: torch.Tensor,
    y_cls: torch.Tensor,
    steps: int = 200,
    lr: float = 0.1,
    m_inducing: int = 128,
    prior_mean: float = 0.0,
) -> Tuple[ApproxGPClassifier, gpytorch.likelihoods.BernoulliLikelihood]:
    """Train the variational GP binary classifier on ``y_cls`` labels.

    Labels are caller-defined: reachability (target reached) in stage 1, or validity
    (feasible operating point exists) in the Step-8 cEI. ``prior_mean`` < 0 trains a
    pessimistic classifier whose far-from-data probability sits below 0.5 (see
    :class:`ApproxGPClassifier`); use it for validity, not reachability.
    """
    y_flat = y_cls.reshape(-1)
    n_inducing = min(m_inducing, max(32, x_cls.shape[0] // 2))
    inducing = SobolEngine(dimension=x_cls.shape[-1], scramble=True, seed=base_seed()).draw(
        n_inducing
    ).to(**TKWARGS)

    model = ApproxGPClassifier(inducing_points=inducing, prior_mean=prior_mean).to(**TKWARGS)
    likelihood = gpytorch.likelihoods.BernoulliLikelihood().to(**TKWARGS)
    model.train()
    likelihood.train()
    optimizer = torch.optim.Adam(
        [{"params": model.parameters()}, {"params": likelihood.parameters()}], lr=lr
    )
    elbo = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=x_cls.shape[0])
    for i in range(steps):
        optimizer.zero_grad()
        loss = -elbo(model(x_cls), y_flat)
        loss.backward()
        optimizer.step()
        if (i + 1) % 100 == 0:
            logger.debug("GPC step %d/%d loss=%.4f", i + 1, steps, float(loss.item()))
    model.eval()
    likelihood.eval()
    return model, likelihood


@torch.no_grad()
def gpc_predict_proba(
    model: ApproxGPClassifier,
    likelihood: gpytorch.likelihoods.BernoulliLikelihood,
    x: torch.Tensor,
) -> torch.Tensor:
    """Return the predicted class probability for each row of ``x`` (reachability or validity)."""
    with gpytorch.settings.fast_pred_var():
        return likelihood(model(x)).mean


def greedy_maximin(
    selected: Optional[torch.Tensor], candidates: torch.Tensor, k: int
) -> torch.Tensor:
    """Greedily pick ``k`` candidates maximizing the minimum distance to the chosen set."""
    if k <= 0 or candidates.numel() == 0:
        return candidates[:0]

    chosen_idx: List[int] = []
    if selected is None or selected.numel() == 0:
        centroid = candidates.mean(dim=0, keepdim=True)
        chosen_idx.append(int(torch.argmax(torch.norm(candidates - centroid, dim=1))))
    else:
        min_d = torch.cdist(candidates, selected).min(dim=1).values
        chosen_idx.append(int(torch.argmax(min_d)))

    while len(chosen_idx) < min(k, candidates.shape[0]):
        sel = candidates[chosen_idx]
        pool = sel if (selected is None or selected.numel() == 0) else torch.cat([selected, sel], dim=0)
        min_d = torch.cdist(candidates, pool).min(dim=1).values
        min_d[chosen_idx] = -1.0
        chosen_idx.append(int(torch.argmax(min_d)))

    return candidates[chosen_idx]


def run_targeting(
    targets_real: torch.Tensor,
    metadata: List[MixtureKey],
    p_real: torch.Tensor,
    normalizer: PropNormalizer,
    onehot: torch.Tensor,
    evaluated_mixtures: set[MixtureKey],
    fluids: List["Fluid"],
    config: AppConfig,
    radius: float = 0.05,
    budget_per_target: int = 2,
    mode: str = "mixture",
) -> Tuple[List[MixtureKey], torch.Tensor, set[MixtureKey], List[dict]]:
    """Drive fluid proposals toward each property target via composite MC qLogEI.

    Each round fits one multi-output GP (:func:`fit_property_gp`) of chemical space ->
    normalized properties. For every still-unsatisfied target, the qLogEI of the composite
    objective ``-||p - p*||_2`` (estimated by Monte-Carlo over posterior property samples)
    is maximized with gradient-based ``optimize_acqf`` over the relaxed one-hot space; the
    optimum is snapped (a pure vertex when ``mode == "pure"``, otherwise a binary mixture
    edge), screened for its properties, and appended. Returns updated metadata, the property
    tensor, the evaluated set, and per-target success records.
    """
    if targets_real.numel() == 0:
        normalizer.maybe_expand(p_real)
        return metadata, p_real, evaluated_mixtures, []

    normalizer.maybe_expand(p_real)
    targets_norm = normalizer.to_norm(targets_real, clip=False)
    pre_flags, pre_rows, pre_dmin = success_mask(
        normalizer.to_norm(p_real, clip=False), targets_norm, radius
    )

    n_targets = targets_norm.shape[0]
    tries_left = [budget_per_target] * n_targets
    done = list(pre_flags)
    hit_row = [r if r >= 0 else None for r in pre_rows]
    hit_dist = list(pre_dmin)

    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([config.bo.mc_samples]))
    t_dim = onehot.shape[1]
    bounds = torch.stack([torch.zeros(t_dim, **TKWARGS), torch.ones(t_dim, **TKWARGS)])

    def x_rows() -> torch.Tensor: # translates (j1, j2, x) tuples into D-dimensional vectors
        rows = [simplex_row(onehot, j1, j2, x1) for (j1, j2, x1) in metadata]
        return torch.stack(rows, dim=0)

    x_train = x_rows()
    round_id = 0
    while any((not done[k]) and tries_left[k] > 0 for k in range(n_targets)):
        round_id += 1
        active = [k for k in range(n_targets) if not done[k] and tries_left[k] > 0]
        normalizer.maybe_expand(p_real)
        p_norm = normalizer.to_norm(p_real, clip=False)
        targets_norm = normalizer.to_norm(targets_real, clip=False)

        # Composite acquisition: GPR1 (chemical space -> properties) is fit once per round;
        # each target k applies its own MC objective g_k(p) = -||p - p_k*||_2 to posterior
        # property samples inside qLogEI, maximized with gradient-based optimize_acqf. After
        # snapping, pure mode degrades toward (near-random) novel-vertex selection - the
        # one-hot equidistance leaves little structure - but the composite form is what makes
        # the continuous mixture space searchable, so both modes share it.
        gpr1 = fit_property_gp(x_train, p_norm)  # GPR1: targeting surrogate (chemical -> props)
        proposed: Dict[int, MixtureKey] = {}
        claimed: set[MixtureKey] = set()
        for k in active:
            target_k = targets_norm[k]
            best_f = -torch.norm(p_norm - target_k.unsqueeze(0), dim=-1).min()
            objective = GenericMCObjective(
                lambda samples, X=None, t=target_k: -torch.norm(samples - t, dim=-1) # convert property samples to scalar whydistances
            )
            acqf = qLogExpectedImprovement(
                model=gpr1, best_f=best_f, sampler=sampler, objective=objective # Expected Improvement with MC sampler - single scalar
            )
            x_cand, _ = optimize_acqf(  # then gradient ascent via backward pass to find vector yielding highest EI (reparameterization trick)
                acq_function=acqf, bounds=bounds, q=1,
                num_restarts=min(config.bo.num_restarts, 2 * t_dim),
                raw_samples=config.bo.raw_samples,
                options={"batch_limit": 5, "maxiter": 200},
            )
            selection = snap_selection(
                mode, x_cand.squeeze(0), onehot, evaluated_mixtures,
                composition_threshold=config.mixture.composition_threshold,
                min_frac=config.mixture.min_mole_frac,
            )
            if selection not in claimed: # so we don't propose same chemical mixture twice, stored in claimed chemicals
                proposed[k] = selection
                claimed.add(selection)

        if not proposed:
            for k in active:
                tries_left[k] = 0
            continue

        for k, (j1, j2, x1) in proposed.items():
            evaluated_mixtures.add((j1, j2, x1))
            f1 = fluids[j1]
            f2 = fluids[j2] if j2 is not None else None
            try:
                tc, pc = thermo.critical_properties(
                    f1.name, f2.name if f2 else None, x1, config.thermo,
                    refprop1=f1.refprop, refprop2=f2.refprop if f2 else None,
                )
            except thermo.ThermoError as exc:
                # No mixing-rule fallback: an unevaluable pair is unrealizable and skipped.
                # The attempt still cost one lab-scale screen (already counted).
                logger.warning("Targeting screen failed for %s: %s",
                               format_mixture_name(f1.name, f2.name if f2 else None, x1), exc)
                continue
            if not np.isfinite(tc) or not np.isfinite(pc):
                logger.warning("Targeting: invalid properties for %s",
                               format_mixture_name(f1.name, f2.name if f2 else None, x1))
                continue
            metadata.append((j1, j2, x1))
            p_real = torch.cat([p_real, torch.tensor([[tc, pc]], **TKWARGS)], dim=0) # [Tc, Pc] vectors
            x_train = torch.cat([x_train, simplex_row(onehot, j1, j2, x1).unsqueeze(0)], dim=0) # chemical composition vectors
            tries_left[k] -= 1

        normalizer.maybe_expand(p_real)
        p_norm = normalizer.to_norm(p_real, clip=False)
        targets_norm = normalizer.to_norm(targets_real, clip=False)
        for k in range(n_targets):
            if done[k]:
                continue
            dmin, arg = torch.min(torch.norm(p_norm - targets_norm[k].unsqueeze(0), dim=-1), dim=0)
            if float(dmin) <= radius: # ball around target p*
                done[k], hit_row[k], hit_dist[k] = True, int(arg.item()), float(dmin.item())

    results = [
        {"success": done[k], "row": hit_row[k] if done[k] else None,
         "dist": hit_dist[k] if done[k] else None}
        for k in range(n_targets)
    ]
    return metadata, p_real, evaluated_mixtures, results
