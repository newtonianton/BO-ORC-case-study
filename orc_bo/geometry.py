"""Geometric projection of continuous suggestions onto fluids and binary mixtures.

The optimizer proposes continuous points in a relaxed one-hot space ``[0, 1]^t`` where
each axis corresponds to a candidate fluid. This module maps such a point to a concrete
working fluid:

* :func:`snap_to_vertex` - pure-fluid mode: snap to the nearest one-hot vertex.
* :func:`snap_to_mixture` - mixture mode: snap to a point on a vertex-to-vertex edge,
  yielding a binary mixture ``(j1, j2, x1)`` with ``min_frac <= x1 <= 1 - min_frac``.

Mixture compositions are discretized to a grid (``composition_threshold``) so that
near-identical mixtures are not re-evaluated, and an ``evaluated_mixtures`` set lets the
caller prefer novel compositions.
"""
from __future__ import annotations

from typing import Optional, Set, Tuple, Union

import numpy as np
import torch

from .config import MixtureConfig
from .logging_setup import get_logger

logger = get_logger(__name__)

# Module-level defaults sourced from the configuration dataclass.
_MIX_DEFAULTS = MixtureConfig()
MIN_MOLE_FRAC: float = _MIX_DEFAULTS.min_mole_frac
MAX_MOLE_FRAC: float = _MIX_DEFAULTS.max_mole_frac
DEFAULT_COMPOSITION_THRESHOLD: float = _MIX_DEFAULTS.composition_threshold

# Canonical mixture key: (component_low, component_high, x_of_low) or pure (j, None, 1.0).
MixtureKey = Tuple[int, Optional[int], float]


def snap_composition_to_threshold(
    x1: Union[float, torch.Tensor],
    threshold: float = DEFAULT_COMPOSITION_THRESHOLD,
    min_frac: float = MIN_MOLE_FRAC,
    max_frac: float = MAX_MOLE_FRAC,
) -> float:
    """Round a mole fraction to the nearest multiple of ``threshold`` and clip to bounds.

    Discretizing compositions avoids evaluating nearly-identical mixtures (e.g. 0.501 vs
    0.502 when ``threshold`` is 0.01).

    Parameters
    ----------
    x1:
        Mole fraction to snap (Python float or scalar tensor).
    threshold:
        Grid spacing for discretization.
    min_frac, max_frac:
        Composition bounds applied after snapping.

    Returns
    -------
    float
        The snapped, clipped mole fraction.
    """
    x1_val = float(x1.item()) if isinstance(x1, torch.Tensor) else float(x1)
    snapped = round(x1_val / threshold) * threshold
    return float(np.clip(snapped, min_frac, max_frac))


def is_composition_novel(
    j1: int,
    j2: Optional[int],
    x1: float,
    evaluated_mixtures: Set[MixtureKey],
    composition_threshold: float = DEFAULT_COMPOSITION_THRESHOLD,
) -> bool:
    """Return whether a candidate mixture differs enough from those already evaluated.

    Multiple compositions of the same fluid pair (edge) are allowed because efficiency
    varies with composition; a candidate within ``composition_threshold`` of a previous
    evaluation on the same edge (accounting for component order) is treated as a
    duplicate. Pure fluids (``j2 is None``) are not supported in mixture mode and return
    ``False``.

    Parameters
    ----------
    j1, j2:
        Component indices of the candidate (``j2`` may be ``None`` for legacy pure keys).
    x1:
        Mole fraction of ``j1``.
    evaluated_mixtures:
        Set of ``(j1, j2, x1)`` tuples already evaluated.
    composition_threshold:
        Minimum difference in ``x1`` to consider the candidate novel.

    Returns
    -------
    bool
        ``True`` if the candidate should be evaluated, ``False`` if it is a near-duplicate.
    """
    if j2 is None:
        return False

    edge_canonical = tuple(sorted((j1, j2)))
    for e1, e2, x_eval in evaluated_mixtures:
        if e2 is None:
            continue
        if tuple(sorted((e1, e2))) != edge_canonical:
            continue
        # Same edge: account for reversed component order (x1 <-> 1 - x1).
        if (j1, j2) == (e1, e2):
            x_diff = abs(x1 - x_eval)
        else:
            x_diff = abs(x1 - (1.0 - x_eval))
        if x_diff < composition_threshold:
            return False
    return True


def snap_to_vertex(x_suggestion: torch.Tensor, onehot_tensor: torch.Tensor) -> int:
    """Pure-fluid mode: snap a continuous suggestion to the nearest one-hot vertex.

    For the standard one-hot basis the nearest vertex is the largest coordinate; ties
    resolve to the smallest index.

    Parameters
    ----------
    x_suggestion:
        ``(t,)`` continuous point in relaxed one-hot space.
    onehot_tensor:
        ``(t, t)`` one-hot basis vectors.

    Returns
    -------
    int
        Index of the selected pure fluid.
    """
    distances = torch.norm(onehot_tensor - x_suggestion.unsqueeze(0), dim=1)
    return int(torch.argmin(distances).item())


def snap_to_mixture(
    x_suggestion: torch.Tensor,
    onehot_tensor: torch.Tensor,
    evaluated_mixtures: Set[MixtureKey],
    min_frac: float = MIN_MOLE_FRAC,
    composition_threshold: float = DEFAULT_COMPOSITION_THRESHOLD,
) -> MixtureKey:
    """Mixture mode: map a continuous suggestion to a binary mixture on an edge.

    Algorithm:

    1. Find the nearest vertex -> component ``j1``.
    2. Compute the direction from ``j1`` to the suggestion.
    3. Choose ``j2`` whose edge direction has the highest cosine similarity.
    4. Project onto the ``j1``-``j2`` edge to get ``x1``; snap to the grid and clip.
    5. Prefer edges/compositions not yet evaluated (novelty), then best direction.

    Pure fluids are never returned: the result always has ``j2 is not None`` and
    ``min_frac <= x1 <= 1 - min_frac``.

    Parameters
    ----------
    x_suggestion:
        ``(t,)`` continuous point in relaxed one-hot space.
    onehot_tensor:
        ``(t, t)`` one-hot basis vectors.
    evaluated_mixtures:
        Set of ``(j1, j2, x1)`` tuples already evaluated (used for novelty).
    min_frac:
        Minimum mole fraction of the minority component.
    composition_threshold:
        Grid spacing for composition snapping and the novelty threshold.

    Returns
    -------
    MixtureKey
        ``(j1, j2, x1)`` with ``j2`` never ``None``.

    Raises
    ------
    ValueError
        If fewer than two components are available (no binary mixture possible).
    """
    t_dim = onehot_tensor.shape[0]
    max_frac = 1.0 - min_frac

    # 1) Nearest vertex -> component 1.
    distances = torch.norm(onehot_tensor - x_suggestion.unsqueeze(0), dim=1)
    j1 = int(torch.argmin(distances).item())

    # 2) Direction from j1 toward the suggestion.
    v1 = onehot_tensor[j1]
    v_suggest = x_suggestion - v1
    v_suggest_norm = torch.norm(v_suggest)
    if v_suggest_norm < 1e-8:
        # At the vertex: pick an arbitrary unit direction to break symmetry.
        rand_dir = torch.randn(t_dim, device=x_suggestion.device, dtype=x_suggestion.dtype)
        v_suggest_unit = rand_dir / torch.norm(rand_dir)
    else:
        v_suggest_unit = v_suggest / v_suggest_norm

    # 3) Score every candidate second component by edge-direction cosine similarity.
    candidates = []
    for j2 in range(t_dim):
        if j2 == j1:
            continue
        edge_vec = onehot_tensor[j2] - v1
        edge_norm = torch.norm(edge_vec)
        if edge_norm < 1e-8:
            continue
        edge_unit = edge_vec / edge_norm
        cosine_sim = float(torch.dot(v_suggest_unit, edge_unit).item())
        projection = float(torch.dot(v_suggest, edge_vec).item()) / float(edge_norm ** 2)
        x1 = snap_composition_to_threshold(projection, composition_threshold, min_frac, max_frac)
        candidates.append(
            {
                "j2": j2,
                "cosine_sim": cosine_sim,
                "x1": x1,
                "is_novel": is_composition_novel(j1, j2, x1, evaluated_mixtures, composition_threshold),
            }
        )

    if not candidates:
        raise ValueError("Cannot create binary mixture: need at least 2 components")

    # 4) Prefer novel candidates, then highest cosine similarity.
    candidates.sort(key=lambda c: (not c["is_novel"], -c["cosine_sim"]))
    best = candidates[0]
    return (j1, best["j2"], best["x1"])


def make_refprop_mixture_string(fluid1: str, fluid2: str, x1: float) -> str:
    """Build a REFPROP mixture string, e.g. ``"REFPROP::R32[0.70000000]&R125[0.30000000]"``."""
    x2 = 1.0 - x1
    return f"REFPROP::{fluid1}[{x1:.8f}]&{fluid2}[{x2:.8f}]"


def edge_key_canonical(j1: int, j2: int) -> Optional[Tuple[int, int]]:
    """Return the canonical edge key ``(low, high)``, or ``None`` if ``j1 == j2``."""
    if j1 == j2:
        return None
    return tuple(sorted((j1, j2)))


def mixture_key_canonical(j1: int, j2: Optional[int], x1: float) -> MixtureKey:
    """Return a canonical mixture key so ``(A, B, 0.7)`` and ``(B, A, 0.3)`` coincide.

    Pure fluids (``j2 is None``) are returned as ``(j1, None, 1.0)`` for backward
    compatibility but are not produced by the mixture pipeline.
    """
    if j2 is None:
        return (j1, None, 1.0)
    if j1 < j2:
        return (j1, j2, x1)
    return (j2, j1, 1.0 - x1)


def format_mixture_name(fluid1: str, fluid2: Optional[str], x1: float) -> str:
    """Human-readable label, e.g. ``"R32[0.30]&R125[0.70]"`` or ``"R134a"`` for pure."""
    if fluid2 is None:
        return fluid1
    x2 = 1.0 - x1
    return f"{fluid1}[{x1:.2f}]&{fluid2}[{x2:.2f}]"
