"""Tests for the geometric projection (pure vertex snap and mixture edge snap)."""
from __future__ import annotations

import torch

from orc_bo import geometry
from orc_bo.geometry import (
    MAX_MOLE_FRAC,
    MIN_MOLE_FRAC,
    snap_composition_to_threshold,
    snap_to_mixture,
    snap_to_vertex,
)

DTYPE = torch.double


def _onehot(n: int) -> torch.Tensor:
    return torch.eye(n, dtype=DTYPE)


def test_snap_to_vertex_picks_largest_coordinate():
    oh = _onehot(4)
    assert snap_to_vertex(torch.tensor([0.1, 0.7, 0.1, 0.1], dtype=DTYPE), oh) == 1
    assert snap_to_vertex(torch.tensor([0.9, 0.0, 0.0, 0.1], dtype=DTYPE), oh) == 0


def test_snap_to_vertex_ties_resolve_to_smallest_index():
    oh = _onehot(3)
    assert snap_to_vertex(torch.tensor([0.5, 0.5, 0.0], dtype=DTYPE), oh) == 0


def test_snap_composition_rounds_and_clips():
    assert snap_composition_to_threshold(0.527, 0.01) == 0.53
    assert snap_composition_to_threshold(0.524, 0.01) == 0.52
    # Clipping to the configured bounds.
    assert snap_composition_to_threshold(0.001, 0.01) == MIN_MOLE_FRAC
    assert snap_composition_to_threshold(0.999, 0.01) == MAX_MOLE_FRAC


def test_snap_to_mixture_never_returns_pure():
    oh = _onehot(5)
    for point in ([0.95, 0.03, 0.02, 0, 0], [0.5, 0.5, 0, 0, 0], [0.7, 0.3, 0, 0, 0]):
        j1, j2, x1 = snap_to_mixture(torch.tensor(point, dtype=DTYPE), oh, set())
        assert j2 is not None
        assert j1 != j2
        assert MIN_MOLE_FRAC <= x1 <= MAX_MOLE_FRAC


def test_snap_to_mixture_midpoint_is_balanced():
    oh = _onehot(3)
    j1, j2, x1 = snap_to_mixture(torch.tensor([0.5, 0.5, 0.0], dtype=DTYPE), oh, set())
    assert {j1, j2} == {0, 1}
    assert 0.4 < x1 < 0.6


def test_snap_to_mixture_requires_two_components():
    oh = _onehot(1)
    try:
        snap_to_mixture(torch.tensor([1.0], dtype=DTYPE), oh, set())
    except ValueError:
        return
    raise AssertionError("Expected ValueError for a single-component space")


def test_is_composition_novel_same_edge_threshold():
    evaluated = {(0, 1, 0.70)}
    # Within threshold on the same edge -> not novel.
    assert not geometry.is_composition_novel(0, 1, 0.705, evaluated, 0.01)
    # Far on the same edge -> novel.
    assert geometry.is_composition_novel(0, 1, 0.50, evaluated, 0.01)
    # Reversed component order maps x1 -> 1 - x1.
    assert not geometry.is_composition_novel(1, 0, 0.30, evaluated, 0.01)


def test_canonical_keys_are_order_independent():
    assert geometry.mixture_key_canonical(0, 1, 0.7) == (0, 1, 0.7)
    assert geometry.mixture_key_canonical(1, 0, 0.3) == (0, 1, 0.7)
    assert geometry.edge_key_canonical(2, 1) == (1, 2)
    assert geometry.edge_key_canonical(2, 2) is None


def test_refprop_mixture_string_format():
    assert geometry.make_refprop_mixture_string("R32", "R125", 0.7) == (
        "REFPROP::R32[0.70000000]&R125[0.30000000]"
    )
