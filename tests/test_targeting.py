"""Tests for the two-stage targeting building blocks (no REFPROP required)."""
from __future__ import annotations

import torch

from orc_bo.targeting import (
    PropNormalizer,
    gpc_predict_proba,
    greedy_maximin,
    success_mask,
    train_gpc,
)

DTYPE = torch.double


def test_prop_normalizer_roundtrip():
    norm = PropNormalizer(("Tc", "Pc"))
    p = torch.tensor([[300.0, 2e6], [400.0, 5e6], [350.0, 3e6]], dtype=DTYPE)
    norm.fit_from_real_points(p)
    z = norm.to_norm(p, clip=False)
    back = norm.to_real(z)
    assert torch.allclose(back, p, atol=1e-6)
    # Normalized values stay within [0, 1] after clipping.
    assert float(norm.to_norm(p).min()) >= 0.0
    assert float(norm.to_norm(p).max()) <= 1.0


def test_prop_normalizer_maybe_expand_detects_change():
    norm = PropNormalizer(("Tc", "Pc"))
    p = torch.tensor([[300.0, 2e6], [400.0, 5e6]], dtype=DTYPE)
    norm.fit_from_real_points(p)
    _, changed = norm.maybe_expand(p)
    assert changed is False
    p2 = torch.cat([p, torch.tensor([[600.0, 9e6]], dtype=DTYPE)], dim=0)
    _, changed2 = norm.maybe_expand(p2)
    assert changed2 is True


def test_success_mask_radius():
    p_norm = torch.tensor([[0.5, 0.5], [0.1, 0.1]], dtype=DTYPE)
    targets = torch.tensor([[0.5, 0.52], [0.9, 0.9]], dtype=DTYPE)
    flags, rows, dists = success_mask(p_norm, targets, radius=0.05)
    assert flags[0] is True and rows[0] == 0
    assert flags[1] is False and rows[1] == -1
    assert dists[0] < dists[1]


def test_greedy_maximin_spreads_points():
    candidates = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0], [0.5, 0.5]], dtype=DTYPE)
    chosen = greedy_maximin(None, candidates, k=3)
    assert chosen.shape == (3, 2)
    # The three chosen points should be mutually distant (corners, not the centre).
    dmin = torch.cdist(chosen, chosen).fill_diagonal_(float("inf")).min()
    assert float(dmin) >= 1.0


def test_gpc_learns_separable_labels():
    torch.manual_seed(0)
    # Feasible region: x[:, 0] > 0.5.
    x = torch.rand(80, 2, dtype=DTYPE)
    y = (x[:, 0] > 0.5).to(DTYPE).unsqueeze(-1)
    model, likelihood = train_gpc(x, y, steps=120, lr=0.1)
    probe = torch.tensor([[0.9, 0.5], [0.1, 0.5]], dtype=DTYPE)
    proba = gpc_predict_proba(model, likelihood, probe)
    assert float(proba[0]) > float(proba[1])
