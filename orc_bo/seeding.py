"""Deterministic seed derivation shared across pipelines.

Seeds are derived from a base seed read from the ``PYTHONHASHSEED`` or ``JOBACK_SEED``
environment variable (so multi-seed benchmark runs are reproducible) combined with a
per-purpose integer tag.
"""
from __future__ import annotations

import os

_UINT32_MAX = 2 ** 32 - 1


def base_seed() -> int:
    """Return the base seed from ``PYTHONHASHSEED``/``JOBACK_SEED`` (default 0)."""
    raw = os.environ.get("PYTHONHASHSEED") or os.environ.get("JOBACK_SEED") or "0"
    try:
        return int(raw)
    except ValueError:
        return 0


def derive_seed(tag: int) -> int:
    """Derive a reproducible 32-bit seed for a given integer ``tag``."""
    return ((base_seed() * 1_000_003) ^ (tag * 97_911)) % _UINT32_MAX
