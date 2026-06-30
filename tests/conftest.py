"""Shared pytest fixtures and the REFPROP availability marker.

Tests marked ``@pytest.mark.refprop`` are automatically skipped when the REFPROP backend
cannot be loaded (e.g. on machines without a REFPROP license), so the math-critical
HEOS/geometry/mixing-rule tests still run everywhere.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from orc_bo import thermo

_GOLDEN_PATH = Path(__file__).parent / "golden" / "golden_values.json"


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip REFPROP-marked tests when REFPROP is unavailable."""
    if thermo.refprop_available():
        return
    skip_refprop = pytest.mark.skip(reason="REFPROP backend not available on this machine")
    for item in items:
        if "refprop" in item.keywords:
            item.add_marker(skip_refprop)


@pytest.fixture(scope="session")
def golden() -> dict:
    """Load the captured golden values (regression baseline)."""
    with open(_GOLDEN_PATH, "r", encoding="utf-8") as handle:
        return json.load(handle)
