"""Tests for the thermodynamic property layer (HEOS pure props, REFPROP-only mixtures)."""
from __future__ import annotations

import math

import pytest

from orc_bo import thermo
from orc_bo.config import ThermoConfig


def test_parse_pure_fluid():
    assert thermo.parse_fluid_string("REFPROP::R134a") == ("R134a", None)
    assert thermo.parse_fluid_string("R134a") == ("R134a", None)


def test_parse_mixture_normalizes_fractions():
    fluid_str, fracs = thermo.parse_fluid_string("REFPROP::R32[0.6]&R125[0.2]")
    assert fluid_str == "R32&R125"
    assert fracs is not None
    assert math.isclose(sum(fracs), 1.0)
    assert math.isclose(fracs[0], 0.75)  # 0.6 / (0.6 + 0.2)


def test_pure_critical_properties_heos():
    tc, pc = thermo.pure_critical_properties("R134a", "HEOS")
    assert 370 < tc < 378  # ~374 K
    assert 4.0e6 < pc < 4.1e6


def test_critical_properties_pure_passthrough():
    cfg = ThermoConfig(backend="HEOS")
    tc, pc = thermo.critical_properties("R134a", None, 1.0, cfg)
    tc_ref, pc_ref = thermo.pure_critical_properties("R134a", "HEOS")
    assert tc == tc_ref and pc == pc_ref


def test_mixture_without_refprop_raises():
    # No mixing-rule fallback exists: mixtures on a non-REFPROP backend must raise, and
    # the failed attempt still counts as one screening call (cost accounting).
    thermo.reset_screen_count()
    cfg = ThermoConfig(backend="HEOS")
    with pytest.raises(RuntimeError):
        thermo.critical_properties("R32", "R125", 0.5, cfg)
    assert thermo.screen_count() == 1


@pytest.mark.refprop
def test_refprop_mixture_matches_within_tolerance():
    cfg = ThermoConfig(backend="REFPROP")
    tc, pc = thermo.critical_properties("R32", "R125", 0.5, cfg)
    # REFPROP mixture critical point should be physically sensible.
    assert 330 < tc < 355
    assert 3.5e6 < pc < 5.9e6
