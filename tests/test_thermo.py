"""Tests for the thermodynamic property layer (HEOS pure props, mixing rules, fallback)."""
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


def test_mixing_rules_math():
    # Kay's rule is linear; inverse-sum lies between the harmonic inputs.
    assert math.isclose(thermo.kay_critical_temperature(300.0, 400.0, 0.25), 375.0)
    pc = thermo.inverse_sum_critical_pressure(4e6, 2e6, 0.5)
    assert math.isclose(pc, 1.0 / (0.5 / 4e6 + 0.5 / 2e6))


def test_critical_properties_pure_passthrough():
    cfg = ThermoConfig(backend="HEOS")
    tc, pc = thermo.critical_properties("R134a", None, 1.0, cfg)
    tc_ref, pc_ref = thermo.pure_critical_properties("R134a", "HEOS")
    assert tc == tc_ref and pc == pc_ref


def test_mixture_fallback_uses_mixing_rules_and_counts(monkeypatch):
    # Force the REFPROP path off so the mixing-rule fallback is exercised and counted.
    thermo.reset_fallback_stats()
    cfg = ThermoConfig(backend="HEOS", allow_mixing_rule_fallback=True)
    tc, pc = thermo.critical_properties("R32", "R125", 0.5, cfg)

    tc1, pc1 = thermo.pure_critical_properties("R32", "HEOS")
    tc2, pc2 = thermo.pure_critical_properties("R125", "HEOS")
    assert math.isclose(tc, thermo.kay_critical_temperature(tc1, tc2, 0.5))
    assert math.isclose(pc, thermo.inverse_sum_critical_pressure(pc1, pc2, 0.5))
    assert thermo.fallback_stats().get("refprop_unavailable", 0) >= 1


def test_mixture_without_fallback_raises():
    cfg = ThermoConfig(backend="HEOS", allow_mixing_rule_fallback=False)
    with pytest.raises(RuntimeError):
        thermo.critical_properties("R32", "R125", 0.5, cfg)


@pytest.mark.refprop
def test_refprop_mixture_matches_within_tolerance():
    cfg = ThermoConfig(backend="REFPROP", allow_mixing_rule_fallback=False)
    tc, pc = thermo.critical_properties("R32", "R125", 0.5, cfg)
    # REFPROP mixture critical point should be physically sensible.
    assert 330 < tc < 355
    assert 3.5e6 < pc < 5.9e6
