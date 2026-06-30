"""Tests for the ORC SCBO constraints and the MixtureConstraintManager."""
from __future__ import annotations

from orc_bo.constraints import (
    MixtureConstraintManager,
    get_low_gwp_constraints,
    orc_constraints,
)
from orc_bo.orc_model import SimulationResult


def test_orc_constraints_convention():
    # Feasible point: condenser below evaporator, both pinches negative.
    result = SimulationResult(eta=0.1, sink_pinch=-2.0, source_pinch=-5.0)
    c_press, c_sink, c_source = orc_constraints(result, p_evap_bar=20.0, p_cond_bar=4.0)
    assert c_press == 4.0 - 20.0  # negative -> feasible ordering
    assert c_sink < 0 and c_source < 0


def test_composition_bounds_default_and_custom():
    mgr = MixtureConstraintManager()
    assert mgr.check_composition("R32", 0.5)
    assert not mgr.check_composition("R32", 0.99)  # outside global [0.05, 0.95]
    mgr.add_composition_bound("R32", 0.2, 0.8)
    assert not mgr.check_composition("R32", 0.1)
    assert mgr.check_composition("R32", 0.5)


def test_forbidden_and_immiscible():
    mgr = MixtureConstraintManager()
    mgr.add_forbidden_fluid("R125")
    mgr.add_immiscible_pair("Water", "R134a")
    assert not mgr.check_forbidden("R125")
    assert not mgr.check_miscibility("R134a", "Water")  # order-independent
    assert mgr.check_miscibility("R32", "R125")


def test_property_range():
    mgr = MixtureConstraintManager()
    mgr.set_property_range("Tc", 300.0, 400.0)
    assert mgr.check_property("Tc", 350.0)
    assert not mgr.check_property("Tc", 500.0)


def test_weighted_gwp():
    mgr = MixtureConstraintManager()
    mgr.set_gwp_data("R32", 675.0)
    mgr.set_gwp_data("R1234yf", 4.0)
    gwp = mgr.compute_gwp("R32", "R1234yf", 0.3)
    assert abs(gwp - (0.3 * 675.0 + 0.7 * 4.0)) < 1e-9
    # Missing data -> None.
    assert mgr.compute_gwp("R32", "Unknown", 0.5) is None


def test_check_mixture_aggregates_violations():
    mgr = MixtureConstraintManager()
    mgr.add_forbidden_fluid("R125")
    is_valid, violations = mgr.check_mixture("R32", "R125", 0.7)
    assert not is_valid
    assert any("forbidden" in v for v in violations)


def test_low_gwp_preset_forbids_high_gwp():
    mgr = get_low_gwp_constraints()
    # Both HFOs: weighted GWP ~5, well under the 150 limit, neither forbidden.
    valid, _ = mgr.check_mixture("R1234yf", "R1234ze", 0.5)
    assert valid
    # R125 is forbidden in the preset.
    invalid, _ = mgr.check_mixture("R32", "R125", 0.5)
    assert not invalid
    # High weighted GWP (0.3*675 + 0.7*4 ~ 205) exceeds the 150 limit.
    over_limit, violations = mgr.check_mixture("R32", "R1234yf", 0.3)
    assert not over_limit
    assert any("GWP" in v for v in violations)
