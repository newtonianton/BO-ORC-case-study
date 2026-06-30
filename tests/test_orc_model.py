"""Golden-value and behavioral tests for the ORC simulation (HEOS backend)."""
from __future__ import annotations

import math

import pytest

from orc_bo.orc_model import ORCSimulator, SimulationResult


@pytest.fixture(scope="module")
def simulator() -> ORCSimulator:
    return ORCSimulator(backend="HEOS")


def test_golden_orc_values_reproduce(simulator, golden):
    """The refactored simulation reproduces the captured golden efficiencies/pinches."""
    assert golden["backend"] == "HEOS"
    for case in golden["orc"]:
        result = simulator.simulate(case["wf"], case["p_evap"], case["p_cond"])
        assert math.isclose(result.eta, case["eta"], rel_tol=1e-9, abs_tol=1e-9)
        assert math.isclose(result.sink_pinch, case["sink_pinch"], rel_tol=1e-9, abs_tol=1e-9)
        assert math.isclose(result.source_pinch, case["source_pinch"], rel_tol=1e-9, abs_tol=1e-9)


def test_result_is_indexable_and_named(simulator):
    result = simulator.simulate("R134a", 23.0, 4.0)
    assert isinstance(result, SimulationResult)
    assert result[0] == result.eta
    assert result[1] == result.sink_pinch
    assert result[2] == result.source_pinch


def test_condenser_above_evaporator_is_infeasible(simulator):
    result = simulator.simulate("R134a", 4.0, 23.0)  # p_cond > p_evap
    assert result.eta == simulator.orc.infeasible_penalty


def test_unknown_fluid_returns_penalty(simulator):
    result = simulator.simulate("NotAFluid", 10.0, 1.0)
    assert result.eta == simulator.orc.infeasible_penalty


def test_efficiency_within_physical_bounds(simulator):
    result = simulator.simulate("Propane", 18.0, 3.0)
    assert 0.0 <= result.eta <= simulator.orc.eta_max
