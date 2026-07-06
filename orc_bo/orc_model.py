"""Organic Rankine Cycle thermodynamic simulation (single source of truth).

This module replaces the ``_simulation`` function that was copy-pasted across every
pipeline script. :class:`ORCSimulator` precomputes the heat-source/sink enthalpies from
:class:`~orc_bo.config.ORCConfig` and evaluates the cycle efficiency and pinch
constraints for a given working fluid and pair of operating pressures.

Pinch constraints use saturation (bubble/dew) temperatures so that zeotropic temperature
glide in mixtures is handled correctly: both ends of each heat exchanger must satisfy the
pinch, and the binding (maximum) violation is returned.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np
import CoolProp.CoolProp as CP

from .config import ORCConfig, ThermoBackend
from .logging_setup import get_logger
from . import thermo

logger = get_logger(__name__)


class SimulationResult(NamedTuple):
    """Outcome of an ORC evaluation.

    Indexable as ``(eta, sink_pinch, source_pinch)`` for backward compatibility, and also
    accessible by attribute. ``sink_pinch``/``source_pinch`` are temperature margins in
    degrees Celsius; positive values violate the pinch (heat would flow the wrong way).
    """

    eta: float
    sink_pinch: float
    source_pinch: float


class ORCSimulator:
    """Evaluate ORC efficiency and pinch constraints for candidate working fluids.

    Parameters
    ----------
    orc:
        Operating conditions and component efficiencies.
    backend:
        Thermodynamic backend for the working fluid (``"REFPROP"`` or ``"HEOS"``).
    """

    def __init__(self, orc: ORCConfig | None = None, backend: ThermoBackend = "REFPROP") -> None:
        self.orc = orc or ORCConfig()
        self.backend = backend
        # Heat-source/sink water enthalpies depend only on configuration; precompute once.
        o = self.orc
        self._hin_src = thermo.enthalpy_tp("water", o.t_in_source_c + 273.15, o.source_pressure_pa)
        self._hout_src = thermo.enthalpy_tp("water", o.t_out_source_c + 273.15, o.source_pressure_pa)
        # Physical efficiency ceiling: the Carnot bound between the hottest source and coldest
        # sink temperatures. No real cycle can exceed it, so any computed eta above it is a
        # numerical artifact (e.g. a fabricated isentropic state) and is rejected below.
        t_hot = o.t_in_source_c + 273.15
        t_cold = o.t_in_sink_c + 273.15
        self._carnot = 1.0 - t_cold / t_hot if t_hot > 0 else 1.0
        # Backend-coverage instrumentation (per fluid handed to can_evaluate).
        self.n_evaluations = 0
        self.n_backend_failures = 0

    @property
    def _penalty(self) -> SimulationResult:
        p = self.orc.infeasible_pinch
        return SimulationResult(self.orc.infeasible_penalty, p, p)

    def can_evaluate(self, wf: str) -> bool:
        """Return whether the backend can build this working fluid's model.

        A mixture equation of state exists only for pairs the backend has parameters for;
        unmatched pairs (common on HEOS) fail at ``AbstractState`` construction, *before* any
        flash. Each call and each failure are counted (see :meth:`backend_failure_report`),
        so callers can skip fluids the backend cannot evaluate instead of wasting SCBO
        retries and mislabelling a backend gap as a merely infeasible fluid.
        """
        self.n_evaluations += 1
        try:
            thermo.make_abstract_state(wf, self.backend)
            return True
        except thermo.ThermoError as exc:
            self.n_backend_failures += 1
            logger.debug("Backend cannot build %s: %s", wf, str(exc)[:80])
            return False

    def backend_failure_report(self) -> str:
        """One-line summary of how many fluids the backend could not evaluate."""
        n, f = self.n_evaluations, self.n_backend_failures
        pct = (100.0 * f / n) if n else 0.0
        return f"backend could not evaluate {f}/{n} fluids ({pct:.0f}%)"

    def simulate(self, wf: str, p_evap_bar: float, p_cond_bar: float) -> SimulationResult:
        """Simulate one ORC operating point.

        Parameters
        ----------
        wf:
            Working-fluid string (pure name or REFPROP mixture spec).
        p_evap_bar, p_cond_bar:
            Evaporator and condenser pressures in bar.

        Returns
        -------
        SimulationResult
            Efficiency and the binding sink/source pinch margins. On any infeasible or
            failed evaluation, returns the configured infeasibility penalty.
        """
        o = self.orc
        p_evap = p_evap_bar * 1e5
        p_cond = p_cond_bar * 1e5

        if p_cond >= p_evap:
            return self._penalty

        try:
            state = thermo.make_abstract_state(wf, self.backend)

            # Evaporator exit: saturated vapor.
            state.update(CP.PQ_INPUTS, p_evap, 1)
            h_g, s_g = state.hmass(), state.smass()

            # Condenser exit: saturated liquid.
            state.update(CP.PQ_INPUTS, p_cond, 0)
            h_f, s_f = state.hmass(), state.smass()

            # Turbine: isentropic expansion with efficiency.
            h2s = thermo.h_isentropic_from_s_p(s_g, p_cond, wf, self.backend, "turbine")
            h2 = h_g - o.turbine_eff * (h_g - h2s)

            # Condenser pinch (handles glide): hot end = dew, cold end = bubble.
            state.update(CP.PQ_INPUTS, p_cond, 1)
            t_dew_cond_c = state.T() - 273.15
            state.update(CP.PQ_INPUTS, p_cond, 0)
            t_bub_cond_c = state.T() - 273.15
            snk_pinch = max(o.t_out_sink_c - t_dew_cond_c, o.t_in_sink_c - t_bub_cond_c)

            # Pump: isentropic compression with efficiency.
            h1s = thermo.h_isentropic_from_s_p(s_f, p_evap, wf, self.backend, "pump")
            h1 = h_f - (h_f - h1s) / o.pump_eff

            # Evaporator pinch (handles glide): cold end = bubble, hot end = dew.
            state.update(CP.PQ_INPUTS, p_evap, 0)
            t_bub_evap_c = state.T() - 273.15
            state.update(CP.PQ_INPUTS, p_evap, 1)
            t_dew_evap_c = state.T() - 273.15
            src_pinch = max(t_dew_evap_c - o.t_in_source_c, t_bub_evap_c - o.t_out_source_c)

            # Energy balance.
            q_evap = o.mfr_source * (self._hin_src - self._hout_src)
            if (h_g - h1) <= 0:
                return self._penalty
            mfr_wf = q_evap / (h_g - h1)
            w_pump = mfr_wf * (h1 - h_f)
            w_turb = mfr_wf * (h_g - h2)
            w_gen = o.generator_eff * w_turb
            eta = (w_gen - w_pump) / q_evap

            ceiling = min(o.eta_max, self._carnot)
            if not np.isfinite(eta) or eta < 0 or eta > ceiling:
                return self._penalty

            return SimulationResult(eta, snk_pinch, src_pinch)

        except thermo.ThermoError as exc:
            logger.debug("ORC simulation failed for wf=%s p=(%.3f,%.3f): %s",
                         wf, p_evap_bar, p_cond_bar, exc)
            return self._penalty
