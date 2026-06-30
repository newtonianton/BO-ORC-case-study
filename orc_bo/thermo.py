"""Thermodynamic property access layer.

This is the only module that talks to CoolProp/REFPROP. Everything else in the package
goes through these helpers so that the backend (REFPROP vs HEOS) and the
REFPROP-unavailable fallback behavior are configured in exactly one place.

Key behaviors
-------------
* Pure fluids work on either backend; binary mixtures generally require REFPROP.
* When REFPROP cannot return mixture critical properties, :func:`critical_properties`
  optionally falls back to analytic mixing rules (Kay's rule for ``Tc``, the inverse-sum
  rule for ``Pc``). Every fallback is **logged and counted** (see :func:`fallback_stats`)
  rather than silently swallowed, so callers can audit how often it happened.
"""
from __future__ import annotations

from collections import Counter
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import numpy as np
import CoolProp.CoolProp as CP

from .config import ThermoBackend, ThermoConfig
from .logging_setup import get_logger

logger = get_logger(__name__)

# CoolProp raises ValueError for most failed property evaluations; some backends raise
# RuntimeError. These are the exceptions we treat as "property unavailable".
ThermoError = (ValueError, RuntimeError)

# Instrumentation: counts of REFPROP->mixing-rule fallbacks, keyed by reason.
_FALLBACK_COUNTS: Counter = Counter()


def fallback_stats() -> Dict[str, int]:
    """Return a copy of the REFPROP->mixing-rule fallback counters."""
    return dict(_FALLBACK_COUNTS)


def reset_fallback_stats() -> None:
    """Reset the fallback counters (useful between runs or tests)."""
    _FALLBACK_COUNTS.clear()


@lru_cache(maxsize=1)
def refprop_available() -> bool:
    """Return whether the REFPROP backend is usable on this machine (cached).

    The result is cached for the process lifetime; REFPROP availability does not change
    while running.
    """
    try:
        CP.PropsSI("Tcrit", "REFPROP::R134a")
        return True
    except ThermoError as exc:
        logger.warning("REFPROP backend unavailable: %s", exc)
        return False


def parse_fluid_string(wf: str) -> Tuple[str, Optional[List[float]]]:
    """Parse a working-fluid string into a CoolProp fluid spec and mole fractions.

    Examples
    --------
    ``"REFPROP::R134a"``            -> ``("R134a", None)``
    ``"R134a"``                     -> ``("R134a", None)``
    ``"REFPROP::R32[0.7]&R125[0.3]"`` -> ``("R32&R125", [0.7, 0.3])``

    Returns
    -------
    tuple
        ``(fluid_string, mole_fractions)`` where ``mole_fractions`` is ``None`` for a
        pure fluid and a normalized list for a mixture.
    """
    fluid_str = wf[len("REFPROP::"):] if wf.startswith("REFPROP::") else wf

    if "&" not in fluid_str:
        return fluid_str, None

    fluids: List[str] = []
    fractions: List[float] = []
    for component in fluid_str.split("&"):
        if "[" in component:
            name, _, frac = component.partition("[")
            fluids.append(name)
            fractions.append(float(frac.rstrip("]")))
        else:
            fluids.append(component)
            fractions.append(1.0)

    total = sum(fractions)
    fractions = [f / total for f in fractions]
    return "&".join(fluids), fractions


def make_abstract_state(wf: str, backend: ThermoBackend = "REFPROP") -> CP.AbstractState:
    """Create a configured CoolProp ``AbstractState`` for a pure fluid or mixture.

    Parameters
    ----------
    wf:
        Working-fluid string (pure name or REFPROP mixture spec).
    backend:
        ``"REFPROP"`` or ``"HEOS"``.

    Returns
    -------
    CoolProp.AbstractState
        State object with mole fractions set for mixtures.
    """
    fluid_str, mole_fracs = parse_fluid_string(wf)
    state = CP.AbstractState(backend, fluid_str)
    if mole_fracs is not None:
        state.set_mole_fractions(mole_fracs)
    return state


@lru_cache(maxsize=256)
def pure_critical_properties(
    fluid: str, backend: ThermoBackend = "REFPROP"
) -> Tuple[float, float]:
    """Return ``(Tcrit [K], Pcrit [Pa])`` for a pure fluid (cached)."""
    spec = f"{backend}::{fluid}" if backend == "REFPROP" else fluid
    return CP.PropsSI("Tcrit", spec), CP.PropsSI("Pcrit", spec)


@lru_cache(maxsize=256)
def triple_pressure(fluid: str, backend: ThermoBackend = "REFPROP") -> float:
    """Return the triple-point pressure [Pa] of a pure fluid (cached)."""
    spec = f"{backend}::{fluid}" if backend == "REFPROP" else fluid
    return CP.PropsSI("ptriple", spec)


def enthalpy_tp(fluid: str, t_k: float, p_pa: float, backend: ThermoBackend = "HEOS") -> float:
    """Return mass-specific enthalpy [J/kg] at temperature ``t_k`` and pressure ``p_pa``.

    Used for the heat-source/sink water streams, which are evaluated with the HEOS backend
    so they work regardless of REFPROP availability.
    """
    spec = f"{backend}::{fluid}" if backend == "REFPROP" else fluid
    return CP.PropsSI("H", "T", t_k, "P", p_pa, spec)


@lru_cache(maxsize=256)
def molar_mass(fluid: str, backend: ThermoBackend = "REFPROP") -> float:
    """Return the molar mass [kg/mol] of a pure fluid (cached).

    Useful for converting mole fractions to mass fractions (e.g. for mass-weighted GWP).
    """
    spec = f"{backend}::{fluid}" if backend == "REFPROP" else fluid
    return CP.PropsSI("molar_mass", spec)


def kay_critical_temperature(tc1: float, tc2: float, x1: float) -> float:
    """Kay's rule pseudo-critical temperature: ``x1*Tc1 + (1-x1)*Tc2``."""
    return x1 * tc1 + (1.0 - x1) * tc2


def inverse_sum_critical_pressure(pc1: float, pc2: float, x1: float) -> float:
    """Inverse-sum mixing rule for pseudo-critical pressure: ``1 / (x1/Pc1 + x2/Pc2)``."""
    return 1.0 / (x1 / pc1 + (1.0 - x1) / pc2)


def critical_properties(
    fluid1: str,
    fluid2: Optional[str],
    x1: float,
    config: Optional[ThermoConfig] = None,
) -> Tuple[float, float]:
    """Return ``(Tcrit, Pcrit)`` for a pure fluid or binary mixture.

    For mixtures the REFPROP equation of state is tried first; if it is unavailable or
    fails and ``config.allow_mixing_rule_fallback`` is set, analytic mixing rules are used
    and the event is logged and counted. With fallback disabled, the underlying error
    propagates.

    Parameters
    ----------
    fluid1:
        First (or only) component name.
    fluid2:
        Second component name, or ``None`` for a pure fluid.
    x1:
        Mole fraction of ``fluid1`` (ignored for pure fluids).
    config:
        Thermo configuration; defaults to :class:`ThermoConfig` defaults.

    Returns
    -------
    tuple
        ``(Tcrit [K], Pcrit [Pa])``.
    """
    config = config or ThermoConfig()

    if fluid2 is None:
        return pure_critical_properties(fluid1, config.backend)

    # Mixture: try the equation of state first.
    if config.backend == "REFPROP" and refprop_available():
        try:
            wf = make_refprop_mixture_string(fluid1, fluid2, x1)
            return CP.PropsSI("Tcrit", wf), CP.PropsSI("Pcrit", wf)
        except ThermoError as exc:
            if not config.allow_mixing_rule_fallback:
                raise
            _FALLBACK_COUNTS["refprop_mixture_failed"] += 1
            logger.warning(
                "REFPROP mixture Tcrit/Pcrit failed for %s/%s x1=%.4f (%s); using mixing rules",
                fluid1,
                fluid2,
                x1,
                exc,
            )
    elif not config.allow_mixing_rule_fallback:
        raise RuntimeError(
            f"Mixture properties for {fluid1}/{fluid2} require REFPROP, which is unavailable"
        )
    else:
        _FALLBACK_COUNTS["refprop_unavailable"] += 1

    # Analytic mixing-rule fallback.
    tc1, pc1 = pure_critical_properties(fluid1, "HEOS")
    tc2, pc2 = pure_critical_properties(fluid2, "HEOS")
    tc_mix = kay_critical_temperature(tc1, tc2, x1)
    pc_mix = inverse_sum_critical_pressure(pc1, pc2, x1) if pc1 > 0 and pc2 > 0 else (
        x1 * pc1 + (1.0 - x1) * pc2
    )
    return tc_mix, pc_mix


def make_refprop_mixture_string(fluid1: str, fluid2: str, x1: float) -> str:
    """Build a REFPROP mixture string (re-exported from :mod:`orc_bo.geometry`)."""
    x2 = 1.0 - x1
    return f"REFPROP::{fluid1}[{x1:.8f}]&{fluid2}[{x2:.8f}]"


def h_isentropic_from_s_p(
    s_target: float,
    p: float,
    wf: str,
    backend: ThermoBackend = "REFPROP",
    phase_hint: str = "turbine",
) -> float:
    """Enthalpy [J/kg] at pressure ``p`` and entropy ``s_target`` for fluid ``wf``.

    Tries a direct ``(P, S)`` flash; if the backend cannot invert directly (a known
    REFPROP limitation, historically "error 1450"), falls back to bisection on enthalpy
    at fixed pressure within an appropriate bracket for the turbine (two-phase) or pump
    (subcooled) leg.

    Parameters
    ----------
    s_target:
        Target mass-specific entropy [J/kg/K].
    p:
        Pressure [Pa].
    wf:
        Working-fluid string.
    backend:
        Thermo backend.
    phase_hint:
        ``"turbine"`` (expand into two-phase) or ``"pump"`` (compress subcooled liquid).

    Returns
    -------
    float
        The isentropic enthalpy [J/kg].
    """
    # Direct inversion.
    try:
        state = make_abstract_state(wf, backend)
        state.update(CP.PSmass_INPUTS, p, s_target)
        return state.hmass()
    except ThermoError:
        pass  # Fall through to bisection.

    state = make_abstract_state(wf, backend)
    if phase_hint == "turbine":
        state.update(CP.PQ_INPUTS, p, 0)
        h_bub, s_bub = state.hmass(), state.smass()
        state.update(CP.PQ_INPUTS, p, 1)
        h_dew, s_dew = state.hmass(), state.smass()
        s_target = min(max(s_target, s_bub * 1.0001), s_dew * 0.9999)
        lo, hi = h_bub, h_dew
    else:
        state.update(CP.PQ_INPUTS, p, 0)
        t_bub, h_bub, s_bub = state.T(), state.hmass(), state.smass()
        t_trip = state.Ttriple()
        t_low = max(t_trip + 5.0, t_bub - 30.0)
        try:
            state.update(CP.PT_INPUTS, p, t_low)
            h_low = state.hmass()
        except ThermoError:
            state.update(CP.PT_INPUTS, p, min(t_bub - 10.0, t_bub * 0.98))
            h_low = state.hmass()
        if s_target > s_bub:
            s_target = s_bub * 0.9999
        lo, hi = min(h_low, h_bub), max(h_low, h_bub)

    def entropy_residual(h: float) -> float:
        state.update(CP.HmassP_INPUTS, h, p)
        return state.smass() - s_target

    try:
        f_lo, f_hi = entropy_residual(lo), entropy_residual(hi)
        if np.isnan(f_lo) or np.isnan(f_hi) or f_lo * f_hi > 0:
            return 0.5 * (lo + hi)
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            f_mid = entropy_residual(mid)
            if not np.isfinite(f_mid) or abs(hi - lo) < 1e-3:
                return mid
            if f_mid * f_lo < 0:
                hi, f_hi = mid, f_mid
            else:
                lo, f_lo = mid, f_mid
        return 0.5 * (lo + hi)
    except ThermoError:
        return 0.5 * (lo + hi)
