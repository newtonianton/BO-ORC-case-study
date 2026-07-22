"""Thermodynamic property access layer.

This is the only module that talks to CoolProp/REFPROP. Everything else in the package
goes through these helpers so that the backend (REFPROP vs HEOS) and the
REFPROP-unavailable fallback behavior are configured in exactly one place.

Key behaviors
-------------
* Pure fluids work on either backend; binary mixtures require REFPROP.
* When REFPROP cannot return mixture critical properties, :func:`critical_properties`
  raises (no analytic mixing-rule fallback): a mixture the reference model cannot
  describe is treated as unrealizable, and pipelines skip it. Mixtures that *screen*
  fine but later fail cycle simulation instead receive the infeasible penalty, which
  labels the validity classifier away from that property region.
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import List, Optional, Tuple

import numpy as np
import CoolProp.CoolProp as CP

from .config import ThermoBackend, ThermoConfig
from .logging_setup import get_logger

logger = get_logger(__name__)

# Default REFPROP install directory on Windows (contains REFPRP64.dll and the fluid files).
_DEFAULT_REFPROP_PATH = r"C:\Program Files (x86)\REFPROP"


def _configure_refprop() -> None:
    """Point CoolProp at the REFPROP installation so its REFPROP backend can load.

    CoolProp needs the directory containing ``REFPRP64.dll`` and the fluid files. It is read
    from the ``ORC_BO_REFPROP_PATH`` environment variable, defaulting to the standard Windows
    location. This must run before the first REFPROP call, so it runs at import time; it is a
    no-op when the directory is absent (machines without REFPROP), leaving pure-fluid HEOS
    work unaffected.
    """
    path = os.environ.get("ORC_BO_REFPROP_PATH", _DEFAULT_REFPROP_PATH)
    if path and os.path.isdir(path):
        try:
            CP.set_config_string(CP.ALTERNATIVE_REFPROP_PATH, path)
        except (ValueError, RuntimeError):  # pragma: no cover - best-effort config
            logger.debug("Could not set REFPROP path to %s", path)


_configure_refprop()

# CoolProp raises ValueError for most failed property evaluations; some backends raise
# RuntimeError. These are the exceptions we treat as "property unavailable".
ThermoError = (ValueError, RuntimeError)

# Instrumentation: standalone property-screen calls (cost accounting). Every call to
# :func:`critical_properties` is one lab-scale screening attempt, including ones whose result
# is later rejected. Pipelines reset this at the start of Stage-1 targeting and read it after,
# so it measures the two-stage screening count L used to charge the cost-weighted budget.
_SCREEN_COUNT: list = [0]


def reset_screen_count() -> None:
    """Reset the property-screen counter (call before a targeting phase)."""
    _SCREEN_COUNT[0] = 0


def screen_count() -> int:
    """Number of :func:`critical_properties` screening calls since the last reset."""
    return _SCREEN_COUNT[0]


@lru_cache(maxsize=1)
def refprop_available() -> bool:
    """Return whether the REFPROP backend is usable on this machine (cached).

    The result is cached for the process lifetime; REFPROP availability does not change
    while running.
    """
    try:
        CP.PropsSI("Tcrit", "REFPROP::R134a") # test common fluid, CoolProp acts as wrapper
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


def critical_properties(
    fluid1: str,
    fluid2: Optional[str],
    x1: float,
    config: Optional[ThermoConfig] = None,
    refprop1: Optional[str] = None,
    refprop2: Optional[str] = None,
) -> Tuple[float, float]:
    """Return ``(Tcrit, Pcrit)`` for a pure fluid or binary mixture.

    Mixture critical points come from the REFPROP equation of state only. If REFPROP is
    unavailable or cannot evaluate the pair, the error **propagates** (there is no analytic
    mixing-rule fallback): a mixture the reference model cannot describe is treated as
    unrealizable and skipped by the pipelines. The failed attempt still counts as one
    lab-scale screen for cost accounting.

    Parameters
    ----------
    fluid1:
        First (or only) component's display/HEOS name.
    fluid2:
        Second component's display/HEOS name, or ``None`` for a pure fluid.
    x1:
        Mole fraction of ``fluid1`` (ignored for pure fluids).
    config:
        Thermo configuration; defaults to :class:`ThermoConfig` defaults.
    refprop1, refprop2:
        Optional REFPROP names for the two components (REFPROP and HEOS do not share a
        name space). Default to ``fluid1``/``fluid2`` when omitted.

    Returns
    -------
    tuple
        ``(Tcrit [K], Pcrit [Pa])``.

    Raises
    ------
    ValueError, RuntimeError
        When the backend cannot evaluate the fluid or mixture.
    """
    config = config or ThermoConfig()
    _SCREEN_COUNT[0] += 1  # one lab-scale screening attempt (cost accounting)
    rp1 = refprop1 or fluid1
    rp2 = refprop2 or fluid2

    if fluid2 is None:
        return pure_critical_properties(rp1 if config.backend == "REFPROP" else fluid1, config.backend)

    # Mixture: only the equation of state is acceptable. Use the AbstractState critical-point
    # API (not PropsSI("Tcrit", ...), which routes through the single-fluid Props1SI path and
    # cannot parse mixture strings, spuriously failing for every mixture).
    if config.backend != "REFPROP" or not refprop_available():
        raise RuntimeError(
            f"Mixture properties for {fluid1}/{fluid2} require REFPROP, which is unavailable"
        )
    state = make_abstract_state(make_refprop_mixture_string(rp1, rp2, x1), "REFPROP")
    return state.T_critical(), state.p_critical()


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

    # Bisect for the enthalpy whose entropy matches the target. If the bracket does not
    # enclose a sign change (the target entropy lies outside the two-phase range) or the
    # backend returns a non-finite residual, the isentropic state cannot be trusted: raise
    # rather than return the bracket midpoint. Returning the midpoint historically fabricated
    # a plausible-looking enthalpy that produced above-Carnot efficiencies; the caller
    # (:meth:`ORCSimulator.simulate`) instead treats the raised error as an infeasible point.
    try:
        f_lo, f_hi = entropy_residual(lo), entropy_residual(hi)
    except ThermoError as exc:
        raise ValueError(f"isentropic solve could not evaluate the bracket for {wf}") from exc
    if not np.isfinite(f_lo) or not np.isfinite(f_hi) or f_lo * f_hi > 0:
        raise ValueError(
            f"isentropic solve could not bracket the target entropy for {wf} at p={p:.0f} Pa"
        )
    for _ in range(60):
        if abs(hi - lo) < 1e-3:
            break
        mid = 0.5 * (lo + hi)
        try:
            f_mid = entropy_residual(mid)
        except ThermoError as exc:
            raise ValueError(f"isentropic solve failed during bisection for {wf}") from exc
        if not np.isfinite(f_mid):
            raise ValueError(f"isentropic solve produced a non-finite residual for {wf}")
        if f_mid * f_lo < 0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid
    return 0.5 * (lo + hi)
