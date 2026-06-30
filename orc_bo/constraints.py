"""Constraint handling for working-fluid optimization.

Two layers live here:

* :func:`orc_constraints` - the inequality constraints the SCBO inner loop enforces on a
  simulated operating point (pressure ordering and the two pinch margins). The convention
  is "feasible when ``<= 0``".
* :class:`MixtureConstraintManager` - optional pre-screening of candidate fluids/mixtures
  against composition bounds, immiscible pairs, property ranges, and environmental limits
  (GWP/ODP) and cost. This is the scaffolding for multi-criteria (e.g. efficiency-vs-GWP)
  selection; it is not required by the core pipeline.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

from .logging_setup import get_logger
from .orc_model import SimulationResult

logger = get_logger(__name__)


def orc_constraints(
    result: SimulationResult, p_evap_bar: float, p_cond_bar: float
) -> Tuple[float, float, float]:
    """Return the ORC inequality constraints; each is feasible when ``<= 0``.

    Parameters
    ----------
    result:
        Output of :meth:`orc_bo.orc_model.ORCSimulator.simulate`.
    p_evap_bar, p_cond_bar:
        Evaporator and condenser pressures in bar.

    Returns
    -------
    tuple
        ``(pressure_ordering, sink_pinch, source_pinch)`` where ``pressure_ordering`` is
        ``p_cond - p_evap`` (condenser pressure must be below evaporator pressure).
    """
    return (p_cond_bar - p_evap_bar, result.sink_pinch, result.source_pinch)


class MixtureConstraintManager:
    """Manage and enforce screening constraints for candidate fluids and mixtures.

    Supported constraints: per-fluid and global composition bounds, immiscible pairs,
    property (Tc/Pc) ranges, mole-fraction-weighted GWP/ODP/cost limits, and an outright
    forbidden-fluid set. Violations are counted for reporting.
    """

    def __init__(self, global_min_frac: float = 0.05, global_max_frac: float = 0.95) -> None:
        self.global_min_frac = global_min_frac
        self.global_max_frac = global_max_frac

        self.composition_bounds: Dict[str, Dict[str, float]] = {}
        self.immiscible_pairs: Set[Tuple[str, str]] = set()
        self.property_ranges: Dict[str, Tuple[float, float]] = {
            "Tc": (200.0, 600.0),  # K
            "Pc": (1e5, 1e7),  # Pa
        }
        self.gwp_limits: Dict[str, float] = {}
        self.max_gwp_weighted: Optional[float] = None
        self.odp_limits: Dict[str, float] = {}
        self.max_odp_weighted: Optional[float] = None
        self.cost_per_kg: Dict[str, float] = {}
        self.max_cost_weighted: Optional[float] = None
        self.forbidden_fluids: Set[str] = set()

        self.total_checks = 0
        self.violations: Dict[str, int] = {
            "composition": 0,
            "immiscible": 0,
            "property": 0,
            "gwp": 0,
            "odp": 0,
            "cost": 0,
            "forbidden": 0,
        }

    # ------------------------------------------------------------------ composition
    def add_composition_bound(self, fluid: str, min_frac: float, max_frac: float) -> None:
        """Set custom composition bounds for a specific fluid."""
        if min_frac < 0 or max_frac > 1 or min_frac > max_frac:
            raise ValueError(f"Invalid bounds: min={min_frac}, max={max_frac}")
        self.composition_bounds[fluid] = {"min": min_frac, "max": max_frac}
        logger.info("Composition bound: %s in [%.3f, %.3f]", fluid, min_frac, max_frac)

    def get_composition_bounds(self, fluid: str) -> Tuple[float, float]:
        """Return the composition bounds for a fluid (custom if set, else global)."""
        if fluid in self.composition_bounds:
            b = self.composition_bounds[fluid]
            return b["min"], b["max"]
        return self.global_min_frac, self.global_max_frac

    def check_composition(self, fluid: str, x: float) -> bool:
        """Return whether mole fraction ``x`` of ``fluid`` is within bounds."""
        min_frac, max_frac = self.get_composition_bounds(fluid)
        return min_frac <= x <= max_frac

    # ----------------------------------------------------------------- miscibility
    def add_immiscible_pair(self, fluid1: str, fluid2: str) -> None:
        """Mark two fluids as immiscible (forbidden to mix)."""
        self.immiscible_pairs.add(tuple(sorted((fluid1, fluid2))))
        logger.info("Immiscible pair: %s - %s", fluid1, fluid2)

    def check_miscibility(self, fluid1: str, fluid2: str) -> bool:
        """Return whether two fluids may be mixed."""
        return tuple(sorted((fluid1, fluid2))) not in self.immiscible_pairs

    # -------------------------------------------------------------------- property
    def set_property_range(self, property_name: str, min_val: float, max_val: float) -> None:
        """Set the allowable range for a property (``"Tc"`` or ``"Pc"``), SI units."""
        if min_val > max_val:
            raise ValueError(f"Invalid range: {min_val} > {max_val}")
        self.property_ranges[property_name] = (min_val, max_val)
        logger.info("Property range: %s in [%g, %g]", property_name, min_val, max_val)

    def check_property(self, property_name: str, value: float) -> bool:
        """Return whether a property value is within its configured range."""
        if property_name not in self.property_ranges:
            return True
        min_val, max_val = self.property_ranges[property_name]
        return min_val <= value <= max_val

    # ----------------------------------------------------------------- weighted env
    @staticmethod
    def _weighted(
        data: Dict[str, float], fluid1: str, fluid2: Optional[str], x1: float
    ) -> Optional[float]:
        """Mole-fraction-weighted average of per-fluid data, or ``None`` if missing."""
        if fluid1 not in data:
            return None
        if fluid2 is None:
            return data[fluid1]
        if fluid2 not in data:
            return None
        return x1 * data[fluid1] + (1.0 - x1) * data[fluid2]

    def set_gwp_data(self, fluid: str, gwp: float) -> None:
        """Set the 100-year GWP (relative to CO2) for a fluid."""
        self.gwp_limits[fluid] = gwp

    def set_max_gwp(self, max_gwp_weighted: float) -> None:
        """Set the maximum allowable weighted GWP for a mixture."""
        self.max_gwp_weighted = max_gwp_weighted
        logger.info("Max weighted GWP: %g", max_gwp_weighted)

    def compute_gwp(self, fluid1: str, fluid2: Optional[str], x1: float) -> Optional[float]:
        """Return the weighted GWP, or ``None`` if data is unavailable."""
        return self._weighted(self.gwp_limits, fluid1, fluid2, x1)

    def check_gwp(self, fluid1: str, fluid2: Optional[str], x1: float) -> bool:
        """Return whether the GWP constraint is satisfied (or unconstrained/no data)."""
        if self.max_gwp_weighted is None:
            return True
        gwp = self.compute_gwp(fluid1, fluid2, x1)
        if gwp is None:
            logger.warning("GWP data unavailable for %s/%s; treating as satisfied", fluid1, fluid2)
            return True
        return gwp <= self.max_gwp_weighted

    def set_odp_data(self, fluid: str, odp: float) -> None:
        """Set the ODP for a fluid."""
        self.odp_limits[fluid] = odp

    def set_max_odp(self, max_odp_weighted: float) -> None:
        """Set the maximum allowable weighted ODP for a mixture."""
        self.max_odp_weighted = max_odp_weighted

    def compute_odp(self, fluid1: str, fluid2: Optional[str], x1: float) -> Optional[float]:
        """Return the weighted ODP, or ``None`` if data is unavailable."""
        return self._weighted(self.odp_limits, fluid1, fluid2, x1)

    def check_odp(self, fluid1: str, fluid2: Optional[str], x1: float) -> bool:
        """Return whether the ODP constraint is satisfied (or unconstrained/no data)."""
        if self.max_odp_weighted is None:
            return True
        odp = self.compute_odp(fluid1, fluid2, x1)
        return True if odp is None else odp <= self.max_odp_weighted

    # ------------------------------------------------------------------------ cost
    def set_cost_data(self, fluid: str, cost_per_kg: float) -> None:
        """Set the per-kg cost for a fluid."""
        self.cost_per_kg[fluid] = cost_per_kg

    def set_max_cost(self, max_cost_weighted: float) -> None:
        """Set the maximum allowable weighted cost for a mixture."""
        self.max_cost_weighted = max_cost_weighted

    def compute_cost(self, fluid1: str, fluid2: Optional[str], x1: float) -> Optional[float]:
        """Return the weighted cost, or ``None`` if data is unavailable."""
        return self._weighted(self.cost_per_kg, fluid1, fluid2, x1)

    def check_cost(self, fluid1: str, fluid2: Optional[str], x1: float) -> bool:
        """Return whether the cost constraint is satisfied (or unconstrained/no data)."""
        if self.max_cost_weighted is None:
            return True
        cost = self.compute_cost(fluid1, fluid2, x1)
        return True if cost is None else cost <= self.max_cost_weighted

    # ------------------------------------------------------------------- forbidden
    def add_forbidden_fluid(self, fluid: str) -> None:
        """Mark a fluid as forbidden (e.g. phased out or unsafe)."""
        self.forbidden_fluids.add(fluid)
        logger.info("Forbidden fluid: %s", fluid)

    def check_forbidden(self, fluid: str) -> bool:
        """Return whether a fluid is allowed (``True``) or forbidden (``False``)."""
        return fluid not in self.forbidden_fluids

    # --------------------------------------------------------------- combined check
    def check_mixture(
        self,
        fluid1: str,
        fluid2: Optional[str],
        x1: float,
        tc: Optional[float] = None,
        pc: Optional[float] = None,
    ) -> Tuple[bool, List[str]]:
        """Run all configured constraints against a candidate.

        Parameters
        ----------
        fluid1:
            First (or only) component.
        fluid2:
            Second component, or ``None`` for a pure fluid.
        x1:
            Mole fraction of ``fluid1``.
        tc, pc:
            Optional critical temperature/pressure for property-range checks.

        Returns
        -------
        tuple
            ``(is_valid, violations)`` where ``violations`` is a list of human-readable
            constraint-failure descriptions (empty when valid).
        """
        self.total_checks += 1
        violations: List[str] = []

        for fluid in (fluid1, fluid2):
            if fluid is not None and not self.check_forbidden(fluid):
                violations.append(f"{fluid} is forbidden")
                self.violations["forbidden"] += 1

        if not self.check_composition(fluid1, x1):
            lo, hi = self.get_composition_bounds(fluid1)
            violations.append(f"{fluid1} composition {x1:.3f} outside [{lo:.3f}, {hi:.3f}]")
            self.violations["composition"] += 1

        if fluid2 is not None:
            x2 = 1.0 - x1
            if not self.check_composition(fluid2, x2):
                lo, hi = self.get_composition_bounds(fluid2)
                violations.append(f"{fluid2} composition {x2:.3f} outside [{lo:.3f}, {hi:.3f}]")
                self.violations["composition"] += 1
            if not self.check_miscibility(fluid1, fluid2):
                violations.append(f"{fluid1} and {fluid2} are immiscible")
                self.violations["immiscible"] += 1

        if tc is not None and not self.check_property("Tc", tc):
            lo, hi = self.property_ranges["Tc"]
            violations.append(f"Tc={tc:.1f}K outside [{lo:.1f}, {hi:.1f}]K")
            self.violations["property"] += 1
        if pc is not None and not self.check_property("Pc", pc):
            lo, hi = self.property_ranges["Pc"]
            violations.append(f"Pc={pc / 1e6:.2f}MPa outside [{lo / 1e6:.2f}, {hi / 1e6:.2f}]MPa")
            self.violations["property"] += 1

        if not self.check_gwp(fluid1, fluid2, x1):
            violations.append(f"GWP={self.compute_gwp(fluid1, fluid2, x1):.1f} exceeds {self.max_gwp_weighted:.1f}")
            self.violations["gwp"] += 1
        if not self.check_odp(fluid1, fluid2, x1):
            violations.append(f"ODP={self.compute_odp(fluid1, fluid2, x1):.4f} exceeds {self.max_odp_weighted:.4f}")
            self.violations["odp"] += 1
        if not self.check_cost(fluid1, fluid2, x1):
            violations.append(f"Cost={self.compute_cost(fluid1, fluid2, x1):.2f}/kg exceeds {self.max_cost_weighted:.2f}/kg")
            self.violations["cost"] += 1

        is_valid = not violations
        if not is_valid:
            logger.debug("Invalid candidate %s/%s x1=%.2f: %s", fluid1, fluid2, x1, "; ".join(violations))
        return is_valid, violations

    def export_config(self) -> Dict[str, object]:
        """Return the constraint configuration as a serializable dictionary."""
        return {
            "global_min_frac": self.global_min_frac,
            "global_max_frac": self.global_max_frac,
            "composition_bounds": self.composition_bounds,
            "immiscible_pairs": sorted(self.immiscible_pairs),
            "property_ranges": self.property_ranges,
            "max_gwp": self.max_gwp_weighted,
            "max_odp": self.max_odp_weighted,
            "max_cost": self.max_cost_weighted,
            "forbidden_fluids": sorted(self.forbidden_fluids),
        }


# --------------------------------------------------------------------------- presets
# 100-year GWP values (relative to CO2) for common refrigerants, from IPCC assessments.
_DEFAULT_GWP_DATA: Dict[str, float] = {
    "R32": 675.0,
    "R125": 3500.0,
    "R134a": 1430.0,
    "R1234yf": 4.0,
    "R1234ze": 6.0,
    "R290": 3.0,  # Propane
    "R600a": 3.0,  # Isobutane
    "R717": 0.0,  # Ammonia
    "R744": 1.0,  # CO2
}


def get_low_gwp_constraints(max_gwp_weighted: float = 150.0) -> MixtureConstraintManager:
    """Constraint manager targeting low-GWP refrigerants (weighted GWP below threshold)."""
    manager = MixtureConstraintManager()
    for fluid, gwp in _DEFAULT_GWP_DATA.items():
        manager.set_gwp_data(fluid, gwp)
    manager.set_max_gwp(max_gwp_weighted)
    manager.add_forbidden_fluid("R125")
    manager.add_forbidden_fluid("R134a")
    return manager


def get_non_flammable_constraints() -> MixtureConstraintManager:
    """Constraint manager forbidding common A3 (flammable) refrigerants."""
    manager = MixtureConstraintManager()
    for fluid in ("R290", "R600", "R600a", "R1270", "R170"):
        manager.add_forbidden_fluid(fluid)
    return manager


def get_orc_temperature_constraints(
    t_src_min: float = 373.0, t_src_max: float = 473.0
) -> MixtureConstraintManager:
    """Constraint manager bounding Tc/Pc for a subcritical ORC at a given source range [K]."""
    manager = MixtureConstraintManager()
    manager.set_property_range("Tc", t_src_min - 100.0, t_src_max - 20.0)
    manager.set_property_range("Pc", 1e5, 5e6)
    return manager
