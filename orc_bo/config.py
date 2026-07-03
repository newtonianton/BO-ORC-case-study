"""Centralized configuration for the orc_bo package.

All tunable constants that were previously hard-coded and duplicated across pipeline
scripts (ORC operating conditions, BO/SCBO hyperparameters, mixture composition bounds,
thermodynamic backend) live here as frozen dataclasses. A single :class:`AppConfig`
bundles them and can be loaded from a TOML file and/or environment variables.

Environment overrides (applied on top of file/defaults):
    ORC_BO_BACKEND       -> thermo.backend ("REFPROP" or "HEOS")
    ORC_BO_DATA_CSV      -> paths.data_csv
    ORC_BO_REFPROP_PATH  -> REFPROP install directory (see orc_bo.thermo; default is the
                            standard Windows location C:\\Program Files (x86)\\REFPROP)
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Literal, Optional

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - older interpreters
    tomllib = None  # type: ignore[assignment]

ThermoBackend = Literal["REFPROP", "HEOS"]

_PACKAGE_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class ORCConfig:
    """Organic Rankine Cycle operating conditions and component efficiencies.

    Temperatures are in degrees Celsius; pressures in Pascal; mass flow in kg/s.
    """

    t_in_source_c: float = 150.0
    t_out_source_c: float = 135.0
    t_in_sink_c: float = 25.0
    t_out_sink_c: float = 35.0
    mfr_source: float = 15.0
    pump_eff: float = 0.8
    turbine_eff: float = 0.65
    generator_eff: float = 0.97
    source_pressure_pa: float = 5e5
    sink_pressure_pa: float = 4e5
    # Feasibility guards used by the simulation.
    eta_max: float = 0.35
    infeasible_penalty: float = -0.05
    infeasible_pinch: float = 500.0


@dataclass(frozen=True)
class MixtureConfig:
    """Binary-mixture composition constraints and discretization."""

    min_mole_frac: float = 0.05
    composition_threshold: float = 0.01

    @property
    def max_mole_frac(self) -> float:
        """Maximum mole fraction of a component (1 - minimum)."""
        return 1.0 - self.min_mole_frac


@dataclass(frozen=True)
class BOConfig:
    """Bayesian optimization and SCBO (TuRBO) hyperparameters."""

    # SCBO inner loop over operating conditions (p_evap, p_cond).
    scbo_dim: int = 2
    batch_size: int = 4
    scbo_n_init: int = 10
    n_candidates: int = 5000
    tr_length_init: float = 0.8
    tr_length_min: float = 0.5 ** 6
    tr_length_max: float = 1.6
    success_tolerance: int = 3
    # Acquisition optimization (outer fluid-space loop).
    mc_samples: int = 512
    num_restarts: int = 10
    raw_samples: int = 512
    # Whole-SCBO retry attempts when no feasible operating point is found.
    scbo_max_retries: int = 4
    # GP fitting noise schedule.
    gp_noise: float = 1e-5
    gp_max_noise: float = 1.0


@dataclass(frozen=True)
class TwoStageConfig:
    """Hyperparameters specific to the two-stage property-targeting pipeline."""

    n_property_targets: int = 20
    # required_valid_init counts REACHED targets (reachability), not validity/operability.
    required_valid_init: int = 8
    target_budget: int = 3
    radius_norm: float = 0.15
    # Probability threshold for the REACHABILITY GPC (a property region is reachable).
    gpc_feasibility_threshold: float = 0.5
    gpc_candidates: int = 20_000
    gpc_max_rounds: int = 4
    gpc_steps: int = 200
    gpc_lr: float = 0.1
    # Step-8 cEI exploitation loop.
    system_budget: int = 3
    failure_allowance: int = 3
    # Operable critical-temperature band [K] for property targets. None -> derived from the
    # ORC source temperature (tc_min = source, tc_max = source + 200 K). Targets (Steps 3 & 6)
    # are sampled only within this band, concentrating the search on fluids that can actually
    # run the cycle instead of e.g. 700 K siloxanes. The band is clamped to the observed range.
    tc_min_k: Optional[float] = None
    tc_max_k: Optional[float] = None


@dataclass(frozen=True)
class ThermoConfig:
    """Thermodynamic backend selection and fallback behavior."""

    backend: ThermoBackend = "REFPROP"
    # When the backend cannot return mixture properties, fall back to analytic mixing
    # rules (logged and counted) instead of raising.
    allow_mixing_rule_fallback: bool = True


@dataclass(frozen=True)
class Paths:
    """Filesystem locations used by the package."""

    package_root: Path = _PACKAGE_ROOT
    data_csv: Path = _PACKAGE_ROOT / "data" / "Joback_Refrigerants.csv"


@dataclass(frozen=True)
class AppConfig:
    """Top-level configuration bundle."""

    orc: ORCConfig = field(default_factory=ORCConfig)
    bo: BOConfig = field(default_factory=BOConfig)
    mixture: MixtureConfig = field(default_factory=MixtureConfig)
    twostage: TwoStageConfig = field(default_factory=TwoStageConfig)
    thermo: ThermoConfig = field(default_factory=ThermoConfig)
    paths: Paths = field(default_factory=Paths)


def _apply_table(obj: Any, table: Optional[Dict[str, Any]]) -> Any:
    """Return a copy of a frozen dataclass with keys from ``table`` overridden."""
    if not table:
        return obj
    known = {k: v for k, v in table.items() if k in obj.__dataclass_fields__}
    return replace(obj, **known)


def load_config(path: Optional[Path] = None) -> AppConfig:
    """Build an :class:`AppConfig` from defaults, an optional TOML file, and env vars.

    Parameters
    ----------
    path:
        Optional path to a TOML file with ``[orc]``, ``[bo]``, ``[mixture]``,
        ``[thermo]`` and/or ``[paths]`` tables. Missing keys keep their defaults.

    Returns
    -------
    AppConfig
        The resolved configuration.
    """
    config = AppConfig()

    if path is not None and tomllib is not None:
        with open(path, "rb") as handle:
            data = tomllib.load(handle)
        config = AppConfig(
            orc=_apply_table(config.orc, data.get("orc")),
            bo=_apply_table(config.bo, data.get("bo")),
            mixture=_apply_table(config.mixture, data.get("mixture")),
            twostage=_apply_table(config.twostage, data.get("twostage")),
            thermo=_apply_table(config.thermo, data.get("thermo")),
            paths=config.paths,
        )

    # Environment overrides take highest precedence.
    backend = os.environ.get("ORC_BO_BACKEND")
    if backend:
        config = replace(config, thermo=replace(config.thermo, backend=backend))  # type: ignore[arg-type]

    data_csv = os.environ.get("ORC_BO_DATA_CSV")
    if data_csv:
        config = replace(config, paths=replace(config.paths, data_csv=Path(data_csv)))

    return config


# A module-level default for convenience; callers may also build their own.
DEFAULT_CONFIG = AppConfig()
