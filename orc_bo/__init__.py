"""orc_bo: constrained Bayesian optimization for ORC working-fluid selection.

The package selects working fluids for an Organic Rankine Cycle (ORC) by running
constrained Bayesian optimization over a one-hot fluid space. It handles both pure
fluids (one-hot vertices) and binary mixtures (edges between vertices) within a single
unified pipeline.

Public modules
--------------
config        Centralized configuration (dataclasses + env/TOML overrides).
logging_setup Logging configuration helper.
thermo        The only module that talks to CoolProp/REFPROP; property access layer.
geometry      Geometric projection of continuous suggestions to fluids/mixtures.
orc_model     Single source of truth for the ORC thermodynamic simulation.
constraints   Composition, pinch and environmental constraint helpers.
scbo          SCBO (TuRBO) trust-region state and step.
targeting     Stage-1 property targeting for the two-stage pipeline.
pipelines     One-stage and two-stage optimization pipelines.
"""
from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
