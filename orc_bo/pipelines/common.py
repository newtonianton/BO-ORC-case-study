"""Shared helpers for the optimization pipelines.

Provides fluid loading, candidate realization (turning a snapped one-hot/edge selection
into a concrete working fluid with its properties), and incremental result recording.
These were previously duplicated between the init and loop phases of every pipeline.
"""
from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

from ..config import AppConfig, ThermoBackend
from ..geometry import format_mixture_name, make_refprop_mixture_string
from ..logging_setup import get_logger
from .. import thermo

logger = get_logger(__name__)

TKWARGS = {"device": "cpu", "dtype": torch.double}

# Standardized evaluation-phase labels, shared by both pipelines so the ``phase`` column of
# ``scbo_results.csv`` and the ``[PHASE order]`` log lines mean the same thing everywhere:
#   INIT   - initial batch of ORC-evaluated candidates, before any surrogate-guided search
#            (one-stage: Latin-hypercube init; two-stage: Step 7 SCBO of the reached targets).
#   OPT    - surrogate-guided optimization/exploitation loop
#            (one-stage: qLogEI loop; two-stage: Step 8 cEI loop).
#   TARGET - two-stage property-space targeting progress (Steps 1-6); writes no result rows,
#            appears in logs only.
PHASE_INIT = "INIT"
PHASE_OPT = "OPT"
PHASE_TARGET = "TARGET"


@dataclass(frozen=True)
class Fluid:
    """A dataset fluid carrying both of its backend-specific names.

    ``name`` is the human-readable / CoolProp-HEOS name (column ``fluid`` of the CSV);
    HEOS accepts all of these. ``refprop`` is the REFPROP name (column ``REFPROP_STANDARD``);
    REFPROP mixture strings must use this. The two backends do not share a name space
    (e.g. Dichloroethane is ``R150`` in REFPROP; ``n-Butane`` is ``BUTANE``), so every
    thermo call uses the name matching the backend it targets.
    """

    name: str
    refprop: str


def _backend_name(fluid: Fluid, backend: ThermoBackend) -> str:
    """The fluid name to feed a working-fluid string for the given backend."""
    return fluid.refprop if backend == "REFPROP" else fluid.name

RESULT_FIELDS = [
    "phase", "order", "mixture", "mode", "comp1", "comp2", "x1",
    "Tc_K", "Pc_Pa", "eta", "p_evap_bar", "p_cond_bar",
]


@dataclass
class Candidate:
    """A realized working-fluid candidate ready for SCBO."""

    j1: int
    j2: Optional[int]
    x1: float
    fluid1: str
    fluid2: Optional[str]
    name: str
    wf: str
    tc: float
    pc: float
    ptriple: float
    x_onehot: torch.Tensor


def load_fluids(csv_path: Path) -> List[Fluid]:
    """Load the ordered, de-duplicated fluids from the dataset CSV.

    Reads both the display/HEOS name (column ``fluid``) and the REFPROP name (column
    ``REFPROP_STANDARD``). If the REFPROP column is absent or blank for a row, the display
    name is reused as a best-effort fallback (CoolProp resolves most display names to
    REFPROP fluids on its own).
    """
    fluids: List[Fluid] = []
    seen = set()
    with open(csv_path, "r", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        next(reader, None)  # header
        for row in reader:
            if row and row[0]:
                name = row[0].strip()
                refprop = row[1].strip() if len(row) > 1 and row[1].strip() else name
                if name and name not in seen:
                    seen.add(name)
                    fluids.append(Fluid(name=name, refprop=refprop))
    if not fluids:
        raise RuntimeError(f"No fluid names found in {csv_path}")
    return fluids


def realize_candidate(
    mode: str,
    j1: int,
    j2: Optional[int],
    x1: float,
    fluids: List[Fluid],
    onehot: torch.Tensor,
    config: AppConfig,
) -> Optional[Candidate]:
    """Turn a snapped selection into a concrete :class:`Candidate`, or ``None`` if it cannot
    be realized (critical properties unavailable). This is about realizability, not the
    reachability/validity notions used in the two-stage pipeline.

    Pure-fluid critical properties and triple pressures are evaluated with the HEOS
    backend (equation-of-state pure properties that are available on any machine, and which
    use the display name); mixture critical properties go through
    :func:`orc_bo.thermo.critical_properties`, which prefers REFPROP and falls back to mixing
    rules. The working-fluid string handed to the simulator uses the backend-appropriate name
    (REFPROP names for the REFPROP backend, display names for HEOS).
    """
    f1 = fluids[j1]
    f2 = fluids[j2] if j2 is not None else None
    backend = config.thermo.backend

    try:
        if mode == "pure" or f2 is None:
            tc, pc = thermo.pure_critical_properties(f1.name, "HEOS")
            ptriple = thermo.triple_pressure(f1.name, "HEOS")
            wf = _backend_name(f1, backend)
            x_onehot = onehot[j1].clone()
        else:
            tc, pc = thermo.critical_properties(
                f1.name, f2.name, x1, config.thermo,
                refprop1=f1.refprop, refprop2=f2.refprop,
            )
            ptriple = min(
                thermo.triple_pressure(f1.name, "HEOS"),
                thermo.triple_pressure(f2.name, "HEOS"),
            )
            wf = make_refprop_mixture_string(
                _backend_name(f1, backend), _backend_name(f2, backend), x1
            )
            x_onehot = x1 * onehot[j1] + (1.0 - x1) * onehot[j2]
    except thermo.ThermoError as exc:
        logger.warning("Property lookup failed for %s/%s x1=%.3f: %s",
                       f1.name, f2.name if f2 else None, x1, exc)
        return None

    if not np.isfinite(tc) or not np.isfinite(pc):
        logger.warning("Non-finite properties for %s/%s x1=%.3f",
                       f1.name, f2.name if f2 else None, x1)
        return None

    name = format_mixture_name(f1.name, f2.name if mode != "pure" and f2 else None, x1)
    return Candidate(j1, j2, x1, f1.name, f2.name if f2 else None,
                     name, wf, tc, pc, ptriple, x_onehot)


def format_run_header(
    config: AppConfig,
    *,
    stage: str,
    mode: str,
    seed: int,
    n_init: int,
    scbo_budget: int,
) -> str:
    """Build a ``#``-commented parameter block for the top of ``summary.txt``.

    Records the run arguments and the full resolved configuration so multiple seed runs can
    be checked for parameter consistency before their results are pooled (the config
    directory name does not encode most hyperparameters). Every line is comment-prefixed so
    the fluid list written below it stays trivially parseable.
    """
    lines = [
        "# ORC-BO run parameters (these must match across seeds before pooling results)",
        f"# run: stage={stage} mode={mode} backend={config.thermo.backend} "
        f"seed={seed} n_init={n_init} scbo_budget={scbo_budget}",
        f"# dataset: {config.paths.data_csv}",
    ]
    for section in ("orc", "bo", "mixture", "twostage", "thermo"):
        params = asdict(getattr(config, section))
        kv = " ".join(f"{k}={v}" for k, v in params.items())
        lines.append(f"# [{section}] {kv}")
    lines.append("# ---- evaluated fluids ----")
    return "\n".join(lines) + "\n"


class RunWriter:
    """Incrementally write per-evaluation results, sequence, and summary files.

    Files are flushed after every row so partial results survive interruption and remain
    safe under parallel multi-seed runs. An optional ``header`` (see
    :func:`format_run_header`) is written to the top of ``summary.txt`` as ``#`` comments.
    """

    def __init__(self, outdir: Path, header: Optional[str] = None) -> None:
        outdir.mkdir(parents=True, exist_ok=True)
        self.outdir = outdir
        self._results = open(outdir / "scbo_results.csv", "w", newline="", encoding="utf-8")
        self._sequence = open(outdir / "sequence.csv", "w", newline="", encoding="utf-8")
        self._summary = open(outdir / "summary.txt", "w", encoding="utf-8")
        self._results_writer = csv.DictWriter(self._results, fieldnames=RESULT_FIELDS)
        self._sequence_writer = csv.writer(self._sequence)
        self._results_writer.writeheader()
        self._sequence_writer.writerow(["order", "mixture"])
        if header:
            self._summary.write(header)
        self._flush()

    def record(
        self, phase: str, order: int, mode: str, cand: Candidate,
        eta: float, p_evap_bar: float, p_cond_bar: float,
    ) -> None:
        """Append one evaluation result and flush to disk."""
        self._results_writer.writerow({
            "phase": phase, "order": order, "mixture": cand.name, "mode": mode,
            "comp1": cand.fluid1, "comp2": cand.fluid2 or "", "x1": cand.x1,
            "Tc_K": cand.tc, "Pc_Pa": cand.pc, "eta": eta,
            "p_evap_bar": p_evap_bar, "p_cond_bar": p_cond_bar,
        })
        self._sequence_writer.writerow([order, cand.name])
        self._summary.write(cand.name + "\n")
        self._flush()

    def _flush(self) -> None:
        for handle in (self._results, self._sequence, self._summary):
            handle.flush()

    def close(self) -> None:
        """Close all output files."""
        for handle in (self._results, self._sequence, self._summary):
            handle.close()

    def __enter__(self) -> "RunWriter":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
