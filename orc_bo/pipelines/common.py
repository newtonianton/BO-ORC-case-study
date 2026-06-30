"""Shared helpers for the optimization pipelines.

Provides fluid loading, candidate realization (turning a snapped one-hot/edge selection
into a concrete working fluid with its properties), and incremental result recording.
These were previously duplicated between the init and loop phases of every pipeline.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
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


def load_fluids(csv_path: Path) -> List[str]:
    """Load the ordered, de-duplicated list of fluid names from the dataset CSV."""
    fluids: List[str] = []
    seen = set()
    with open(csv_path, "r", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        next(reader, None)  # header
        for row in reader:
            if row and row[0]:
                name = row[0].strip()
                if name and name not in seen:
                    seen.add(name)
                    fluids.append(name)
    if not fluids:
        raise RuntimeError(f"No fluid names found in {csv_path}")
    return fluids


def realize_candidate(
    mode: str,
    j1: int,
    j2: Optional[int],
    x1: float,
    fluids: List[str],
    onehot: torch.Tensor,
    config: AppConfig,
) -> Optional[Candidate]:
    """Turn a snapped selection into a concrete :class:`Candidate`, or ``None`` if invalid.

    Pure-fluid critical properties and triple pressures are evaluated with the HEOS
    backend (equation-of-state pure properties that are available on any machine); mixture
    critical properties go through :func:`orc_bo.thermo.critical_properties`, which prefers
    REFPROP and falls back to mixing rules.
    """
    fluid1 = fluids[j1]
    fluid2 = fluids[j2] if j2 is not None else None

    try:
        if mode == "pure" or fluid2 is None:
            tc, pc = thermo.pure_critical_properties(fluid1, "HEOS")
            ptriple = thermo.triple_pressure(fluid1, "HEOS")
            wf = fluid1
            x_onehot = onehot[j1].clone()
        else:
            tc, pc = thermo.critical_properties(fluid1, fluid2, x1, config.thermo)
            ptriple = min(
                thermo.triple_pressure(fluid1, "HEOS"),
                thermo.triple_pressure(fluid2, "HEOS"),
            )
            wf = make_refprop_mixture_string(fluid1, fluid2, x1)
            x_onehot = x1 * onehot[j1] + (1.0 - x1) * onehot[j2]
    except thermo.ThermoError as exc:
        logger.warning("Property lookup failed for %s/%s x1=%.3f: %s", fluid1, fluid2, x1, exc)
        return None

    if not np.isfinite(tc) or not np.isfinite(pc):
        logger.warning("Non-finite properties for %s/%s x1=%.3f", fluid1, fluid2, x1)
        return None

    name = format_mixture_name(fluid1, fluid2 if mode != "pure" else None, x1)
    return Candidate(j1, j2, x1, fluid1, fluid2, name, wf, tc, pc, ptriple, x_onehot)


class RunWriter:
    """Incrementally write per-evaluation results, sequence, and summary files.

    Files are flushed after every row so partial results survive interruption and remain
    safe under parallel multi-seed runs.
    """

    def __init__(self, outdir: Path) -> None:
        outdir.mkdir(parents=True, exist_ok=True)
        self.outdir = outdir
        self._results = open(outdir / "scbo_results.csv", "w", newline="", encoding="utf-8")
        self._sequence = open(outdir / "sequence.csv", "w", newline="", encoding="utf-8")
        self._summary = open(outdir / "summary.txt", "w", encoding="utf-8")
        self._results_writer = csv.DictWriter(self._results, fieldnames=RESULT_FIELDS)
        self._sequence_writer = csv.writer(self._sequence)
        self._results_writer.writeheader()
        self._sequence_writer.writerow(["order", "mixture"])
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
