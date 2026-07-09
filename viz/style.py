"""Shared styling, palette, and data loaders for the orc_bo visualisation suite.

Every figure script imports from here so the four benchmark configurations wear the same
colours, the same recessive chrome, and the same magnitude ramp across the whole suite.

Design (from the data-viz method):
* **Colour = stage.** One-stage = blue, two-stage = aqua — the two algorithmic strategies
  being compared. The pair is validated colourblind-safe (worst adjacent CVD dE 21.6).
* **Mode = layout.** Pure vs mixture is a small-multiple panel or an x-axis group, never a
  colour, so the stage colours stay comparable everywhere.
* **Magnitude = one blue ramp.** Efficiency (eta) uses a single-hue light->dark sequential
  colormap; "near zero" recedes toward the surface.

All hex values are from the method's validated reference palette.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

import matplotlib as mpl
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

# Make the sibling ``benchmarks`` package importable however this script is launched, so
# figures reuse the exact best-eta / bootstrap-CI methodology behind the LaTeX table.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Default locations. DATA_ROOT can be overridden with the ORC_BO_BENCH environment variable
# so a renamed results tree does not need a code edit.
DATA_ROOT = Path(os.environ.get("ORC_BO_BENCH", REPO_ROOT / "bench" / "full2"))
FIG_DIR = Path(__file__).resolve().parent / "figures"

# ---- Palette (validated reference instance) -------------------------------------------
INK = "#0b0b0b"          # primary text
INK_SECONDARY = "#52514e"
MUTED = "#898781"        # axes / ticks / infeasible marks
GRID = "#e1e0d9"         # hairline gridline
BASELINE = "#c3c2b7"     # axis / spine
SURFACE = "#ffffff"      # figure/axes background (report white)

STAGE_COLOR = {"one-stage": "#2a78d6", "two-stage": "#1baf7a"}  # blue, aqua

# Single-hue blue ramp for efficiency magnitude (light -> dark).
_BLUE_STEPS = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SEQ_CMAP = LinearSegmentedColormap.from_list("orc_blue", _BLUE_STEPS)


@dataclass(frozen=True)
class Cfg:
    """One benchmark configuration (a ``bench/full`` subdirectory)."""

    key: str      # directory name, e.g. "twostage_mixture"
    stage: str    # "one-stage" | "two-stage"
    mode: str     # "pure" | "mixture"

    @property
    def label(self) -> str:
        return f"{self.stage} · {self.mode}"

    @property
    def color(self) -> str:
        return STAGE_COLOR[self.stage]


CONFIGS: List[Cfg] = [
    Cfg("onestage_pure", "one-stage", "pure"),
    Cfg("onestage_mixture", "one-stage", "mixture"),
    Cfg("twostage_pure", "two-stage", "pure"),
    Cfg("twostage_mixture", "two-stage", "mixture"),
]
MODES = ["pure", "mixture"]
STAGES = ["one-stage", "two-stage"]


def apply_style() -> None:
    """Install recessive, print-friendly matplotlib defaults for the whole suite."""
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
        "font.size": 10,
        "text.color": INK,
        "axes.edgecolor": BASELINE,
        "axes.linewidth": 1.0,
        "axes.labelcolor": INK_SECONDARY,
        "axes.titlecolor": INK,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK_SECONDARY,
        "ytick.labelcolor": INK_SECONDARY,
        "legend.frameon": False,
        "legend.fontsize": 9,
    })


def load_runs(key: str, root: Path = DATA_ROOT) -> List[pd.DataFrame]:
    """Load every seed's ``scbo_results.csv`` for one configuration (in seed order)."""
    files = sorted((root / key).glob("seed_*/scbo_results.csv"))
    frames = []
    for f in files:
        try:
            df = pd.read_csv(f)
        except (OSError, pd.errors.EmptyDataError):
            continue
        if len(df):
            frames.append(df)
    return frames


def savefig(fig, name: str, figdir: Path = FIG_DIR) -> Path:
    """Save ``fig`` as ``<figdir>/<name>.png`` and return the path."""
    figdir.mkdir(parents=True, exist_ok=True)
    out = figdir / f"{name}.png"
    fig.savefig(out)
    return out
