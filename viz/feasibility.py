"""Feasibility figure: how often each method proposes an operable fluid.

For every configuration, the fraction of SCBO evaluations that returned a constraint-feasible
operating point (eta > 0), pooled across seeds. This is the "validity hit-rate" — a method
that spends its budget on inoperable fluids scores low here even if its best design is good.
Grouped by fluid space, one bar per stage.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from style import (CONFIGS, FIG_DIR, MODES, STAGES, STAGE_COLOR, apply_style,
                   load_runs, savefig)

_OFFSET = {"one-stage": -0.19, "two-stage": 0.19}
_WIDTH = 0.34


def _valid_fraction(key: str) -> tuple[float, int, int]:
    frames = load_runs(key)
    if not frames:
        return 0.0, 0, 0
    eta = pd.concat(frames, ignore_index=True)["eta"].to_numpy(dtype=float)
    n_valid = int((eta > 0).sum())
    return (n_valid / eta.size if eta.size else 0.0), n_valid, eta.size


def main(figdir: Path = FIG_DIR) -> Path:
    apply_style()
    fig, ax = plt.subplots(figsize=(7, 4.4))
    for cfg in CONFIGS:
        frac, n_valid, n_total = _valid_fraction(cfg.key)
        x = MODES.index(cfg.mode) + _OFFSET[cfg.stage]
        ax.bar(x, frac, width=_WIDTH, color=cfg.color, zorder=2)
        ax.annotate(f"{frac:.0%}\n{n_valid}/{n_total}", (x, frac), textcoords="offset points",
                    xytext=(0, 4), ha="center", fontsize=8.5, color="#0b0b0b")

    ax.set_xticks(range(len(MODES)))
    ax.set_xticklabels([m.capitalize() for m in MODES])
    ax.set_ylim(0, 1.15)  # headroom so the ~100% value labels clear the title
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0))
    ax.set_ylabel("Operable evaluations  (eta > 0)")
    ax.set_title("Feasibility hit-rate  (share of SCBO evaluations that are operable)", pad=12)
    ax.legend(handles=[Patch(facecolor=STAGE_COLOR[s], label=s) for s in STAGES],
              title="Stage")
    ax.margins(x=0.15)
    fig.tight_layout()
    out = savefig(fig, "feasibility_rate", figdir)
    plt.close(fig)
    return out


if __name__ == "__main__":
    print("wrote", main())
