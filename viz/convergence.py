"""Convergence figure: best valid efficiency found so far vs SCBO-evaluation count.

For each configuration and seed, the running maximum of ``max(eta, 0)`` (0 = no operable
design yet) is the best valid efficiency discovered up to each expensive SCBO evaluation.
Seeds are aligned by evaluation index and summarised by their median with an inter-quartile
band. Pure and mixture are separate panels (shared y-axis) so the one-stage vs two-stage
comparison is read by colour within each fluid space.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from style import CONFIGS, FIG_DIR, MODES, STAGE_COLOR, apply_style, load_runs, savefig


def _best_so_far_matrix(frames: List) -> np.ndarray:
    """Stack seeds' best-valid-eta-so-far curves, truncated to the common length."""
    curves = []
    for df in frames:
        eta = df["eta"].to_numpy(dtype=float)
        curves.append(np.maximum.accumulate(np.clip(eta, 0.0, None)))
    if not curves:
        return np.empty((0, 0))
    n = min(len(c) for c in curves)
    return np.vstack([c[:n] for c in curves])


def main(figdir: Path = FIG_DIR) -> Path:
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)

    for ax, mode in zip(axes, MODES):
        for cfg in (c for c in CONFIGS if c.mode == mode):
            mat = _best_so_far_matrix(load_runs(cfg.key))
            if mat.size == 0:
                continue
            x = np.arange(1, mat.shape[1] + 1)
            med = np.median(mat, axis=0)
            lo, hi = np.percentile(mat, [25, 75], axis=0)
            color = STAGE_COLOR[cfg.stage]
            ax.fill_between(x, lo, hi, color=color, alpha=0.15, linewidth=0)
            ax.plot(x, med, color=color, linewidth=2.0, label=cfg.stage)
            # Direct end-label (relief for the aqua contrast warning; no color-only identity).
            ax.annotate(f"{cfg.stage}", (x[-1], med[-1]), color=color, fontsize=9,
                        xytext=(4, 0), textcoords="offset points", va="center", weight="bold")

        ax.set_title(f"{mode.capitalize()} fluids")
        ax.set_xlabel("SCBO evaluations")
        ax.margins(x=0.02)
    axes[0].set_ylabel("Best valid efficiency  $\\eta$")

    fig.suptitle("Optimisation convergence  (median across seeds, band = 25–75th pct)",
                 fontsize=12, weight="bold", x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = savefig(fig, "convergence", figdir)
    plt.close(fig)
    return out


if __name__ == "__main__":
    print("wrote", main())
