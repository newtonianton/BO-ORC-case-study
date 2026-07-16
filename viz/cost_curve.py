"""Cost-curve figure: best valid efficiency vs cumulative cost.

Under the cost-weighted budget each SCBO evaluation costs 1.0 and each standalone lab-scale
property screen costs ``lab_cost`` (two-stage Stage-1 targeting). The ``cost`` column of
``scbo_results.csv`` is the cumulative cost at each evaluation, so a run's best-eta-so-far is a
step function of cumulative cost. This plots that curve, aggregated across seeds (median +
25-75th-pct band), which compares the pipelines at *every* cost level rather than at one budget.

Two-stage's curve starts shifted right: it pays its screening cost (``0.1 * L``) upfront with
no efficiency to show, then rises steeply because the screening pre-selected promising fluids.
One-stage rises from cost ~0. The figure answers: does the pre-screening pay back its head start?

Run on its own: ``python viz/cost_curve.py`` (needs a results tree with a ``cost`` column).
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from style import CONFIGS, FIG_DIR, MODES, STAGE_COLOR, apply_style, load_runs, savefig


def _best_so_far_on_grid(frames: List, grid: np.ndarray) -> np.ndarray:
    """Each seed's best-valid-eta-so-far sampled as a step function on the shared cost grid."""
    curves = []
    for df in frames:
        if "cost" not in df.columns or len(df) == 0:
            continue
        cost = df["cost"].to_numpy(dtype=float)
        bsf = np.maximum.accumulate(np.clip(df["eta"].to_numpy(dtype=float), 0.0, None))
        # value at grid cost c = best-so-far of the last evaluation with cost <= c (else 0).
        idx = np.searchsorted(cost, grid, side="right") - 1
        curves.append(np.where(idx >= 0, bsf[np.clip(idx, 0, len(bsf) - 1)], 0.0))
    return np.vstack(curves) if curves else np.empty((0, grid.size))


def _max_cost() -> float:
    hi = 0.0
    for cfg in CONFIGS:
        for df in load_runs(cfg.key):
            if "cost" in df.columns and len(df):
                hi = max(hi, float(df["cost"].max()))
    return hi or 20.0


def main(figdir: Path = FIG_DIR) -> Path:
    apply_style()
    grid = np.linspace(0.0, _max_cost(), 250)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharey=True)

    for ax, mode in zip(axes, MODES):
        for cfg in (c for c in CONFIGS if c.mode == mode):
            mat = _best_so_far_on_grid(load_runs(cfg.key), grid)
            if mat.size == 0:
                continue
            med = np.median(mat, axis=0)
            lo, hi = np.percentile(mat, [25, 75], axis=0)
            color = STAGE_COLOR[cfg.stage]
            ax.fill_between(grid, lo, hi, color=color, alpha=0.15, linewidth=0)
            ax.plot(grid, med, color=color, linewidth=2.0, label=cfg.stage)
            ax.annotate(cfg.stage, (grid[-1], med[-1]), color=color, fontsize=9,
                        xytext=(4, 0), textcoords="offset points", va="center", weight="bold")
        ax.set_title(f"{mode.capitalize()} fluids")
        ax.set_xlabel("cumulative cost  (SCBO-equivalent units)")
        ax.margins(x=0.02)
    axes[0].set_ylabel("Best valid efficiency  $\\eta$")

    fig.suptitle("Cost efficiency: best $\\eta$ vs cumulative cost  "
                 "(1.0 / SCBO, 0.1 / lab-scale screen; median + IQR)",
                 fontsize=12, weight="bold", x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = savefig(fig, "cost_curve", figdir)
    plt.close(fig)
    return out


if __name__ == "__main__":
    print("wrote", main())
