"""Cost-ratio sweep figures: best valid efficiency vs cumulative cost, one line per ratio.

Each lab-scale screen costs ``ratio`` SCBO-equivalents, so at smaller ratios two-stage's
curve starts further left (smaller screening offset) and fits more simulations inside the
same total budget of 20. Plotting the across-seed summary of best-valid-eta-so-far as a step
function of cumulative cost, with one curve per ratio, shows how the screening bill shifts
and stretches the same underlying search. The ratio-invariant one-stage baseline (which does
no standalone screening) is the dashed reference.

Two variants are rendered, differing only in the across-seed statistic:
  ratio_sweep.png       - per-ratio MEDIAN across seeds (robust to outlier / slow seeds)
  ratio_sweep_mean.png  - per-ratio MEAN across seeds   (sensitive to slow/failed seeds; a
                          seed still at the penalty pulls the running mean down)

Cross-tree figure: reads several ``bench/`` trees, so ORC_BO_BENCH does not apply;
output goes to ``viz/figures/ratio_sweep/``.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from style import MODES, REPO_ROOT, STAGE_COLOR, apply_style, savefig

RATIO_TREES = {0.10: "cost01", 0.05: "cost005", 0.02: "cost002", 0.00: "cost000"}
BASELINE_TREE = "cost01"  # one-stage does no standalone screening -> ratio-invariant
# Ordered ramp within the two-stage hue: darker = larger screening-cost ratio.
RATIO_COLOR = {0.00: "#a7e3cc", 0.02: "#5fc79e", 0.05: "#22a06e", 0.10: "#0d6b4a"}
FIGDIR = Path(__file__).resolve().parent / "figures" / "ratio_sweep"


def _frames(tree: str, config: str) -> List[pd.DataFrame]:
    out = []
    for f in sorted((REPO_ROOT / "bench" / tree / config).glob("seed_*/scbo_results.csv")):
        df = pd.read_csv(f)
        if len(df) and "cost" in df.columns:
            out.append(df)
    return out


def _seed_curves(frames: List[pd.DataFrame], grid: np.ndarray) -> np.ndarray:
    """Each seed's best-valid-eta-so-far sampled as a step function on ``grid``."""
    curves = []
    for df in frames:
        cost = df["cost"].to_numpy(dtype=float)
        bsf = np.maximum.accumulate(np.clip(df["eta"].to_numpy(dtype=float), 0.0, None))
        idx = np.searchsorted(cost, grid, side="right") - 1
        curves.append(np.where(idx >= 0, bsf[np.clip(idx, 0, len(bsf) - 1)], 0.0))
    return np.vstack(curves)


def _summary_curve(frames: List[pd.DataFrame], grid: np.ndarray, stat: str) -> np.ndarray:
    """Across-seed ``stat`` ('median' or 'mean') of best-valid-eta-so-far on ``grid``."""
    curves = _seed_curves(frames, grid)
    return np.mean(curves, axis=0) if stat == "mean" else np.median(curves, axis=0)


def main(figdir: Path = FIGDIR, stat: str = "median") -> Path:
    apply_style()
    grid = np.linspace(0.0, 20.6, 260)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharey=True)

    for ax, mode in zip(axes, MODES):
        base = _summary_curve(_frames(BASELINE_TREE, f"onestage_{mode}"), grid, stat)
        ax.plot(grid, base, color=STAGE_COLOR["one-stage"], linewidth=1.8,
                linestyle=(0, (4, 3)), label="one-stage (no screening)")
        for ratio in sorted(RATIO_TREES, reverse=True):  # dark (0.10) drawn first
            curve = _summary_curve(_frames(RATIO_TREES[ratio], f"twostage_{mode}"), grid, stat)
            ax.plot(grid, curve, color=RATIO_COLOR[ratio], linewidth=1.8,
                    label=f"two-stage, ratio {ratio:.2f}")
        ax.set_title(f"{mode.capitalize()} fluids")
        ax.set_xlabel("cumulative cost  (SCBO-equivalent units)")
        ax.set_xlim(0, 20.6)
        ax.margins(x=0.02)
    axes[0].set_ylabel("Best valid efficiency  $\\eta$")
    axes[0].legend(loc="lower right", fontsize=8.5)

    label = "mean" if stat == "mean" else "median"
    fig.suptitle("Best valid efficiency vs cumulative cost across screening-cost ratios  "
                 f"(per-ratio {label} across 20 seeds; total budget 20)",
                 fontsize=12, weight="bold", x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    name = "ratio_sweep" if stat == "median" else "ratio_sweep_mean"
    out = savefig(fig, name, figdir)
    plt.close(fig)
    return out


if __name__ == "__main__":
    print("wrote", main(stat="median"))
    print("wrote", main(stat="mean"))
