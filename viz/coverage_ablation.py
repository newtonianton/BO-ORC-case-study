"""Coverage ablation: two-stage best efficiency vs pre-SCBO property coverage.

Plots two-stage best-eta against the pre-SCBO property coverage (`n_property_targets` in
{8, 20, 40}), with the coverage-invariant one-stage result drawn as a horizontal baseline
(+ CI band). Answers empirically whether denser property screening improves final
efficiency, or whether the ~25-evaluation SCBO budget is the binding constraint.

Reads the sweep trees produced by::

    ORC_BO_N_TARGETS=8  ... --outdir bench/cov08
    (default 20)        ... --outdir bench/full3
    ORC_BO_N_TARGETS=40 ... --outdir bench/cov40

Run on its own: ``python viz/coverage_ablation.py``.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from style import (FIG_DIR, MODES, REPO_ROOT, STAGE_COLOR, apply_style, load_runs, savefig)

# Pre-SCBO coverage level (n_property_targets) -> results tree.
COV_ROOTS = {8: REPO_ROOT / "bench" / "cov08",
             20: REPO_ROOT / "bench" / "full3",
             40: REPO_ROOT / "bench" / "cov40"}
BASELINE_ROOT = REPO_ROOT / "bench" / "full3"  # one-stage (coverage-invariant)


def _bests(frames: List) -> np.ndarray:
    """Best-of-run efficiency per seed (max eta over that seed's evaluations)."""
    return np.array([df["eta"].astype(float).max() for df in frames])


def _mean_ci(vals: np.ndarray) -> tuple[float, float, float]:
    """Mean and 95% percentile-bootstrap CI (10k resamples), matching the summary table."""
    rng = np.random.default_rng(0)
    boot = rng.choice(vals, size=(10000, vals.size), replace=True).mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(vals.mean()), float(lo), float(hi)


def main(figdir: Path = FIG_DIR) -> Path:
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharey=True)

    for ax, mode in zip(axes, MODES):
        # Two-stage curve across coverage levels present on disk.
        covs, means, los, his = [], [], [], []
        for cov, root in sorted(COV_ROOTS.items()):
            frames = load_runs(f"twostage_{mode}", root)
            if not frames:
                continue
            m, lo, hi = _mean_ci(_bests(frames))
            covs.append(cov); means.append(m); los.append(lo); his.append(hi)
        if covs:
            covs = np.array(covs, dtype=float)
            yerr = [np.array(means) - np.array(los), np.array(his) - np.array(means)]
            ax.errorbar(covs, means, yerr=yerr, marker="o", markersize=7,
                        color=STAGE_COLOR["two-stage"], linewidth=2.0, capsize=4,
                        label="two-stage", zorder=3)
            ax.set_xticks(covs)

        # One-stage baseline (independent of coverage): horizontal line + CI band.
        base = load_runs(f"onestage_{mode}", BASELINE_ROOT)
        if base and covs.size:
            om, olo, ohi = _mean_ci(_bests(base))
            ax.axhline(om, color=STAGE_COLOR["one-stage"], linestyle="--", linewidth=1.6,
                       label="one-stage (any coverage)", zorder=2)
            ax.fill_between([covs.min(), covs.max()], olo, ohi,
                            color=STAGE_COLOR["one-stage"], alpha=0.12, zorder=1)

        ax.set_title(f"{mode.capitalize()} fluids")
        ax.set_xlabel("pre-SCBO coverage  (n_property_targets)")
        ax.margins(x=0.15)
    axes[0].set_ylabel("Mean best efficiency  $\\eta$")
    axes[0].legend(loc="lower right")

    fig.suptitle("Coverage ablation: two-stage best $\\eta$ vs pre-SCBO property coverage",
                 fontsize=12, weight="bold", x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = savefig(fig, "coverage_ablation", figdir)
    plt.close(fig)
    return out


if __name__ == "__main__":
    print("wrote", main())
