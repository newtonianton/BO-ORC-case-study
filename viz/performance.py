"""Final-performance figures: mean best efficiency per configuration, and its spread.

Both figures reuse ``benchmarks.aggregate_results`` (the same best-eta definition and 10k
percentile-bootstrap CI behind the LaTeX summary table), so the plots and the table can
never disagree.

* ``final_performance`` - grouped bars of mean best eta with 95% bootstrap CI, one x-group
  per fluid space, one bar per stage.
* ``seed_distribution`` - the per-seed best-eta points (with a box) behind each bar, showing
  robustness/variance rather than just the mean.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from style import CONFIGS, FIG_DIR, MODES, STAGES, STAGE_COLOR, apply_style, savefig

from benchmarks.aggregate_results import ConfigSummary, collect_results, summarize
from style import DATA_ROOT

_OFFSET = {"one-stage": -0.19, "two-stage": 0.19}
_WIDTH = 0.34


def _summaries() -> Dict[str, ConfigSummary]:
    return {s.config: s for s in summarize(collect_results(DATA_ROOT))}


def _best_by_config() -> Dict[str, List[float]]:
    out: Dict[str, List[float]] = {c.key: [] for c in CONFIGS}
    for r in collect_results(DATA_ROOT):
        out.setdefault(r.config, []).append(r.best_eta)
    return out


def _stage_legend() -> List[Patch]:
    return [Patch(facecolor=STAGE_COLOR[s], label=s) for s in STAGES]


def final_performance(figdir: Path) -> Path:
    summaries = _summaries()
    fig, ax = plt.subplots(figsize=(7, 4.4))
    for cfg in CONFIGS:
        s = summaries.get(cfg.key)
        if s is None:
            continue
        x = MODES.index(cfg.mode) + _OFFSET[cfg.stage]
        yerr = [[s.mean_best_eta - s.ci_low], [s.ci_high - s.mean_best_eta]]
        ax.bar(x, s.mean_best_eta, width=_WIDTH, color=cfg.color, zorder=2)
        ax.errorbar(x, s.mean_best_eta, yerr=yerr, fmt="none", ecolor="#0b0b0b",
                    elinewidth=1.2, capsize=4, zorder=3)
        ax.annotate(f"{s.mean_best_eta:.3f}", (x, s.ci_high), textcoords="offset points",
                    xytext=(0, 4), ha="center", fontsize=9, color="#0b0b0b")

    ax.set_xticks(range(len(MODES)))
    ax.set_xticklabels([m.capitalize() for m in MODES])
    ax.set_ylabel("Mean best efficiency  $\\eta$")
    ax.set_title("Best efficiency by configuration  (error bars: 95% bootstrap CI)")
    ax.legend(handles=_stage_legend(), title="Stage")
    ax.margins(x=0.15)
    fig.tight_layout()
    out = savefig(fig, "final_performance", figdir)
    plt.close(fig)
    return out


def seed_distribution(figdir: Path) -> Path:
    best = _best_by_config()
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(7, 4.4))
    for cfg in CONFIGS:
        vals = np.asarray(best.get(cfg.key, []), dtype=float)
        if vals.size == 0:
            continue
        x = MODES.index(cfg.mode) + _OFFSET[cfg.stage]
        bp = ax.boxplot(vals, positions=[x], widths=_WIDTH, patch_artist=True,
                        showfliers=False, medianprops=dict(color="#0b0b0b", linewidth=1.4),
                        whiskerprops=dict(color=cfg.color), capprops=dict(color=cfg.color),
                        boxprops=dict(facecolor=cfg.color, alpha=0.22, edgecolor=cfg.color))
        jitter = (rng.random(vals.size) - 0.5) * (_WIDTH * 0.6)
        ax.scatter(np.full(vals.size, x) + jitter, vals, s=16, color=cfg.color,
                   edgecolor="white", linewidth=0.4, zorder=3)

    ax.set_xticks(range(len(MODES)))
    ax.set_xticklabels([m.capitalize() for m in MODES])
    ax.set_ylabel("Best efficiency  $\\eta$  (one point per seed)")
    ax.set_title("Per-seed best efficiency  (box: median / IQR)")
    ax.legend(handles=_stage_legend(), title="Stage")
    ax.margins(x=0.15)
    fig.tight_layout()
    out = savefig(fig, "seed_distribution", figdir)
    plt.close(fig)
    return out


def main(figdir: Path = FIG_DIR) -> List[Path]:
    apply_style()
    return [final_performance(figdir), seed_distribution(figdir)]


if __name__ == "__main__":
    for p in main():
        print("wrote", p)
