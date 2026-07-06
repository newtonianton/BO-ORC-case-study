"""Property-space exploration: where each method searched in (Tc, Pc) space.

Every evaluated fluid (pooled over seeds) is a point at its critical temperature and
pressure, coloured by the efficiency SCBO achieved for it on the single-hue blue ramp.
Infeasible evaluations (no operable point, eta <= 0) are drawn as small muted crosses so
the operable region stands out. One panel per configuration (2x2), shared colour scale.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from style import (CONFIGS, FIG_DIR, MUTED, SEQ_CMAP, apply_style, load_runs, savefig)


def _pooled(key: str) -> pd.DataFrame:
    frames = load_runs(key)
    if not frames:
        return pd.DataFrame(columns=["Tc_K", "Pc_Pa", "eta"])
    return pd.concat(frames, ignore_index=True)


def main(figdir: Path = FIG_DIR) -> Path:
    apply_style()
    pooled = {c.key: _pooled(c.key) for c in CONFIGS}

    valid = np.concatenate([
        df.loc[df["eta"] > 0, "eta"].to_numpy(dtype=float)
        for df in pooled.values() if len(df)
    ]) if any(len(df) for df in pooled.values()) else np.array([0.0, 1.0])
    vmin, vmax = float(valid.min()), float(valid.max())

    fig, axes = plt.subplots(2, 2, figsize=(9.5, 8), sharex=True, sharey=True)
    scat = None
    for ax, cfg in zip(axes.ravel(), CONFIGS):
        df = pooled[cfg.key]
        if len(df):
            tc = df["Tc_K"].to_numpy(dtype=float)
            pc = df["Pc_Pa"].to_numpy(dtype=float) / 1e6  # MPa
            eta = df["eta"].to_numpy(dtype=float)
            bad = eta <= 0
            ax.scatter(tc[bad], pc[bad], s=14, marker="x", color=MUTED, alpha=0.5,
                       linewidth=0.8, zorder=1)
            scat = ax.scatter(tc[~bad], pc[~bad], c=eta[~bad], cmap=SEQ_CMAP,
                              vmin=vmin, vmax=vmax, s=26, edgecolor="white",
                              linewidth=0.3, zorder=2)
        n_valid = int((df["eta"] > 0).sum()) if len(df) else 0
        ax.set_title(f"{cfg.label}   ({n_valid} operable)")

    for ax in axes[-1]:
        ax.set_xlabel("Critical temperature  $T_c$  [K]")
    for ax in axes[:, 0]:
        ax.set_ylabel("Critical pressure  $P_c$  [MPa]")

    if scat is not None:
        cbar = fig.colorbar(scat, ax=axes, fraction=0.046, pad=0.03)
        cbar.set_label("Efficiency  $\\eta$  (operable designs)")
        cbar.outline.set_edgecolor(MUTED)
    fig.suptitle("Property-space exploration  (× = infeasible)", fontsize=12,
                 weight="bold", x=0.02, ha="left")
    out = savefig(fig, "property_space", figdir)
    plt.close(fig)
    return out


if __name__ == "__main__":
    print("wrote", main())
