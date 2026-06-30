"""Aggregate multi-seed optimization results into summary statistics and tables.

Reads the ``scbo_results.csv`` files written under ``<root>/**/seed_XXX/`` and computes,
per configuration, the best efficiency found in each seed plus summary statistics (mean,
standard deviation, and a bootstrap confidence interval). Outputs CSV and LaTeX tables.

Per the project's current scope this module contains NO plotting; it produces tabular
artifacts only.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np


@dataclass
class SeedResult:
    """Best efficiency found in a single seed run."""

    config: str
    seed: int
    best_eta: float
    n_evaluations: int


def _read_best_eta(results_csv: Path) -> Optional[tuple[float, int]]:
    """Return ``(best_eta, n_rows)`` from a results CSV, or ``None`` if empty/unreadable."""
    best = -np.inf
    rows = 0
    try:
        with open(results_csv, "r", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                rows += 1
                try:
                    eta = float(row.get("eta") or row.get("eta_best"))
                except (TypeError, ValueError):
                    continue
                best = max(best, eta)
    except OSError:
        return None
    if rows == 0 or not np.isfinite(best):
        return None
    return best, rows


def collect_results(root: Path) -> List[SeedResult]:
    """Walk ``root`` for ``seed_*/scbo_results.csv`` files and gather best efficiencies.

    The configuration label is the seed directory's parent name (e.g. ``onestage_pure``).
    """
    results: List[SeedResult] = []
    for results_csv in sorted(root.rglob("seed_*/scbo_results.csv")):
        seed_dir = results_csv.parent
        config = seed_dir.parent.name or "default"
        try:
            seed = int(seed_dir.name.split("_")[-1])
        except ValueError:
            seed = -1
        parsed = _read_best_eta(results_csv)
        if parsed is not None:
            best_eta, n_rows = parsed
            results.append(SeedResult(config, seed, best_eta, n_rows))
    return results


def _bootstrap_ci(
    values: Sequence[float], n_boot: int = 10_000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float]:
    """Return a percentile bootstrap confidence interval for the mean."""
    arr = np.asarray(values, dtype=float)
    if arr.size < 2:
        return (float(arr.mean()), float(arr.mean())) if arr.size else (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    means = rng.choice(arr, size=(n_boot, arr.size), replace=True).mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


@dataclass
class ConfigSummary:
    """Aggregate statistics for one configuration across seeds."""

    config: str
    n_seeds: int
    mean_best_eta: float
    std_best_eta: float
    ci_low: float
    ci_high: float
    max_best_eta: float


def summarize(results: List[SeedResult]) -> List[ConfigSummary]:
    """Group per-seed results by configuration and compute summary statistics."""
    by_config: Dict[str, List[float]] = {}
    for r in results:
        by_config.setdefault(r.config, []).append(r.best_eta)

    summaries: List[ConfigSummary] = []
    for config, etas in sorted(by_config.items()):
        arr = np.asarray(etas, dtype=float)
        ci_low, ci_high = _bootstrap_ci(arr)
        summaries.append(
            ConfigSummary(
                config=config,
                n_seeds=arr.size,
                mean_best_eta=float(arr.mean()),
                std_best_eta=float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
                ci_low=ci_low,
                ci_high=ci_high,
                max_best_eta=float(arr.max()),
            )
        )
    return summaries


def write_csv(summaries: List[ConfigSummary], path: Path) -> None:
    """Write the summary table as CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["config", "n_seeds", "mean_best_eta", "std_best_eta", "ci_low", "ci_high", "max_best_eta"]
        )
        for s in summaries:
            writer.writerow(
                [s.config, s.n_seeds, f"{s.mean_best_eta:.6f}", f"{s.std_best_eta:.6f}",
                 f"{s.ci_low:.6f}", f"{s.ci_high:.6f}", f"{s.max_best_eta:.6f}"]
            )


def to_latex(summaries: List[ConfigSummary]) -> str:
    """Render the summary table as a LaTeX ``tabular`` (mean +/- std, 95% CI)."""
    lines = [
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Configuration & Seeds & Mean best $\eta$ (95\% CI) & Max $\eta$ \\",
        r"\midrule",
    ]
    for s in summaries:
        lines.append(
            f"{s.config} & {s.n_seeds} & "
            f"{s.mean_best_eta:.4f} $\\pm$ {s.std_best_eta:.4f} "
            f"[{s.ci_low:.4f}, {s.ci_high:.4f}] & {s.max_best_eta:.4f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point for aggregation."""
    parser = argparse.ArgumentParser(description="Aggregate multi-seed ORC-BO results")
    parser.add_argument("root", type=Path, help="Directory containing seed_*/scbo_results.csv")
    parser.add_argument("--out", type=Path, default=None, help="Output CSV path")
    parser.add_argument("--latex", action="store_true", help="Also print a LaTeX table")
    args = parser.parse_args(argv)

    results = collect_results(args.root)
    if not results:
        print(f"No results found under {args.root}")
        return 1
    summaries = summarize(results)

    out = args.out or (args.root / "summary.csv")
    write_csv(summaries, out)
    print(f"Wrote summary: {out}")
    for s in summaries:
        print(f"  {s.config:24s} n={s.n_seeds:3d}  mean={s.mean_best_eta:.4f} "
              f"[{s.ci_low:.4f}, {s.ci_high:.4f}]  max={s.max_best_eta:.4f}")
    if args.latex:
        print("\n" + to_latex(summaries))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
