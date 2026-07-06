"""Multi-seed benchmark runner for the orc_bo pipelines.

Runs the requested pipeline stage(s) and fluid mode(s) across a range of seeds, each in an
isolated subprocess with ``PYTHONHASHSEED`` set for reproducibility, then aggregates the
per-seed results into summary statistics. Supports a small hyperparameter sweep over
initial-sample and budget values for sensitivity studies.

This replaces the previous per-experiment benchmark wrappers and works for both pure and
mixture modes and for one-stage and two-stage pipelines.

Example
-------
Pure-fluid one-stage, seeds 0-9, 4 workers::

    python -m benchmarks.run_benchmark --stages onestage --modes pure --backend HEOS \
        --seeds 10 --n-init 3 --scbo-budget 10 --outdir bench/pure --workers 4
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import List, Optional, Sequence

try:  # `python -m benchmarks.run_benchmark` (runs with a package context)
    from .aggregate_results import collect_results, summarize, to_latex, write_csv
except ImportError:  # `python benchmarks/run_benchmark.py` (script context: benchmarks/ on sys.path)
    from aggregate_results import collect_results, summarize, to_latex, write_csv


@dataclass(frozen=True)
class Job:
    """A single pipeline invocation (one stage/mode/seed/hyperparameter combination)."""

    stage: str
    mode: str
    seed: int
    n_init: int
    scbo_budget: int
    csv: Optional[str]
    backend: Optional[str]
    config_dir: Path

    def command(self) -> List[str]:
        cmd = [
            sys.executable, "-m", "orc_bo.cli", self.stage,
            "--mode", self.mode,
            "--n-init", str(self.n_init),
            "--scbo-budget", str(self.scbo_budget),
            "--outdir", str(self.config_dir),
            "--log-level", "WARNING",
        ]
        if self.csv:
            cmd += ["--csv", self.csv]
        if self.backend:
            cmd += ["--backend", self.backend]
        return cmd


def _run_job(job: Job) -> tuple[str, int]:
    """Run one job in a subprocess with a per-seed PYTHONHASHSEED; return (label, rc)."""
    env = dict(os.environ, PYTHONHASHSEED=str(job.seed))
    label = f"{job.config_dir.name}/seed_{job.seed:03d}"
    proc = subprocess.run(job.command(), env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(f"[FAIL] {label}\n{proc.stderr[-800:]}\n")
    return label, proc.returncode


def build_jobs(args: argparse.Namespace) -> List[Job]:
    """Expand the requested stages/modes/seeds/sweep into individual jobs."""
    seeds = list(range(args.seeds)) if args.seed_list is None else args.seed_list
    n_inits = args.sweep_n_init or [args.n_init]
    budgets = args.sweep_budget or [args.scbo_budget]
    outroot = Path(args.outdir)

    jobs: List[Job] = []
    for stage, mode, n_init, budget in product(args.stages, args.modes, n_inits, budgets):
        suffix = f"_ni{n_init}_b{budget}" if (args.sweep_n_init or args.sweep_budget) else ""
        config_dir = outroot / f"{stage}_{mode}{suffix}"
        for seed in seeds:
            jobs.append(Job(stage, mode, seed, n_init, budget, args.csv, args.backend, config_dir))
    return jobs


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point for the benchmark runner."""
    parser = argparse.ArgumentParser(description="Multi-seed ORC-BO benchmark runner")
    parser.add_argument("--stages", default="onestage",
                        help="Comma-separated: onestage,twostage")
    parser.add_argument("--modes", default="mixture",
                        help="Comma-separated: pure,mixture")
    parser.add_argument("--seeds", type=int, default=10, help="Run seeds 0..N-1")
    parser.add_argument("--seed-list", type=lambda s: [int(x) for x in s.split(",")],
                        default=None, help="Explicit comma-separated seed list (overrides --seeds)")
    parser.add_argument("--n-init", type=int, default=3)
    parser.add_argument("--scbo-budget", type=int, default=10)
    parser.add_argument("--sweep-n-init", type=lambda s: [int(x) for x in s.split(",")],
                        default=None, help="Sweep over these n-init values")
    parser.add_argument("--sweep-budget", type=lambda s: [int(x) for x in s.split(",")],
                        default=None, help="Sweep over these scbo-budget values")
    parser.add_argument("--csv", default=None, help="Dataset CSV (default: packaged)")
    parser.add_argument("--backend", choices=["REFPROP", "HEOS"], default=None)
    parser.add_argument("--outdir", default="bench")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)

    args.stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    args.modes = [m.strip() for m in args.modes.split(",") if m.strip()]

    jobs = build_jobs(args)
    print(f"Launching {len(jobs)} jobs ({args.workers} workers)...")
    failures = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_run_job, job): job for job in jobs}
        for future in as_completed(futures):
            label, rc = future.result()
            status = "ok" if rc == 0 else "FAIL"
            failures += rc != 0
            print(f"  [{status}] {label}")

    print(f"\nCompleted: {len(jobs) - failures}/{len(jobs)} succeeded")

    root = Path(args.outdir)
    results = collect_results(root)
    if results:
        summaries = summarize(results)
        write_csv(summaries, root / "summary.csv")
        print(f"\nSummary -> {root / 'summary.csv'}")
        for s in summaries:
            print(f"  {s.config:28s} n={s.n_seeds:3d}  mean={s.mean_best_eta:.4f} "
                  f"[{s.ci_low:.4f}, {s.ci_high:.4f}]  max={s.max_best_eta:.4f}")
        print("\n" + to_latex(summaries))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
