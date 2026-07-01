"""Command-line interface for the orc_bo pipelines.

Examples
--------
Pure-fluid one-stage run (HEOS backend, works without REFPROP)::

    python -m orc_bo.cli onestage --mode pure --backend HEOS \
        --n-init 3 --scbo-budget 4 --outdir runs/pure

Mixture one-stage run (requires REFPROP)::

    python -m orc_bo.cli onestage --mode mixture --n-init 3 --scbo-budget 10 \
        --outdir runs/mixture
"""
from __future__ import annotations

import argparse
import random
from dataclasses import replace
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch

from .config import AppConfig, load_config
from .logging_setup import configure_logging, configure_warnings
from .seeding import base_seed


def _seed_everything() -> None:
    seed = base_seed()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _build_config(args: argparse.Namespace) -> AppConfig:
    config = load_config(Path(args.config) if args.config else None)
    if args.backend:
        config = replace(config, thermo=replace(config.thermo, backend=args.backend))
    return config


def _add_common(sub: argparse.ArgumentParser) -> None:
    sub.add_argument("--csv", type=Path, default=None, help="Dataset CSV (default: packaged data)")
    sub.add_argument("--mode", choices=["pure", "mixture"], default="mixture")
    sub.add_argument("--n-init", type=int, default=3, help="Number of initial selections")
    sub.add_argument("--scbo-budget", type=int, default=4, help="BO-loop iterations")
    sub.add_argument("--outdir", type=Path, default=Path("runs/onestage"))
    sub.add_argument("--backend", choices=["REFPROP", "HEOS"], default=None)
    sub.add_argument("--config", type=str, default=None, help="Path to a TOML config file")
    sub.add_argument("--log-level", default="INFO")


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point. Returns a process exit code."""
    parser = argparse.ArgumentParser(prog="orc-bo", description="ORC working-fluid BO")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_one = subparsers.add_parser("onestage", help="One-stage BO in one-hot fluid space")
    _add_common(p_one)

    p_two = subparsers.add_parser("twostage", help="Two-stage property-targeting BO")
    _add_common(p_two)

    args = parser.parse_args(argv)
    configure_logging(args.log_level, force=True)
    configure_warnings(show=str(args.log_level).upper() == "DEBUG")
    _seed_everything()
    config = _build_config(args)
    csv_path = args.csv or config.paths.data_csv

    if args.command == "onestage":
        from .pipelines.onestage import run_onestage

        run_onestage(
            csv_path=csv_path,
            mode=args.mode,
            n_init=args.n_init,
            scbo_budget=args.scbo_budget,
            outdir=args.outdir,
            config=config,
        )
        return 0

    if args.command == "twostage":
        from .pipelines.twostage import run_twostage

        run_twostage(
            csv_path=csv_path,
            mode=args.mode,
            n_init=args.n_init,
            scbo_budget=args.scbo_budget,
            outdir=args.outdir,
            config=config,
        )
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
