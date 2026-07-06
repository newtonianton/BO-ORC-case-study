"""Generate the full orc_bo visualisation suite into ``viz/figures/``.

Run from the repository root::

    python viz/make_all.py                 # all figures -> viz/figures/
    python viz/make_all.py --figdir out/   # somewhere else

Each figure suite is also runnable on its own (``python viz/convergence.py`` etc.).
"""
from __future__ import annotations

import argparse
from pathlib import Path

from style import FIG_DIR, apply_style

import convergence
import feasibility
import performance
import property_space

SUITES = [convergence, performance, property_space, feasibility]


def main() -> int:
    parser = argparse.ArgumentParser(description="Render the orc_bo visualisation suite")
    parser.add_argument("--figdir", type=Path, default=FIG_DIR,
                        help="Output directory for all PNGs (default: viz/figures)")
    args = parser.parse_args()

    apply_style()
    print(f"Rendering {len(SUITES)} suites -> {args.figdir}")
    for mod in SUITES:
        result = mod.main(args.figdir)
        for path in (result if isinstance(result, list) else [result]):
            print(f"  [ok] {path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
