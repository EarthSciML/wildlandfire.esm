"""Command-line entry point for the Camp Fire validation harness.

Two modes:

* ``--demo`` builds a synthetic expanding-disk fire and a slightly perturbed
  observed reference, then validates one against the other — a self-contained
  example needing no external data (this is how the harness is exercised before
  the live E1 reference and E3 run exist).

* ``--run RUN.npz --reference REF.nc`` validates a real saved
  :class:`LevelSetRun` against an observed-reference NetCDF (the E3 wiring).

Examples::

    python -m simulations.validation --demo --markdown report.md --json report.json
    python -m simulations.validation --run camp_fire_run.npz \\
        --reference camp_fire_observed.nc --json report.json
"""
from __future__ import annotations

import argparse
import sys
from typing import Optional, Sequence

from .data import LevelSetRun, ObservedReference
from .harness import demo_report, validate
from .report import ValidationConfig


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m simulations.validation",
        description="Validate a simulated Camp Fire level-set run against observed data.",
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--demo", action="store_true", help="run on synthetic built-in data")
    src.add_argument("--run", metavar="RUN.npz", help="saved LevelSetRun (.npz)")
    p.add_argument("--reference", metavar="REF.nc", help="observed-reference NetCDF (with --run)")
    p.add_argument("--json", metavar="PATH", help="write the report as JSON")
    p.add_argument("--markdown", metavar="PATH", help="write the report as Markdown")
    p.add_argument(
        "--fraction-threshold",
        type=float,
        default=ValidationConfig.fraction_threshold,
        help="observed burned-fraction threshold (default: %(default)s)",
    )
    p.add_argument(
        "--levelset-threshold",
        type=float,
        default=ValidationConfig.levelset_threshold,
        help="simulated psi burned threshold (default: %(default)s)",
    )
    p.add_argument(
        "--quiet", action="store_true", help="do not print the Markdown report to stdout"
    )
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    config = ValidationConfig(
        levelset_threshold=args.levelset_threshold,
        fraction_threshold=args.fraction_threshold,
    )

    if args.demo:
        report = demo_report(config)
    else:
        if not args.reference:
            print("error: --reference is required with --run", file=sys.stderr)
            return 2
        run = LevelSetRun.load_npz(args.run)
        reference = ObservedReference.from_netcdf(args.reference)
        report = validate(run, reference, config)

    written = report.write(json_path=args.json, markdown_path=args.markdown)
    if not args.quiet:
        print(report.to_markdown())
    for kind, path in written.items():
        print(f"[validation] wrote {kind}: {path}", file=sys.stderr)
    # Soft oracle: a flagged ("warn") report is not a process failure.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
