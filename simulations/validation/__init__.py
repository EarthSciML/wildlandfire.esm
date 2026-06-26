"""Camp Fire validation harness (campaign bead E2).

A *soft oracle* that scores a simulated level-set fire run against observed
Camp Fire data, computing three metric families and emitting a report:

* **burned-area agreement** — final-footprint area ratio + spatial overlap
  (IoU / Dice / precision / recall) versus MTBS + the last perimeter;
* **perimeter over time / spread rate** — burned-area-vs-time error and mean
  spread rate versus the NIFC/GeoMAC daily perimeter progression;
* **ignition match** — ignition location distance + timing offset versus the
  VIIRS/MODIS first active-fire detection.

Typical use::

    from simulations.validation import LevelSetRun, ObservedReference, validate

    run = LevelSetRun.from_simulate_result(result, dx=2000.0, t0=window_start)
    ref = ObservedReference.from_netcdf("camp_fire_observed.nc")
    report = validate(run, ref)
    report.write(json_path="report.json", markdown_path="report.md")

The data model is pure numpy; ``netCDF4`` is only imported when reading a
reference from disk. Run ``python -m simulations.validation --demo`` for a
self-contained example, or point it at a saved run + reference.
"""
from __future__ import annotations

from .data import LevelSetRun, ObservedReference
from .harness import (
    demo_report,
    synthetic_reference,
    synthetic_run,
    validate,
)
from .report import MetricVerdict, ValidationConfig, ValidationReport

__all__ = [
    "LevelSetRun",
    "ObservedReference",
    "ValidationConfig",
    "ValidationReport",
    "MetricVerdict",
    "validate",
    "demo_report",
    "synthetic_run",
    "synthetic_reference",
]
