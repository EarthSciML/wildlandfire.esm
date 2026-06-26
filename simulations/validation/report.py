"""Validation configuration, per-metric verdicts, and report rendering.

A Camp Fire validation is a *soft oracle*: observed-versus-simulated wildfire
agreement is a research comparison, not a bit-exact check (the campaign plan and
README both say the model "never fully ran" end-to-end). So the thresholds in
:class:`ValidationConfig` are **advisory** — they colour each metric ``ok`` or
``warn`` to guide the eye, and a ``warn`` is a flag for a human to look, not a
hard failure. The numbers themselves are always reported verbatim.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

OK = "ok"
WARN = "warn"
NA = "n/a"


@dataclass
class ValidationConfig:
    """Thresholds for the harness. Masks are hard; verdict bounds are advisory.

    Parameters
    ----------
    levelset_threshold:
        ``psi <= this`` is burned in the simulated field (the front is 0).
    fraction_threshold:
        Observed burned fraction ``>= this`` counts a cell as burned.
    area_ratio_ok:
        ``(low, high)`` band for simulated/observed total burned-area ratio.
    dice_ok:
        Minimum final-footprint Dice overlap considered acceptable. Fire-spread
        footprints overlap poorly even for good models, so the default (0.30) is
        deliberately lenient.
    ignition_distance_ok_m, ignition_time_ok_hours:
        Tolerances for the simulated-vs-observed ignition point and instant.
    """

    levelset_threshold: float = 0.0
    fraction_threshold: float = 0.5
    area_ratio_ok: tuple[float, float] = (0.5, 2.0)
    dice_ok: float = 0.30
    ignition_distance_ok_m: float = 5000.0
    ignition_time_ok_hours: float = 6.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["area_ratio_ok"] = list(self.area_ratio_ok)
        return d


@dataclass
class MetricVerdict:
    """A single advisory pass/flag on one reported quantity."""

    metric: str
    status: str  # OK | WARN | NA
    value: Optional[float]
    criterion: str
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "metric": self.metric,
            "status": self.status,
            "value": _finite_or_none(self.value),
            "criterion": self.criterion,
            "note": self.note,
        }


@dataclass
class ValidationReport:
    """Assembled validation result for one run-vs-reference comparison."""

    title: str
    metadata: dict
    burned_area: dict
    perimeter_timing: dict
    ignition: dict
    verdicts: list[MetricVerdict] = field(default_factory=list)
    config: dict = field(default_factory=dict)

    def overall_status(self) -> str:
        """``warn`` if any non-skipped metric is flagged, else ``ok``."""
        graded = [v.status for v in self.verdicts if v.status != NA]
        if not graded:
            return NA
        return WARN if any(s == WARN for s in graded) else OK

    def to_dict(self) -> dict:
        return _sanitize(
            {
                "title": self.title,
                "overall_status": self.overall_status(),
                "metadata": self.metadata,
                "verdicts": [v.to_dict() for v in self.verdicts],
                "burned_area": self.burned_area,
                "perimeter_timing": self.perimeter_timing,
                "ignition": self.ignition,
                "config": self.config,
            }
        )

    def to_json(self, indent: int = 2) -> str:
        # Non-finite floats are already sanitized to null -> valid JSON.
        return json.dumps(self.to_dict(), indent=indent, allow_nan=False)

    def to_markdown(self) -> str:
        return _render_markdown(self)

    def write(
        self,
        json_path: Optional[str | Path] = None,
        markdown_path: Optional[str | Path] = None,
    ) -> dict:
        written = {}
        if json_path is not None:
            Path(json_path).write_text(self.to_json())
            written["json"] = str(json_path)
        if markdown_path is not None:
            Path(markdown_path).write_text(self.to_markdown())
            written["markdown"] = str(markdown_path)
        return written


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _finite_or_none(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _sanitize(obj: Any) -> Any:
    """Recursively replace non-finite floats with ``None`` for valid JSON."""
    if isinstance(obj, float):
        return _finite_or_none(obj)
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    return obj


def _fmt(value: Any, spec: str = ".3g") -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        if not math.isfinite(value):
            return "n/a"
        return format(value, spec)
    return str(value)


def _status_mark(status: str) -> str:
    return {OK: "OK", WARN: "WARN", NA: "—"}.get(status, status)


def _render_markdown(report: ValidationReport) -> str:
    lines: list[str] = []
    lines.append(f"# {report.title}")
    lines.append("")
    lines.append(
        f"**Overall:** {_status_mark(report.overall_status())} "
        "— *soft oracle; thresholds are advisory, not a hard pass/fail.*"
    )
    lines.append("")

    md = report.metadata
    lines.append("## Run / reference")
    lines.append("")
    lines.append(f"- Reference source: `{md.get('reference_source', 'unspecified')}`")
    grid = md.get("grid", {})
    lines.append(
        f"- Grid: {grid.get('nx')} × {grid.get('ny')} cells, "
        f"cell {_fmt(grid.get('cell_size_m'))} m, "
        f"area {_fmt(grid.get('cell_area_km2'))} km²/cell"
    )
    window = md.get("run_window", {})
    if window.get("t0"):
        lines.append(
            f"- Run window: {window.get('t0')} for {_fmt(window.get('duration_hours'))} h "
            f"({window.get('n_times')} saved slices)"
        )
    lines.append("")

    lines.append("## Verdicts")
    lines.append("")
    lines.append("| Metric | Value | Criterion | Status |")
    lines.append("|---|---|---|---|")
    for v in report.verdicts:
        lines.append(
            f"| {v.metric} | {_fmt(v.value)} | {v.criterion} | {_status_mark(v.status)} |"
        )
    lines.append("")

    ba = report.burned_area
    lines.append("## Burned-area agreement (final footprint vs MTBS + last perimeter)")
    lines.append("")
    lines.append(f"- Simulated burned area: {_fmt(ba.get('area_sim_km2'))} km²")
    lines.append(f"- Observed burned area: {_fmt(ba.get('area_obs_km2'))} km²")
    lines.append(
        f"- Area ratio (sim/obs): {_fmt(ba.get('area_ratio'))} "
        f"(bias {_fmt(ba.get('area_bias_km2'))} km², "
        f"{_fmt(ba.get('area_abs_pct_error'))} % abs)"
    )
    lines.append(
        f"- Overlap: IoU {_fmt(ba.get('iou'))}, Dice {_fmt(ba.get('dice'))}, "
        f"precision {_fmt(ba.get('precision'))}, recall {_fmt(ba.get('recall'))}"
    )
    lines.append("")

    pt = report.perimeter_timing
    lines.append("## Perimeter over time / spread rate (vs NIFC/GeoMAC perimeters)")
    lines.append("")
    if pt.get("status") == "skipped":
        lines.append(f"- *Skipped: {pt.get('reason', 'no timed perimeter data')}.*")
    else:
        lines.append(
            f"- Matched perimeter times: {pt.get('n_matched')} of {pt.get('n_available')}"
        )
        lines.append(
            f"- Burned-area-vs-time RMSE: {_fmt(pt.get('area_rmse_km2'))} km² "
            f"(bias {_fmt(pt.get('area_bias_km2'))} km²)"
        )
        lines.append(
            f"- Mean spread rate: sim {_fmt(pt.get('spread_rate_sim_m_per_s'))} m/s vs "
            f"obs {_fmt(pt.get('spread_rate_obs_m_per_s'))} m/s"
        )
        lines.append(f"- Mean per-time IoU: {_fmt(pt.get('mean_iou'))}")
        per = pt.get("per_time", [])
        if per:
            lines.append("")
            lines.append("| Time | Elapsed (h) | Sim km² | Obs km² | IoU |")
            lines.append("|---|---|---|---|---|")
            for row in per:
                lines.append(
                    f"| {row.get('time')} | {_fmt(row.get('elapsed_hours'))} | "
                    f"{_fmt(row.get('area_sim_km2'))} | {_fmt(row.get('area_obs_km2'))} | "
                    f"{_fmt(row.get('iou'))} |"
                )
    lines.append("")

    ig = report.ignition
    lines.append("## Ignition match (vs VIIRS/MODIS first detection)")
    lines.append("")
    if ig.get("status") == "skipped":
        lines.append(f"- *Skipped: {ig.get('reason', 'no observed ignition')}.*")
    else:
        lines.append(f"- Ignition distance: {_fmt(ig.get('distance_km'))} km")
        if ig.get("time_offset_hours") is not None:
            lines.append(f"- Ignition time offset: {_fmt(ig.get('time_offset_hours'))} h")
        lines.append(
            f"- Sim ignition (x, y): {_fmt(ig.get('sim_xy', [None, None])[0])}, "
            f"{_fmt(ig.get('sim_xy', [None, None])[1])} m"
        )
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append(
        "*Generated by the `simulations.validation` harness "
        "(campfire-e2e E2). Burned region = level-set `psi <= "
        f"{report.config.get('levelset_threshold', 0.0)}`; observed burned = "
        f"fraction >= {report.config.get('fraction_threshold', 0.5)}.*"
    )
    return "\n".join(lines)
