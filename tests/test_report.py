"""Tests for report rendering: valid JSON, NaN sanitization, verdict logic, IO."""
import json

from simulations.validation import demo_report
from simulations.validation.report import (
    NA,
    OK,
    WARN,
    MetricVerdict,
    ValidationConfig,
    ValidationReport,
)


def _report(verdicts):
    return ValidationReport(
        title="t",
        metadata={"grid": {}, "run_window": {}},
        burned_area={},
        perimeter_timing={},
        ignition={},
        verdicts=verdicts,
        config=ValidationConfig().to_dict(),
    )


def test_overall_status_logic():
    assert _report([MetricVerdict("a", OK, 1.0, "")]).overall_status() == OK
    assert _report([MetricVerdict("a", OK, 1.0, ""),
                    MetricVerdict("b", WARN, 2.0, "")]).overall_status() == WARN
    # only n/a verdicts -> n/a overall
    assert _report([MetricVerdict("a", NA, None, "")]).overall_status() == NA
    # n/a never forces a warn
    assert _report([MetricVerdict("a", OK, 1.0, ""),
                    MetricVerdict("b", NA, None, "")]).overall_status() == OK


def test_json_is_valid_and_sanitizes_non_finite():
    report = _report([MetricVerdict("inf_metric", WARN, float("inf"), ">x"),
                      MetricVerdict("nan_metric", NA, float("nan"), "-")])
    report.burned_area = {"area_ratio": float("inf"), "iou": 0.5, "pct": float("nan")}
    text = report.to_json()
    parsed = json.loads(text)  # raises if NaN/Infinity leaked through
    assert parsed["burned_area"]["area_ratio"] is None
    assert parsed["burned_area"]["pct"] is None
    assert parsed["burned_area"]["iou"] == 0.5
    assert parsed["verdicts"][0]["value"] is None


def test_demo_report_json_round_trips():
    report = demo_report()
    parsed = json.loads(report.to_json())
    assert parsed["overall_status"] == "ok"
    assert "burned_area" in parsed and "perimeter_timing" in parsed
    assert parsed["metadata"]["grid"]["nx"] == 61


def test_markdown_has_all_sections():
    md = demo_report().to_markdown()
    for heading in (
        "# Camp Fire validation",
        "Burned-area agreement",
        "Perimeter over time",
        "Ignition match",
        "soft oracle",
    ):
        assert heading in md


def test_write_emits_both_artifacts(tmp_path):
    report = demo_report()
    written = report.write(
        json_path=tmp_path / "r.json", markdown_path=tmp_path / "r.md"
    )
    assert set(written) == {"json", "markdown"}
    assert json.loads((tmp_path / "r.json").read_text())["title"].startswith("Camp Fire")
    assert (tmp_path / "r.md").read_text().startswith("# Camp Fire")
