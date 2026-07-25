from __future__ import annotations

from datetime import datetime

from detection import DetectionEvidence
from investigation import InvestigationReportBuilder
from risk import RiskAssessment, RiskLevel, RiskScoreBreakdown
from app import evidence_rows_from_assessment
from ui.components import (
    breakdown_rows,
    evidence_rows,
    format_amount,
    format_amount_compact,
    format_risk_score,
    ibm_dataset_status_label,
    parse_and_plan_query,
    timeline_rows,
)
from ui.demo_data import INVESTIGATION_SOURCE_LABEL, build_demo_bundle, build_demo_evidence, summarize_query_plan


def test_demo_evidence_is_deterministic() -> None:
    first = build_demo_evidence()
    second = build_demo_evidence()

    assert [item.model_dump() for item in first] == [item.model_dump() for item in second]


def test_demo_pipeline_uses_real_backend() -> None:
    bundle = build_demo_bundle(generated_at=datetime(2022, 9, 1, 12, 0))

    assert bundle.is_synthetic is True
    assert bundle.source_label == INVESTIGATION_SOURCE_LABEL
    assert bundle.assessment is not None
    assert bundle.report is not None
    assert bundle.report.risk_score == bundle.assessment.risk_score
    assert bundle.report.report_id


def test_metric_formatting_helpers() -> None:
    assert format_amount(1234.5) == "$1,234.50"
    assert format_risk_score(70.048) == "70.0"
    assert format_amount_compact(50800.0) == "$50.8K"


def test_breakdown_conversion_is_stable() -> None:
    assessment = build_demo_bundle(generated_at=datetime(2022, 9, 1, 12, 0)).assessment
    assert assessment is not None
    rows = breakdown_rows(assessment.score_breakdown)

    assert rows[0]["component"] == "Evidence strength"
    assert rows[-1]["component"] == "Activity magnitude"


def test_timeline_conversion_is_chronological() -> None:
    report = build_demo_bundle(generated_at=datetime(2022, 9, 1, 12, 0)).report
    assert report is not None

    frame = timeline_rows(report)

    assert frame["Start"].is_monotonic_increasing
    assert len(frame) == len(report.timeline)


def test_query_plan_helper_returns_expected_route() -> None:
    preview = parse_and_plan_query("Find circular transaction cycles")

    assert preview["intent"] == "cycle_detection"
    assert preview["aml_pattern"] == "cycle"
    assert preview["tool_names"] == ["cycle_detector"]


def test_query_plan_summary_matches_preview() -> None:
    preview = parse_and_plan_query("Find gather-scatter laundering patterns")
    summary = summarize_query_plan("Find gather-scatter laundering patterns")

    assert summary["intent"] == preview["intent"]
    assert summary["tools"] == preview["tool_names"]


def test_empty_state_handling_is_safe() -> None:
    report = InvestigationReportBuilder().build(
        RiskAssessment(
            primary_account_id="ACCT-EMPTY",
            risk_score=0.0,
            risk_level=RiskLevel.LOW,
            evidence_count=0,
            typologies_detected=[],
            detector_names=[],
            involved_account_ids=["ACCT-EMPTY"],
            entity_ids=[],
            assessment_start_time=None,
            assessment_end_time=None,
            total_suspicious_amount=0.0,
            total_suspicious_transactions=0,
            contributing_evidence=[],
            score_breakdown=RiskScoreBreakdown(component_reasons=["No contributing evidence was supplied."]),
            reasons=[],
            metadata={},
        ),
        generated_at=datetime(2022, 9, 1, 12, 0),
    )

    assert report.evidence_summary == []
    assert report.timeline == []
    assert report.key_findings == []


def test_label_like_metadata_does_not_change_demo_report() -> None:
    assessment = build_demo_bundle(generated_at=datetime(2022, 9, 1, 12, 0)).assessment
    assert assessment is not None
    mutated = assessment.model_copy(
        update={
            "metadata": {
                **assessment.metadata,
                "is_laundering": 1,
                "Is Laundering": 1,
                "laundering_event_count": 99,
                "annotation_typology": "CYCLE",
            }
        }
    )

    builder = InvestigationReportBuilder()
    report_a = builder.build(assessment, generated_at=datetime(2022, 9, 1, 12, 0))
    report_b = builder.build(mutated, generated_at=datetime(2022, 9, 1, 12, 0))

    assert report_a.model_dump() == report_b.model_dump()
    assert "is_laundering" not in str(report_b.model_dump())


def test_report_rendering_helper_uses_assessment_breakdown_not_report_breakdown() -> None:
    bundle = build_demo_bundle(generated_at=datetime(2022, 9, 1, 12, 0))
    assert bundle.assessment is not None
    assert bundle.report is not None

    rows = breakdown_rows(bundle.assessment.score_breakdown)

    assert rows
    assert not hasattr(bundle.report, "score_breakdown")
    assert evidence_rows_from_assessment(bundle.assessment).shape[0] == bundle.assessment.evidence_count


def test_all_typologies_are_representable_in_demo_mode() -> None:
    bundle = build_demo_bundle(["FAN-OUT", "FAN-IN", "VELOCITY", "CYCLE", "GATHER-SCATTER", "SCATTER-GATHER"], generated_at=datetime(2022, 9, 1, 12, 0))

    assert set(bundle.report.typologies_detected) == {
        "FAN-OUT",
        "FAN-IN",
        "VELOCITY",
        "CYCLE",
        "GATHER-SCATTER",
        "SCATTER-GATHER",
    }
    assert bundle.report.evidence_count == 6


def test_synthetic_evidence_is_marked_as_demo_only() -> None:
    bundle = build_demo_bundle(generated_at=datetime(2022, 9, 1, 12, 0))

    assert bundle.is_synthetic is True
    assert bundle.source_label == "Demo / Synthetic Evidence"
    assert all(item.metadata.get("source") == "synthetic_demo" for item in bundle.evidence)


def test_ibm_dataset_status_label_is_privacy_safe() -> None:
    available = ibm_dataset_status_label(True)
    missing = ibm_dataset_status_label(False)

    assert available == "IBM AML dataset: Available"
    assert missing == "IBM AML dataset: Not found"
    assert "C:\\" not in available
    assert "C:\\" not in missing
