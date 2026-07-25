from __future__ import annotations

from datetime import datetime, timedelta

from detection import DetectionEvidence
from investigation import InvestigationReportBuilder, InvestigationReportBuilderTool
from risk import RiskAssessment, RiskLevel, RiskScoreBreakdown
from agent.tool_registry import build_default_tool_registry


def _evidence(
    detector_name: str,
    typology: str,
    *,
    start_offset_minutes: int,
    transaction_count: int,
    total_amount: float,
    evidence_strength: float,
    severity: str,
    transaction_references: list[str],
) -> DetectionEvidence:
    base = datetime(2022, 9, 1, 0, 0)
    return DetectionEvidence(
        detector_name=detector_name,
        typology=typology,
        primary_account_id="ACCT-1",
        involved_account_ids=["ACCT-1", "B", "C", "D"],
        entity_ids=["ENT-1"],
        start_time=base + timedelta(minutes=start_offset_minutes),
        end_time=base + timedelta(minutes=start_offset_minutes + 20),
        transaction_count=transaction_count,
        total_amount=total_amount,
        evidence_strength=evidence_strength,
        severity=severity,  # type: ignore[arg-type]
        reasons=[f"{detector_name} flagged {typology} behavior."],
        detector_parameters={"window_hours": 24},
        transaction_references=transaction_references,
        metadata={"note": detector_name},
    )


def _assessment(
    *,
    risk_score: float,
    risk_level: RiskLevel,
    typologies: list[str],
    evidence: list[DetectionEvidence],
    breakdown: RiskScoreBreakdown,
    metadata: dict[str, object] | None = None,
) -> RiskAssessment:
    base = datetime(2022, 9, 1, 0, 0)
    return RiskAssessment(
        primary_account_id="ACCT-1",
        risk_score=risk_score,
        risk_level=risk_level,
        evidence_count=len(evidence),
        typologies_detected=typologies,
        detector_names=[item.detector_name for item in evidence],
        involved_account_ids=["ACCT-1", "B", "C", "D"],
        entity_ids=["ENT-1"],
        assessment_start_time=base,
        assessment_end_time=base + timedelta(hours=2),
        total_suspicious_amount=sum(item.total_amount for item in evidence),
        total_suspicious_transactions=sum(item.transaction_count for item in evidence),
        contributing_evidence=evidence,
        score_breakdown=breakdown,
        reasons=["Aggregated suspicious evidence for account ACCT-1."],
        metadata=metadata or {},
    )


def _builder():
    return InvestigationReportBuilder()


def test_low_risk_report() -> None:
    assessment = _assessment(
        risk_score=18.0,
        risk_level=RiskLevel.LOW,
        typologies=["FAN-OUT"],
        evidence=[
            _evidence("fan_out_detector", "FAN-OUT", start_offset_minutes=0, transaction_count=3, total_amount=300.0, evidence_strength=0.3, severity="low", transaction_references=["tx-1", "tx-2", "tx-3"])
        ],
        breakdown=RiskScoreBreakdown(
            evidence_strength_score=8.0,
            severity_score=4.0,
            typology_diversity_score=0.0,
            repeated_evidence_score=0.0,
            activity_magnitude_score=6.0,
            total_before_clamp=18.0,
            total_after_clamp=18.0,
            component_reasons=["strength", "severity", "magnitude"],
        ),
    )

    report = _builder().build(assessment, generated_at=datetime(2022, 9, 2, 12, 0))

    assert report.risk_level == "LOW"
    assert "routine monitoring" in " ".join(report.recommended_actions).lower()
    assert report.compliance_disclaimer


def test_medium_risk_report() -> None:
    assessment = _assessment(
        risk_score=35.0,
        risk_level=RiskLevel.MEDIUM,
        typologies=["FAN-IN", "VELOCITY"],
        evidence=[
            _evidence("fan_in_detector", "FAN-IN", start_offset_minutes=0, transaction_count=3, total_amount=300.0, evidence_strength=0.4, severity="medium", transaction_references=["tx-1", "tx-2", "tx-3"]),
            _evidence("velocity_detector", "VELOCITY", start_offset_minutes=30, transaction_count=4, total_amount=500.0, evidence_strength=0.5, severity="low", transaction_references=[]),
        ],
        breakdown=RiskScoreBreakdown(
            evidence_strength_score=12.0,
            severity_score=8.0,
            typology_diversity_score=4.0,
            repeated_evidence_score=3.0,
            activity_magnitude_score=8.0,
            total_before_clamp=35.0,
            total_after_clamp=35.0,
            component_reasons=["strength", "severity", "diversity", "repeat", "magnitude"],
        ),
    )

    report = _builder().build(assessment, generated_at=datetime(2022, 9, 2, 12, 0))

    assert report.risk_level == "MEDIUM"
    assert any("enhanced monitoring" in action.lower() for action in report.recommended_actions)


def test_high_risk_report() -> None:
    assessment = _assessment(
        risk_score=68.0,
        risk_level=RiskLevel.HIGH,
        typologies=["CYCLE", "FAN-OUT", "VELOCITY"],
        evidence=[
            _evidence("cycle_detector", "CYCLE", start_offset_minutes=0, transaction_count=4, total_amount=400.0, evidence_strength=0.8, severity="high", transaction_references=["tx-1", "tx-2", "tx-3", "tx-4"]),
            _evidence("fan_out_detector", "FAN-OUT", start_offset_minutes=60, transaction_count=3, total_amount=300.0, evidence_strength=0.7, severity="medium", transaction_references=["tx-5", "tx-6", "tx-7"]),
            _evidence("velocity_detector", "VELOCITY", start_offset_minutes=120, transaction_count=5, total_amount=700.0, evidence_strength=0.6, severity="medium", transaction_references=[]),
        ],
        breakdown=RiskScoreBreakdown(
            evidence_strength_score=20.0,
            severity_score=16.0,
            typology_diversity_score=12.0,
            repeated_evidence_score=8.0,
            activity_magnitude_score=12.0,
            total_before_clamp=68.0,
            total_after_clamp=68.0,
            component_reasons=["strength", "severity", "diversity", "repeat", "magnitude"],
        ),
    )

    report = _builder().build(assessment, generated_at=datetime(2022, 9, 2, 12, 0))

    assert report.risk_level == "HIGH"
    assert any("prioritize" in action.lower() for action in report.recommended_actions)


def test_critical_risk_report() -> None:
    assessment = _assessment(
        risk_score=91.0,
        risk_level=RiskLevel.CRITICAL,
        typologies=["CYCLE", "FAN-OUT", "VELOCITY", "SCATTER-GATHER"],
        evidence=[
            _evidence("cycle_detector", "CYCLE", start_offset_minutes=0, transaction_count=5, total_amount=500.0, evidence_strength=1.0, severity="high", transaction_references=["tx-1", "tx-2", "tx-3", "tx-4", "tx-5"]),
            _evidence("scatter_gather_detector", "SCATTER-GATHER", start_offset_minutes=90, transaction_count=5, total_amount=900.0, evidence_strength=0.95, severity="high", transaction_references=["tx-6", "tx-7", "tx-8", "tx-9", "tx-10"]),
        ],
        breakdown=RiskScoreBreakdown(
            evidence_strength_score=28.0,
            severity_score=16.0,
            typology_diversity_score=16.0,
            repeated_evidence_score=16.0,
            activity_magnitude_score=15.0,
            total_before_clamp=91.0,
            total_after_clamp=91.0,
            component_reasons=["strength", "severity", "diversity", "repeat", "magnitude"],
        ),
    )

    report = _builder().build(assessment, generated_at=datetime(2022, 9, 2, 12, 0))

    assert report.risk_level == "CRITICAL"
    assert any("immediate" in action.lower() for action in report.recommended_actions)


def test_deterministic_report_id() -> None:
    assessment = _assessment(
        risk_score=68.0,
        risk_level=RiskLevel.HIGH,
        typologies=["CYCLE", "FAN-OUT", "VELOCITY"],
        evidence=[
            _evidence("cycle_detector", "CYCLE", start_offset_minutes=0, transaction_count=4, total_amount=400.0, evidence_strength=0.8, severity="high", transaction_references=["tx-1", "tx-2", "tx-3", "tx-4"]),
        ],
        breakdown=RiskScoreBreakdown(
            evidence_strength_score=20.0,
            severity_score=16.0,
            typology_diversity_score=12.0,
            repeated_evidence_score=8.0,
            activity_magnitude_score=12.0,
            total_before_clamp=68.0,
            total_after_clamp=68.0,
            component_reasons=["strength"],
        ),
    )

    builder = _builder()
    report_a = builder.build(assessment, generated_at=datetime(2022, 9, 2, 12, 0))
    report_b = builder.build(assessment, generated_at=datetime(2022, 9, 3, 12, 0))

    assert report_a.report_id == report_b.report_id


def test_deterministic_substantive_content() -> None:
    assessment = _assessment(
        risk_score=35.0,
        risk_level=RiskLevel.MEDIUM,
        typologies=["FAN-IN", "VELOCITY"],
        evidence=[
            _evidence("fan_in_detector", "FAN-IN", start_offset_minutes=0, transaction_count=3, total_amount=300.0, evidence_strength=0.4, severity="medium", transaction_references=["tx-1", "tx-2", "tx-3"]),
            _evidence("velocity_detector", "VELOCITY", start_offset_minutes=30, transaction_count=4, total_amount=500.0, evidence_strength=0.5, severity="low", transaction_references=[]),
        ],
        breakdown=RiskScoreBreakdown(
            evidence_strength_score=12.0,
            severity_score=8.0,
            typology_diversity_score=4.0,
            repeated_evidence_score=3.0,
            activity_magnitude_score=8.0,
            total_before_clamp=35.0,
            total_after_clamp=35.0,
            component_reasons=["strength", "severity", "diversity", "repeat", "magnitude"],
        ),
    )

    builder = _builder()
    report_a = builder.build(assessment, generated_at=datetime(2022, 9, 2, 12, 0))
    report_b = builder.build(assessment, generated_at=datetime(2022, 9, 2, 12, 0))

    assert report_a.model_dump() == report_b.model_dump()


def test_executive_summary_populated_and_typologies_preserved() -> None:
    assessment = _assessment(
        risk_score=68.0,
        risk_level=RiskLevel.HIGH,
        typologies=["CYCLE", "FAN-OUT", "VELOCITY"],
        evidence=[
            _evidence("cycle_detector", "CYCLE", start_offset_minutes=0, transaction_count=4, total_amount=400.0, evidence_strength=0.8, severity="high", transaction_references=["tx-1", "tx-2", "tx-3", "tx-4"]),
        ],
        breakdown=RiskScoreBreakdown(
            evidence_strength_score=20.0,
            severity_score=16.0,
            typology_diversity_score=12.0,
            repeated_evidence_score=8.0,
            activity_magnitude_score=12.0,
            total_before_clamp=68.0,
            total_after_clamp=68.0,
            component_reasons=["strength"],
        ),
    )

    report = _builder().build(assessment, generated_at=datetime(2022, 9, 2, 12, 0))

    assert report.executive_summary
    assert report.typologies_detected == ["CYCLE", "FAN-OUT", "VELOCITY"]
    assert report.involved_account_ids == ["ACCT-1", "B", "C", "D"]
    assert report.entity_ids == ["ENT-1"]


def test_key_findings_and_summaries_are_bounded_and_deterministic() -> None:
    assessment = _assessment(
        risk_score=91.0,
        risk_level=RiskLevel.CRITICAL,
        typologies=["CYCLE", "FAN-OUT", "VELOCITY", "SCATTER-GATHER"],
        evidence=[
            _evidence("cycle_detector", "CYCLE", start_offset_minutes=0, transaction_count=5, total_amount=500.0, evidence_strength=1.0, severity="high", transaction_references=["tx-1", "tx-2", "tx-3", "tx-4", "tx-5"]),
            _evidence("fan_out_detector", "FAN-OUT", start_offset_minutes=10, transaction_count=3, total_amount=300.0, evidence_strength=0.7, severity="medium", transaction_references=["tx-6", "tx-7", "tx-8"]),
            _evidence("velocity_detector", "VELOCITY", start_offset_minutes=20, transaction_count=5, total_amount=700.0, evidence_strength=0.6, severity="medium", transaction_references=[]),
            _evidence("scatter_gather_detector", "SCATTER-GATHER", start_offset_minutes=30, transaction_count=4, total_amount=900.0, evidence_strength=0.9, severity="high", transaction_references=["tx-9", "tx-10", "tx-11", "tx-12"]),
        ],
        breakdown=RiskScoreBreakdown(
            evidence_strength_score=28.0,
            severity_score=16.0,
            typology_diversity_score=16.0,
            repeated_evidence_score=16.0,
            activity_magnitude_score=15.0,
            total_before_clamp=91.0,
            total_after_clamp=91.0,
            component_reasons=["strength", "severity", "diversity", "repeat", "magnitude"],
        ),
    )

    report = _builder().build(assessment, generated_at=datetime(2022, 9, 2, 12, 0))

    assert len(report.key_findings) <= 8
    assert report.evidence_summary[0].detector in {"cycle_detector", "scatter_gather_detector"}
    assert report.timeline == sorted(report.timeline, key=lambda item: (item.timestamp or datetime.min, item.end_time or datetime.min, item.typology, item.detector))


def test_score_explanation_reflects_breakdown_and_is_not_recomputed() -> None:
    assessment = _assessment(
        risk_score=68.0,
        risk_level=RiskLevel.HIGH,
        typologies=["CYCLE", "FAN-OUT", "VELOCITY"],
        evidence=[
            _evidence("cycle_detector", "CYCLE", start_offset_minutes=0, transaction_count=4, total_amount=400.0, evidence_strength=0.8, severity="high", transaction_references=["tx-1", "tx-2", "tx-3", "tx-4"]),
        ],
        breakdown=RiskScoreBreakdown(
            evidence_strength_score=20.0,
            severity_score=16.0,
            typology_diversity_score=12.0,
            repeated_evidence_score=8.0,
            activity_magnitude_score=12.0,
            total_before_clamp=68.0,
            total_after_clamp=68.0,
            component_reasons=["custom strength", "custom severity"],
        ),
    )

    report = _builder().build(assessment, generated_at=datetime(2022, 9, 2, 12, 0))

    assert report.score_explanation == ["custom strength", "custom severity"]
    assert report.risk_score == 68.0


def test_no_automatic_accusation_or_sar_instruction() -> None:
    assessment = _assessment(
        risk_score=35.0,
        risk_level=RiskLevel.MEDIUM,
        typologies=["FAN-IN"],
        evidence=[
            _evidence("fan_in_detector", "FAN-IN", start_offset_minutes=0, transaction_count=3, total_amount=300.0, evidence_strength=0.4, severity="medium", transaction_references=["tx-1", "tx-2", "tx-3"]),
        ],
        breakdown=RiskScoreBreakdown(
            evidence_strength_score=12.0,
            severity_score=8.0,
            typology_diversity_score=4.0,
            repeated_evidence_score=3.0,
            activity_magnitude_score=8.0,
            total_before_clamp=35.0,
            total_after_clamp=35.0,
            component_reasons=["strength"],
        ),
    )

    report = _builder().build(assessment, generated_at=datetime(2022, 9, 2, 12, 0))

    combined = " ".join(report.recommended_actions + [report.executive_summary, report.compliance_disclaimer]).lower()
    assert "money launderer" not in combined
    assert "sar" not in combined or "file a sar" not in combined


def test_limitations_and_label_sanitization() -> None:
    assessment = _assessment(
        risk_score=35.0,
        risk_level=RiskLevel.MEDIUM,
        typologies=["FAN-IN"],
        evidence=[
            _evidence("fan_in_detector", "FAN-IN", start_offset_minutes=0, transaction_count=3, total_amount=300.0, evidence_strength=0.4, severity="medium", transaction_references=["tx-1", "tx-2", "tx-3"]),
        ],
        breakdown=RiskScoreBreakdown(
            evidence_strength_score=12.0,
            severity_score=8.0,
            typology_diversity_score=4.0,
            repeated_evidence_score=3.0,
            activity_magnitude_score=8.0,
            total_before_clamp=35.0,
            total_after_clamp=35.0,
            component_reasons=["strength"],
        ),
        metadata={
            "annotation_typology": "CYCLE",
            "Is Laundering": 1,
            "is_laundering": 1,
            "custom": "value",
        },
    )

    report = _builder().build(assessment, generated_at=datetime(2022, 9, 2, 12, 0))

    assert report.limitations
    assert "annotation_typology" not in str(report.model_dump())
    assert "is_laundering" not in str(report.model_dump())


def test_all_six_typologies_are_representable_and_minimal_assessment_is_safe() -> None:
    evidence = [
        _evidence("fan_out_detector", "FAN-OUT", start_offset_minutes=0, transaction_count=3, total_amount=300.0, evidence_strength=0.4, severity="medium", transaction_references=["tx-1", "tx-2", "tx-3"]),
        _evidence("fan_in_detector", "FAN-IN", start_offset_minutes=10, transaction_count=3, total_amount=250.0, evidence_strength=0.4, severity="medium", transaction_references=["tx-4", "tx-5", "tx-6"]),
        _evidence("velocity_detector", "VELOCITY", start_offset_minutes=20, transaction_count=6, total_amount=600.0, evidence_strength=0.7, severity="high", transaction_references=[]),
        _evidence("cycle_detector", "CYCLE", start_offset_minutes=30, transaction_count=4, total_amount=400.0, evidence_strength=0.8, severity="high", transaction_references=["tx-7", "tx-8", "tx-9", "tx-10"]),
        _evidence("gather_scatter_detector", "GATHER-SCATTER", start_offset_minutes=40, transaction_count=5, total_amount=500.0, evidence_strength=0.6, severity="high", transaction_references=["tx-11", "tx-12", "tx-13", "tx-14", "tx-15"]),
        _evidence("scatter_gather_detector", "SCATTER-GATHER", start_offset_minutes=50, transaction_count=5, total_amount=500.0, evidence_strength=0.6, severity="high", transaction_references=["tx-16", "tx-17", "tx-18", "tx-19", "tx-20"]),
    ]

    assessment = _assessment(
        risk_score=91.0,
        risk_level=RiskLevel.CRITICAL,
        typologies=["FAN-OUT", "FAN-IN", "VELOCITY", "CYCLE", "GATHER-SCATTER", "SCATTER-GATHER"],
        evidence=evidence,
        breakdown=RiskScoreBreakdown(
            evidence_strength_score=28.0,
            severity_score=16.0,
            typology_diversity_score=16.0,
            repeated_evidence_score=16.0,
            activity_magnitude_score=15.0,
            total_before_clamp=91.0,
            total_after_clamp=91.0,
            component_reasons=["strength"],
        ),
    )

    report = _builder().build(assessment, generated_at=datetime(2022, 9, 2, 12, 0))
    minimal = _builder().build(
        _assessment(
            risk_score=0.0,
            risk_level=RiskLevel.LOW,
            typologies=[],
            evidence=[],
            breakdown=RiskScoreBreakdown(component_reasons=["No contributing evidence was supplied."]),
        ),
        generated_at=datetime(2022, 9, 2, 12, 0),
    )

    assert set(report.typologies_detected) == {
        "FAN-OUT",
        "FAN-IN",
        "VELOCITY",
        "CYCLE",
        "GATHER-SCATTER",
        "SCATTER-GATHER",
    }
    assert minimal.evidence_count == 0
    assert minimal.timeline == []
    assert minimal.key_findings == []


def test_registry_entry_works() -> None:
    registry = build_default_tool_registry()

    assert registry.exists("investigation_report")
    assert registry.get("investigation_report").metadata.availability.value == "implemented"


def test_investigation_report_tool_builds_report() -> None:
    assessment = _assessment(
        risk_score=35.0,
        risk_level=RiskLevel.MEDIUM,
        typologies=["FAN-IN"],
        evidence=[
            _evidence("fan_in_detector", "FAN-IN", start_offset_minutes=0, transaction_count=3, total_amount=300.0, evidence_strength=0.4, severity="medium", transaction_references=["tx-1", "tx-2", "tx-3"]),
        ],
        breakdown=RiskScoreBreakdown(
            evidence_strength_score=12.0,
            severity_score=8.0,
            typology_diversity_score=4.0,
            repeated_evidence_score=3.0,
            activity_magnitude_score=8.0,
            total_before_clamp=35.0,
            total_after_clamp=35.0,
            component_reasons=["strength"],
        ),
    )

    result = InvestigationReportBuilderTool().execute(None, {"assessment": assessment, "generated_at": datetime(2022, 9, 2, 12, 0)})

    assert result.status.value == "SUCCESS"
    assert result.data["report"].primary_account_id == "ACCT-1"
