from __future__ import annotations

from datetime import datetime, timedelta

from detection import DetectionEvidence
from risk import EvidenceAggregator, RiskLevel, RiskScoringConfig, RiskScorer
from agent.tool_registry import build_default_tool_registry


def _evidence(
    *,
    detector_name: str,
    typology: str,
    primary_account_id: str,
    start_offset_minutes: int,
    transaction_count: int,
    total_amount: float,
    evidence_strength: float,
    severity: str,
    transaction_references: list[str],
    involved_account_ids: list[str] | None = None,
    entity_ids: list[str] | None = None,
    metadata: dict[str, object] | None = None,
) -> DetectionEvidence:
    base_time = datetime(2022, 9, 1, 0, 0)
    return DetectionEvidence(
        detector_name=detector_name,
        typology=typology,
        primary_account_id=primary_account_id,
        involved_account_ids=involved_account_ids or [primary_account_id],
        entity_ids=entity_ids or [],
        start_time=base_time + timedelta(minutes=start_offset_minutes),
        end_time=base_time + timedelta(minutes=start_offset_minutes + 30),
        transaction_count=transaction_count,
        total_amount=total_amount,
        evidence_strength=evidence_strength,
        severity=severity,  # type: ignore[arg-type]
        reasons=[f"{detector_name} flagged suspicious-pattern evidence."],
        detector_parameters={"window_hours": 24},
        transaction_references=transaction_references,
        metadata=metadata or {},
    )


def _score(evidence: list[DetectionEvidence]):
    aggregator = EvidenceAggregator()
    scorer = RiskScorer()
    assessments = aggregator.aggregate(evidence)
    scored = scorer.score(assessments)
    return assessments, scored


def test_no_evidence_returns_no_assessment() -> None:
    assert EvidenceAggregator().aggregate([]) == []
    assert RiskScorer().score([]) == []


def test_single_weak_evidence_scores_low_or_medium() -> None:
    evidence = [
        _evidence(
            detector_name="fan_out_detector",
            typology="FAN-OUT",
            primary_account_id="A",
            start_offset_minutes=0,
            transaction_count=2,
            total_amount=100.0,
            evidence_strength=0.2,
            severity="low",
            transaction_references=["tx-1", "tx-2"],
            involved_account_ids=["A", "B", "C"],
        )
    ]

    _, scored = _score(evidence)

    assert scored[0].risk_score < 25
    assert scored[0].risk_level == RiskLevel.LOW


def test_single_strong_evidence_scores_higher_than_weak_evidence() -> None:
    weak = [
        _evidence(
            detector_name="fan_out_detector",
            typology="FAN-OUT",
            primary_account_id="A",
            start_offset_minutes=0,
            transaction_count=2,
            total_amount=100.0,
            evidence_strength=0.2,
            severity="low",
            transaction_references=["tx-1", "tx-2"],
        )
    ]
    strong = [
        _evidence(
            detector_name="cycle_detector",
            typology="CYCLE",
            primary_account_id="A",
            start_offset_minutes=10,
            transaction_count=6,
            total_amount=2000.0,
            evidence_strength=0.95,
            severity="high",
            transaction_references=["tx-3", "tx-4", "tx-5", "tx-6", "tx-7", "tx-8"],
            involved_account_ids=["A", "B", "C", "D"],
        )
    ]

    _, weak_scored = _score(weak)
    _, strong_scored = _score(strong)

    assert strong_scored[0].risk_score > weak_scored[0].risk_score
    assert strong_scored[0].risk_level in {RiskLevel.HIGH, RiskLevel.CRITICAL}


def test_multiple_evidence_from_same_detector_increases_score() -> None:
    single = [
        _evidence(
            detector_name="fan_in_detector",
            typology="FAN-IN",
            primary_account_id="X",
            start_offset_minutes=0,
            transaction_count=3,
            total_amount=300.0,
            evidence_strength=0.5,
            severity="medium",
            transaction_references=["tx-1", "tx-2", "tx-3"],
        )
    ]
    multiple = single + [
        _evidence(
            detector_name="fan_in_detector",
            typology="FAN-IN",
            primary_account_id="X",
            start_offset_minutes=60,
            transaction_count=3,
            total_amount=300.0,
            evidence_strength=0.5,
            severity="medium",
            transaction_references=["tx-4", "tx-5", "tx-6"],
        )
    ]

    _, scored_single = _score(single)
    _, scored_multiple = _score(multiple)

    assert scored_multiple[0].evidence_count == 2
    assert scored_multiple[0].risk_score > scored_single[0].risk_score


def test_multiple_independent_typologies_raise_score() -> None:
    evidence = [
        _evidence(
            detector_name="fan_out_detector",
            typology="FAN-OUT",
            primary_account_id="A",
            start_offset_minutes=0,
            transaction_count=3,
            total_amount=250.0,
            evidence_strength=0.4,
            severity="medium",
            transaction_references=["tx-1", "tx-2", "tx-3"],
        ),
        _evidence(
            detector_name="velocity_detector",
            typology="VELOCITY",
            primary_account_id="A",
            start_offset_minutes=30,
            transaction_count=8,
            total_amount=1500.0,
            evidence_strength=0.8,
            severity="high",
            transaction_references=[],
        ),
        _evidence(
            detector_name="cycle_detector",
            typology="CYCLE",
            primary_account_id="A",
            start_offset_minutes=60,
            transaction_count=4,
            total_amount=800.0,
            evidence_strength=0.7,
            severity="high",
            transaction_references=["tx-4", "tx-5", "tx-6", "tx-7"],
            involved_account_ids=["A", "B", "C"],
        ),
    ]

    _, scored = _score(evidence)

    assert scored[0].typologies_detected == ["CYCLE", "FAN-OUT", "VELOCITY"] or scored[0].typologies_detected == ["FAN-OUT", "VELOCITY", "CYCLE"]
    assert scored[0].risk_score >= 50


def test_risk_score_stays_within_bounds() -> None:
    evidence = [
        _evidence(
            detector_name="scatter_gather_detector",
            typology="SCATTER-GATHER",
            primary_account_id="Z",
            start_offset_minutes=0,
            transaction_count=15,
            total_amount=100_000.0,
            evidence_strength=1.0,
            severity="high",
            transaction_references=[f"tx-{index}" for index in range(15)],
        )
    ]

    _, scored = _score(evidence)

    assert 0.0 <= scored[0].risk_score <= 100.0


def test_risk_level_boundaries_are_deterministic() -> None:
    config = RiskScoringConfig()

    assert config.level_for_score(0.0) == RiskLevel.LOW
    assert config.level_for_score(24.0) == RiskLevel.LOW
    assert config.level_for_score(25.0) == RiskLevel.MEDIUM
    assert config.level_for_score(49.999) == RiskLevel.MEDIUM
    assert config.level_for_score(50.0) == RiskLevel.HIGH
    assert config.level_for_score(74.999) == RiskLevel.HIGH
    assert config.level_for_score(75.0) == RiskLevel.CRITICAL


def test_scoring_is_deterministic_for_repeated_runs() -> None:
    evidence = [
        _evidence(
            detector_name="fan_out_detector",
            typology="FAN-OUT",
            primary_account_id="A",
            start_offset_minutes=0,
            transaction_count=4,
            total_amount=400.0,
            evidence_strength=0.6,
            severity="medium",
            transaction_references=["tx-1", "tx-2", "tx-3", "tx-4"],
        )
    ]

    first = _score(evidence)[1]
    second = _score(evidence)[1]

    assert first[0].model_dump() == second[0].model_dump()


def test_aggregation_groups_by_account() -> None:
    evidence = [
        _evidence(
            detector_name="fan_out_detector",
            typology="FAN-OUT",
            primary_account_id="A",
            start_offset_minutes=0,
            transaction_count=2,
            total_amount=200.0,
            evidence_strength=0.4,
            severity="medium",
            transaction_references=["tx-1", "tx-2"],
        ),
        _evidence(
            detector_name="fan_in_detector",
            typology="FAN-IN",
            primary_account_id="B",
            start_offset_minutes=10,
            transaction_count=2,
            total_amount=150.0,
            evidence_strength=0.3,
            severity="low",
            transaction_references=["tx-3", "tx-4"],
        ),
    ]

    assessments = EvidenceAggregator().aggregate(evidence)

    assert [assessment.primary_account_id for assessment in assessments] == ["A", "B"]


def test_union_fields_are_preserved() -> None:
    evidence = [
        _evidence(
            detector_name="fan_out_detector",
            typology="FAN-OUT",
            primary_account_id="A",
            start_offset_minutes=0,
            transaction_count=2,
            total_amount=200.0,
            evidence_strength=0.4,
            severity="medium",
            transaction_references=["tx-1", "tx-2"],
            involved_account_ids=["A", "B"],
            entity_ids=["E1"],
        ),
        _evidence(
            detector_name="velocity_detector",
            typology="VELOCITY",
            primary_account_id="A",
            start_offset_minutes=5,
            transaction_count=1,
            total_amount=50.0,
            evidence_strength=0.5,
            severity="low",
            transaction_references=[],
            involved_account_ids=["A", "C"],
            entity_ids=["E2"],
        ),
    ]

    assessment = EvidenceAggregator().aggregate(evidence)[0]

    assert assessment.involved_account_ids == ["A", "B", "C"]
    assert assessment.entity_ids == ["E1", "E2"]
    assert assessment.typologies_detected == ["FAN-OUT", "VELOCITY"]


def test_assessment_time_range_is_derived_from_evidence() -> None:
    evidence = [
        _evidence(
            detector_name="fan_out_detector",
            typology="FAN-OUT",
            primary_account_id="A",
            start_offset_minutes=0,
            transaction_count=2,
            total_amount=200.0,
            evidence_strength=0.4,
            severity="medium",
            transaction_references=["tx-1", "tx-2"],
        ),
        _evidence(
            detector_name="cycle_detector",
            typology="CYCLE",
            primary_account_id="A",
            start_offset_minutes=120,
            transaction_count=4,
            total_amount=400.0,
            evidence_strength=0.8,
            severity="high",
            transaction_references=["tx-3", "tx-4", "tx-5", "tx-6"],
        ),
    ]

    assessment = EvidenceAggregator().aggregate(evidence)[0]

    assert assessment.assessment_start_time == datetime(2022, 9, 1, 0, 0)
    assert assessment.assessment_end_time == datetime(2022, 9, 1, 2, 30)


def test_transaction_reference_deduplication_prevents_double_counting() -> None:
    evidence = [
        _evidence(
            detector_name="fan_out_detector",
            typology="FAN-OUT",
            primary_account_id="A",
            start_offset_minutes=0,
            transaction_count=2,
            total_amount=200.0,
            evidence_strength=0.5,
            severity="medium",
            transaction_references=["tx-1", "tx-2"],
        ),
        _evidence(
            detector_name="fan_out_detector",
            typology="FAN-OUT",
            primary_account_id="A",
            start_offset_minutes=0,
            transaction_count=2,
            total_amount=200.0,
            evidence_strength=0.5,
            severity="medium",
            transaction_references=["tx-1", "tx-2"],
        ),
    ]

    assessment = EvidenceAggregator().aggregate(evidence)[0]

    assert assessment.evidence_count == 1
    assert assessment.total_suspicious_transactions == 2
    assert assessment.total_suspicious_amount == 200.0


def test_overlapping_evidence_does_not_obviously_double_count_identical_transactions() -> None:
    evidence = [
        _evidence(
            detector_name="fan_out_detector",
            typology="FAN-OUT",
            primary_account_id="A",
            start_offset_minutes=0,
            transaction_count=3,
            total_amount=300.0,
            evidence_strength=0.5,
            severity="medium",
            transaction_references=["tx-1", "tx-2", "tx-3"],
        ),
        _evidence(
            detector_name="cycle_detector",
            typology="CYCLE",
            primary_account_id="A",
            start_offset_minutes=10,
            transaction_count=3,
            total_amount=300.0,
            evidence_strength=0.6,
            severity="high",
            transaction_references=["tx-2", "tx-3", "tx-4"],
        ),
    ]

    assessment = EvidenceAggregator().aggregate(evidence)[0]

    assert assessment.total_suspicious_transactions == 4
    assert assessment.total_suspicious_amount <= 600.0


def test_reasons_and_score_breakdown_are_populated() -> None:
    evidence = [
        _evidence(
            detector_name="scatter_gather_detector",
            typology="SCATTER-GATHER",
            primary_account_id="A",
            start_offset_minutes=0,
            transaction_count=5,
            total_amount=500.0,
            evidence_strength=0.7,
            severity="high",
            transaction_references=["tx-1", "tx-2", "tx-3", "tx-4", "tx-5"],
        )
    ]

    scored = _score(evidence)[1][0]

    assert scored.reasons
    assert scored.score_breakdown.component_reasons
    assert scored.score_breakdown.total_after_clamp == scored.risk_score


def test_contributing_evidence_remains_traceable() -> None:
    evidence = [
        _evidence(
            detector_name="cycle_detector",
            typology="CYCLE",
            primary_account_id="A",
            start_offset_minutes=0,
            transaction_count=4,
            total_amount=400.0,
            evidence_strength=0.8,
            severity="high",
            transaction_references=["tx-1", "tx-2", "tx-3", "tx-4"],
        )
    ]

    assessment = EvidenceAggregator().aggregate(evidence)[0]

    assert assessment.contributing_evidence[0].detector_name == "cycle_detector"
    assert assessment.contributing_evidence[0].transaction_references == ["tx-1", "tx-2", "tx-3", "tx-4"]


def test_label_and_annotation_fields_cannot_influence_scoring() -> None:
    base = [
        _evidence(
            detector_name="fan_out_detector",
            typology="FAN-OUT",
            primary_account_id="A",
            start_offset_minutes=0,
            transaction_count=3,
            total_amount=300.0,
            evidence_strength=0.5,
            severity="medium",
            transaction_references=["tx-1", "tx-2", "tx-3"],
            metadata={"note": "baseline"},
        )
    ]
    leaked = [
        base[0].model_copy(
            update={
                "metadata": {
                    "note": "baseline",
                    "Is Laundering": 1,
                    "is_laundering": 1,
                    "annotation_typology": "CYCLE",
                    "expected_ground_truth": "yes",
                }
            }
        )
    ]

    base_scored = _score(base)[1][0]
    leaked_scored = _score(leaked)[1][0]

    assert base_scored.model_dump() == leaked_scored.model_dump()


def test_all_existing_detector_typologies_are_consumed_without_errors() -> None:
    evidence = [
        _evidence(
            detector_name="fan_out_detector",
            typology="FAN-OUT",
            primary_account_id="A",
            start_offset_minutes=0,
            transaction_count=3,
            total_amount=300.0,
            evidence_strength=0.4,
            severity="medium",
            transaction_references=["tx-1", "tx-2", "tx-3"],
        ),
        _evidence(
            detector_name="fan_in_detector",
            typology="FAN-IN",
            primary_account_id="A",
            start_offset_minutes=10,
            transaction_count=3,
            total_amount=250.0,
            evidence_strength=0.4,
            severity="medium",
            transaction_references=["tx-4", "tx-5", "tx-6"],
        ),
        _evidence(
            detector_name="velocity_detector",
            typology="VELOCITY",
            primary_account_id="A",
            start_offset_minutes=20,
            transaction_count=6,
            total_amount=600.0,
            evidence_strength=0.7,
            severity="high",
            transaction_references=[],
        ),
        _evidence(
            detector_name="cycle_detector",
            typology="CYCLE",
            primary_account_id="A",
            start_offset_minutes=30,
            transaction_count=4,
            total_amount=400.0,
            evidence_strength=0.8,
            severity="high",
            transaction_references=["tx-7", "tx-8", "tx-9", "tx-10"],
        ),
        _evidence(
            detector_name="gather_scatter_detector",
            typology="GATHER-SCATTER",
            primary_account_id="A",
            start_offset_minutes=40,
            transaction_count=5,
            total_amount=500.0,
            evidence_strength=0.6,
            severity="high",
            transaction_references=["tx-11", "tx-12", "tx-13", "tx-14", "tx-15"],
        ),
        _evidence(
            detector_name="scatter_gather_detector",
            typology="SCATTER-GATHER",
            primary_account_id="A",
            start_offset_minutes=50,
            transaction_count=5,
            total_amount=500.0,
            evidence_strength=0.6,
            severity="high",
            transaction_references=["tx-16", "tx-17", "tx-18", "tx-19", "tx-20"],
        ),
    ]

    scored = _score(evidence)[1][0]

    assert set(scored.typologies_detected) == {
        "FAN-OUT",
        "FAN-IN",
        "VELOCITY",
        "CYCLE",
        "GATHER-SCATTER",
        "SCATTER-GATHER",
    }
    assert scored.evidence_count == 6


def test_risk_tools_are_registered() -> None:
    registry = build_default_tool_registry()

    assert registry.exists("evidence_aggregator")
    assert registry.exists("risk_scorer")
    assert registry.get("evidence_aggregator").metadata.availability.value == "implemented"
    assert registry.get("risk_scorer").metadata.availability.value == "implemented"
