from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import pytest
from pydantic import ValidationError

from agent.planner import AgentPlanner
from agent.query_parser import QueryParser
from agent.query_schema import AMLPattern, QueryIntent
from agent.tool_registry import build_default_tool_registry
from detection import DetectionEvidence, FanInDetector, FanOutDetector, IBMPatternAnnotationParser, VelocityDetector


def _transaction_frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    base_time = datetime(2022, 9, 1, 0, 0)
    defaults = {
        "from_bank": "001",
        "to_bank": "002",
        "amount_paid": 100.0,
        "amount_received": 100.0,
        "payment_currency": "USD",
        "receiving_currency": "USD",
        "payment_format": "ACH",
        "is_laundering": 1,
    }
    normalized = []
    for index, row in enumerate(rows):
        merged = {**defaults, **row}
        merged.setdefault("timestamp", base_time + timedelta(minutes=index * 10))
        merged.setdefault("transaction_reference", f"tx-{index}")
        normalized.append(merged)
    return pd.DataFrame(normalized)


def _velocity_features() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "account_id": "FAST",
                "entity_id": "ENT_FAST",
                "transaction_count": 30,
                "total_outgoing_amount": 1000.0,
                "total_incoming_amount": 500.0,
                "max_transactions_in_hour": 18,
                "rapid_gap_count": 12,
                "transactions_per_active_hour": 9.0,
                "is_laundering": 1,
                "laundering_event_count": 4,
            },
            {
                "account_id": "QUIET",
                "entity_id": "ENT_QUIET",
                "transaction_count": 3,
                "total_outgoing_amount": 10.0,
                "total_incoming_amount": 15.0,
                "max_transactions_in_hour": 2,
                "rapid_gap_count": 0,
                "transactions_per_active_hour": 1.0,
                "is_laundering": 0,
                "laundering_event_count": 0,
            },
        ]
    )


def test_detection_evidence_validation() -> None:
    evidence = DetectionEvidence(
        detector_name="fan_out_detector",
        typology="fan out",
        primary_account_id="A",
        transaction_count=3,
        total_amount=300.0,
        evidence_strength=0.8,
        reasons=["Suspicious-pattern evidence only."],
    )

    assert evidence.typology == "FAN-OUT"
    assert evidence.assessment_type == "suspicious_pattern_evidence"

    with pytest.raises(ValidationError):
        DetectionEvidence(
            detector_name="fan_out_detector",
            typology="FAN-OUT",
            primary_account_id="A",
            transaction_count=-1,
        )


def test_fan_out_positive_case() -> None:
    frame = _transaction_frame(
        [
            {"sender_account": "A", "receiver_account": "B"},
            {"sender_account": "A", "receiver_account": "C"},
            {"sender_account": "A", "receiver_account": "D"},
        ]
    )

    evidence = FanOutDetector().detect(
        transaction_frame=frame,
        min_distinct_receivers=3,
        min_transaction_count=3,
        time_window_hours=1,
    )

    assert len(evidence) == 1
    assert evidence[0].primary_account_id == "A"
    assert evidence[0].metadata["receiver_count"] == 3


def test_fan_out_negative_case() -> None:
    frame = _transaction_frame(
        [
            {"sender_account": "A", "receiver_account": "B"},
            {"sender_account": "A", "receiver_account": "B"},
            {"sender_account": "A", "receiver_account": "C"},
        ]
    )

    evidence = FanOutDetector().detect(
        transaction_frame=frame,
        min_distinct_receivers=3,
        min_transaction_count=3,
        time_window_hours=1,
    )

    assert evidence == []


def test_fan_out_configurable_threshold() -> None:
    frame = _transaction_frame(
        [
            {"sender_account": "A", "receiver_account": "B"},
            {"sender_account": "A", "receiver_account": "C"},
            {"sender_account": "A", "receiver_account": "D"},
        ]
    )

    detector = FanOutDetector()

    assert detector.detect(transaction_frame=frame, min_distinct_receivers=4, min_transaction_count=3) == []
    assert detector.detect(transaction_frame=frame, min_distinct_receivers=3, min_transaction_count=3)


def test_fan_in_positive_case() -> None:
    frame = _transaction_frame(
        [
            {"sender_account": "A", "receiver_account": "X"},
            {"sender_account": "B", "receiver_account": "X"},
            {"sender_account": "C", "receiver_account": "X"},
        ]
    )

    evidence = FanInDetector().detect(
        transaction_frame=frame,
        min_distinct_senders=3,
        min_transaction_count=3,
        time_window_hours=1,
    )

    assert len(evidence) == 1
    assert evidence[0].primary_account_id == "X"
    assert evidence[0].metadata["sender_count"] == 3


def test_fan_in_negative_case() -> None:
    frame = _transaction_frame(
        [
            {"sender_account": "A", "receiver_account": "X"},
            {"sender_account": "A", "receiver_account": "X"},
            {"sender_account": "B", "receiver_account": "X"},
        ]
    )

    evidence = FanInDetector().detect(
        transaction_frame=frame,
        min_distinct_senders=3,
        min_transaction_count=3,
        time_window_hours=1,
    )

    assert evidence == []


def test_fan_in_configurable_threshold() -> None:
    frame = _transaction_frame(
        [
            {"sender_account": "A", "receiver_account": "X"},
            {"sender_account": "B", "receiver_account": "X"},
            {"sender_account": "C", "receiver_account": "X"},
        ]
    )

    detector = FanInDetector()

    assert detector.detect(transaction_frame=frame, min_distinct_senders=4, min_transaction_count=3) == []
    assert detector.detect(transaction_frame=frame, min_distinct_senders=3, min_transaction_count=3)


def test_velocity_positive_case() -> None:
    evidence = VelocityDetector().detect(
        feature_frame=_velocity_features(),
        min_max_transactions_in_hour=10,
        min_rapid_gap_count=5,
        min_transactions_per_active_hour=5,
    )

    assert [item.primary_account_id for item in evidence] == ["FAST"]
    assert "busiest hour" in evidence[0].reasons[0]


def test_velocity_negative_case() -> None:
    quiet = _velocity_features()[lambda frame: frame["account_id"] == "QUIET"]

    evidence = VelocityDetector().detect(
        feature_frame=quiet,
        min_max_transactions_in_hour=10,
        min_rapid_gap_count=5,
        min_transactions_per_active_hour=5,
    )

    assert evidence == []


def test_velocity_configurable_threshold() -> None:
    quiet = _velocity_features()[lambda frame: frame["account_id"] == "QUIET"]
    detector = VelocityDetector()

    assert detector.detect(feature_frame=quiet, min_max_transactions_in_hour=3, min_rapid_gap_count=2, min_transactions_per_active_hour=2) == []
    assert detector.detect(feature_frame=quiet, min_max_transactions_in_hour=2, min_rapid_gap_count=2, min_transactions_per_active_hour=2)


def test_evidence_contains_reason_and_detector_parameters() -> None:
    frame = _transaction_frame(
        [
            {"sender_account": "A", "receiver_account": "B"},
            {"sender_account": "A", "receiver_account": "C"},
            {"sender_account": "A", "receiver_account": "D"},
        ]
    )

    evidence = FanOutDetector().detect(transaction_frame=frame, min_distinct_receivers=3, min_transaction_count=3)[0]

    assert evidence.reasons
    assert "configured threshold" in evidence.reasons[0]
    assert evidence.detector_parameters["min_distinct_receivers"] == 3


def test_pattern_parser_parses_fan_out(tmp_path) -> None:
    path = tmp_path / "patterns.txt"
    path.write_text(
        "\n".join(
            [
                "BEGIN LAUNDERING ATTEMPT - FAN-OUT: Max 3-degree Fan-Out",
                "2022/09/01 00:00,001,A,002,B,100.00,US Dollar,100.00,US Dollar,ACH,1",
                "END LAUNDERING ATTEMPT - FAN-OUT",
            ]
        ),
        encoding="utf-8",
    )

    summary = IBMPatternAnnotationParser(path).parse()

    assert summary.attempts[0].typology == "FAN-OUT"
    assert summary.attempts[0].transactions[0]["sender_account"] == "A"


def test_pattern_parser_parses_multiple_attempts(tmp_path) -> None:
    path = tmp_path / "patterns.txt"
    path.write_text(
        "\n".join(
            [
                "BEGIN LAUNDERING ATTEMPT - FAN-OUT: Max 2-degree Fan-Out",
                "2022/09/01 00:00,001,A,002,B,100.00,US Dollar,100.00,US Dollar,ACH,1",
                "END LAUNDERING ATTEMPT - FAN-OUT",
                "BEGIN LAUNDERING ATTEMPT - FAN-IN: Max 2-degree Fan-In",
                "2022/09/01 00:10,001,C,002,X,200.00,US Dollar,200.00,US Dollar,ACH,1",
                "END LAUNDERING ATTEMPT - FAN-IN",
            ]
        ),
        encoding="utf-8",
    )

    summary = IBMPatternAnnotationParser(path).parse()

    assert [attempt.attempt_id for attempt in summary.attempts] == [1, 2]
    assert summary.attempt_counts_by_typology() == {"FAN-IN": 1, "FAN-OUT": 1}


def test_pattern_parser_extracts_involved_accounts(tmp_path) -> None:
    path = tmp_path / "patterns.txt"
    path.write_text(
        "\n".join(
            [
                "BEGIN LAUNDERING ATTEMPT - FAN-IN: Max 2-degree Fan-In",
                "2022/09/01 00:00,001,A,002,X,100.00,US Dollar,100.00,US Dollar,ACH,1",
                "2022/09/01 00:10,001,B,002,X,100.00,US Dollar,100.00,US Dollar,ACH,1",
                "END LAUNDERING ATTEMPT - FAN-IN",
            ]
        ),
        encoding="utf-8",
    )

    summary = IBMPatternAnnotationParser(path).parse()

    assert summary.attempts[0].accounts == ["A", "B", "X"]


def test_pattern_parser_discovers_unique_typologies(tmp_path) -> None:
    path = tmp_path / "patterns.txt"
    path.write_text(
        "\n".join(
            [
                "BEGIN LAUNDERING ATTEMPT - FAN-OUT: Max 2-degree Fan-Out",
                "2022/09/01 00:00,001,A,002,B,100.00,US Dollar,100.00,US Dollar,ACH,1",
                "END LAUNDERING ATTEMPT - FAN-OUT",
                "BEGIN LAUNDERING ATTEMPT - GATHER-SCATTER: Max 3-degree Fan-In",
                "2022/09/01 00:10,001,C,002,X,200.00,US Dollar,200.00,US Dollar,ACH,1",
                "END LAUNDERING ATTEMPT - GATHER-SCATTER",
            ]
        ),
        encoding="utf-8",
    )

    summary = IBMPatternAnnotationParser(path).parse()

    assert summary.unique_typologies == ["FAN-OUT", "GATHER-SCATTER"]


def test_laundering_label_is_not_detector_evidence_or_input() -> None:
    frame = _transaction_frame(
        [
            {"sender_account": "A", "receiver_account": "B", "is_laundering": 1},
            {"sender_account": "A", "receiver_account": "C", "is_laundering": 1},
            {"sender_account": "A", "receiver_account": "D", "is_laundering": 1},
        ]
    )

    evidence = FanOutDetector().detect(transaction_frame=frame, min_distinct_receivers=3, min_transaction_count=3)[0]
    serialized = evidence.model_dump()

    assert "is_laundering" not in str(serialized)
    assert "laundering_event_count" not in str(VelocityDetector().detect(feature_frame=_velocity_features())[0].model_dump())


def test_pattern_annotations_never_become_detector_inputs(tmp_path) -> None:
    path = tmp_path / "patterns.txt"
    path.write_text(
        "\n".join(
            [
                "BEGIN LAUNDERING ATTEMPT - FAN-OUT: Max 3-degree Fan-Out",
                "2022/09/01 00:00,001,A,002,B,100.00,US Dollar,100.00,US Dollar,ACH,1",
                "2022/09/01 00:10,001,A,002,C,100.00,US Dollar,100.00,US Dollar,ACH,1",
                "2022/09/01 00:20,001,A,002,D,100.00,US Dollar,100.00,US Dollar,ACH,1",
                "END LAUNDERING ATTEMPT - FAN-OUT",
            ]
        ),
        encoding="utf-8",
    )
    attempt = IBMPatternAnnotationParser(path).parse().attempts[0]
    frame = pd.DataFrame(attempt.transactions).rename(
        columns={
            "timestamp": "timestamp",
            "sender_account": "sender_account",
            "receiver_account": "receiver_account",
            "amount_paid": "amount_paid",
        }
    )
    frame["annotation_typology"] = attempt.typology

    evidence = FanOutDetector().detect(transaction_frame=frame, min_distinct_receivers=3, min_transaction_count=3)[0]

    assert "annotation_typology" not in str(evidence.model_dump())
    assert evidence.detector_parameters == {
        "min_distinct_receivers": 3,
        "time_window_hours": 24,
        "min_transaction_count": 3,
    }


def test_planner_routes_fan_out_query_correctly() -> None:
    parsed = QueryParser().parse("Find fan-out laundering patterns")
    plan = AgentPlanner().plan(parsed)

    assert parsed.intent == QueryIntent.fan_out_detection
    assert parsed.aml_pattern == AMLPattern.fan_out
    assert [step.tool_name for step in plan.steps] == ["fan_out_detector"]


def test_planner_routes_fan_in_query_correctly() -> None:
    parsed = QueryParser().parse("Find accounts receiving money from many different accounts")
    plan = AgentPlanner().plan(parsed)

    assert parsed.intent == QueryIntent.fan_in_detection
    assert parsed.aml_pattern == AMLPattern.fan_in
    assert [step.tool_name for step in plan.steps] == ["fan_in_detector"]


def test_planner_routes_velocity_query_correctly() -> None:
    parsed = QueryParser().parse("Find customers with unusually high transaction velocity")
    plan = AgentPlanner().plan(parsed)

    assert parsed.intent == QueryIntent.velocity_analysis
    assert "velocity_detector" in [step.tool_name for step in plan.steps]


def test_planner_avoids_unrelated_detectors_for_targeted_queries() -> None:
    plan = AgentPlanner().plan(QueryParser().parse("Find fan-out laundering patterns"))
    step_names = [step.tool_name for step in plan.steps]

    assert "fan_in_detector" not in step_names
    assert "velocity_detector" not in step_names
    assert "anomaly_detector" not in step_names


def test_default_registry_wires_phase5a_detectors() -> None:
    registry = build_default_tool_registry()

    assert registry.get("fan_out_detector").metadata.availability.value == "implemented"
    assert registry.get("fan_in_detector").metadata.availability.value == "implemented"
    assert registry.get("velocity_detector").metadata.availability.value == "implemented"
