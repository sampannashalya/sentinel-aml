from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import pytest

from agent.planner import AgentPlanner
from agent.query_parser import QueryParser
from agent.query_schema import AMLPattern, QueryIntent
from agent.tool_registry import build_default_tool_registry
from detection import CycleDetector, GatherScatterDetector, ScatterGatherDetector


def _base_frame(rows: list[dict[str, object]]) -> pd.DataFrame:
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


def _cycle_frame() -> pd.DataFrame:
    return _base_frame(
        [
            {"sender_account": "A", "receiver_account": "B"},
            {"sender_account": "B", "receiver_account": "C"},
            {"sender_account": "C", "receiver_account": "D"},
            {"sender_account": "D", "receiver_account": "A"},
        ]
    )


def _gather_scatter_frame() -> pd.DataFrame:
    return _base_frame(
        [
            {"sender_account": "A", "receiver_account": "X"},
            {"sender_account": "B", "receiver_account": "X"},
            {"sender_account": "C", "receiver_account": "X"},
            {"sender_account": "X", "receiver_account": "Y"},
            {"sender_account": "X", "receiver_account": "Z"},
        ]
    )


def _scatter_gather_frame() -> pd.DataFrame:
    return _base_frame(
        [
            {"sender_account": "A", "receiver_account": "C"},
            {"sender_account": "A", "receiver_account": "D"},
            {"sender_account": "C", "receiver_account": "X"},
            {"sender_account": "D", "receiver_account": "X"},
        ]
    )


def test_cycle_positive_case() -> None:
    evidence = CycleDetector().detect(transaction_frame=_cycle_frame(), min_hops=4, max_hops=4, max_elapsed_hours=2)

    assert len(evidence) == 1
    assert evidence[0].typology == "CYCLE"
    assert evidence[0].primary_account_id == "A"
    assert evidence[0].metadata["cycle_path"] == ["A", "B", "C", "D", "A"]
    assert evidence[0].metadata["hop_count"] == 4
    assert evidence[0].reasons
    assert evidence[0].detector_parameters == {"min_hops": 4, "max_hops": 4, "max_elapsed_hours": 2.0}


def test_cycle_negative_non_cycle() -> None:
    frame = _base_frame(
        [
            {"sender_account": "A", "receiver_account": "B"},
            {"sender_account": "B", "receiver_account": "C"},
            {"sender_account": "C", "receiver_account": "D"},
            {"sender_account": "D", "receiver_account": "E"},
        ]
    )

    assert CycleDetector().detect(transaction_frame=frame, min_hops=4, max_hops=4, max_elapsed_hours=2) == []


def test_cycle_minimum_hop_behavior() -> None:
    frame = _base_frame(
        [
            {"sender_account": "A", "receiver_account": "B"},
            {"sender_account": "B", "receiver_account": "C"},
            {"sender_account": "C", "receiver_account": "A"},
        ]
    )

    assert CycleDetector().detect(transaction_frame=frame, min_hops=4, max_hops=4, max_elapsed_hours=2) == []
    assert CycleDetector().detect(transaction_frame=frame, min_hops=3, max_hops=4, max_elapsed_hours=2)


def test_cycle_maximum_hop_behavior() -> None:
    frame = _base_frame(
        [
            {"sender_account": "A", "receiver_account": "B"},
            {"sender_account": "B", "receiver_account": "C"},
            {"sender_account": "C", "receiver_account": "D"},
            {"sender_account": "D", "receiver_account": "E"},
            {"sender_account": "E", "receiver_account": "A"},
        ]
    )

    assert CycleDetector().detect(transaction_frame=frame, min_hops=4, max_hops=4, max_elapsed_hours=2) == []
    assert CycleDetector().detect(transaction_frame=frame, min_hops=5, max_hops=5, max_elapsed_hours=2)


def test_cycle_temporal_window_rejection() -> None:
    frame = _base_frame(
        [
            {"sender_account": "A", "receiver_account": "B", "timestamp": datetime(2022, 9, 1, 0, 0)},
            {"sender_account": "B", "receiver_account": "C", "timestamp": datetime(2022, 9, 2, 0, 0)},
            {"sender_account": "C", "receiver_account": "D", "timestamp": datetime(2022, 9, 3, 0, 0)},
            {"sender_account": "D", "receiver_account": "A", "timestamp": datetime(2022, 9, 4, 0, 0)},
        ]
    )

    assert CycleDetector().detect(transaction_frame=frame, min_hops=4, max_hops=4, max_elapsed_hours=24) == []


def test_cycle_chronological_ordering() -> None:
    frame = _cycle_frame().sample(frac=1.0, random_state=7).reset_index(drop=True)
    evidence = CycleDetector().detect(transaction_frame=frame, min_hops=4, max_hops=4, max_elapsed_hours=2)

    assert len(evidence) == 1
    assert evidence[0].transaction_references == ["tx-0", "tx-1", "tx-2", "tx-3"]


def test_cycle_ordered_path_appears_in_evidence() -> None:
    evidence = CycleDetector().detect(transaction_frame=_cycle_frame(), min_hops=4, max_hops=4, max_elapsed_hours=2)

    assert evidence[0].metadata["cycle_path"] == ["A", "B", "C", "D", "A"]


def test_gather_scatter_positive_detection() -> None:
    evidence = GatherScatterDetector().detect(
        transaction_frame=_gather_scatter_frame(),
        min_distinct_incoming_senders=3,
        min_distinct_outgoing_destinations=2,
        max_time_window_hours=2,
    )

    assert len(evidence) == 1
    assert evidence[0].primary_account_id == "X"
    assert evidence[0].metadata["central_account"] == "X"
    assert evidence[0].metadata["incoming_senders"] == ["A", "B", "C"]
    assert evidence[0].metadata["outgoing_destinations"] == ["Y", "Z"]
    assert evidence[0].reasons
    assert evidence[0].detector_parameters == {
        "min_distinct_incoming_senders": 3,
        "min_distinct_outgoing_destinations": 2,
        "max_time_window_hours": 2,
    }


def test_gather_scatter_negative_detection() -> None:
    frame = _base_frame(
        [
            {"sender_account": "A", "receiver_account": "X"},
            {"sender_account": "A", "receiver_account": "X"},
            {"sender_account": "X", "receiver_account": "Y"},
        ]
    )

    assert GatherScatterDetector().detect(
        transaction_frame=frame,
        min_distinct_incoming_senders=3,
        min_distinct_outgoing_destinations=1,
        max_time_window_hours=2,
    ) == []


def test_gather_scatter_insufficient_incoming_senders() -> None:
    frame = _base_frame(
        [
            {"sender_account": "A", "receiver_account": "X"},
            {"sender_account": "B", "receiver_account": "X"},
            {"sender_account": "X", "receiver_account": "Y"},
        ]
    )

    assert GatherScatterDetector().detect(
        transaction_frame=frame,
        min_distinct_incoming_senders=3,
        min_distinct_outgoing_destinations=1,
        max_time_window_hours=2,
    ) == []


def test_gather_scatter_insufficient_outgoing_destinations() -> None:
    frame = _base_frame(
        [
            {"sender_account": "A", "receiver_account": "X"},
            {"sender_account": "B", "receiver_account": "X"},
            {"sender_account": "C", "receiver_account": "X"},
            {"sender_account": "X", "receiver_account": "Y"},
            {"sender_account": "X", "receiver_account": "Y"},
        ]
    )

    assert GatherScatterDetector().detect(
        transaction_frame=frame,
        min_distinct_incoming_senders=3,
        min_distinct_outgoing_destinations=2,
        max_time_window_hours=2,
    ) == []


def test_gather_scatter_wrong_temporal_order_rejected() -> None:
    frame = _base_frame(
        [
            {"sender_account": "A", "receiver_account": "X"},
            {"sender_account": "X", "receiver_account": "Y"},
            {"sender_account": "B", "receiver_account": "X"},
            {"sender_account": "C", "receiver_account": "X"},
        ]
    )

    assert GatherScatterDetector().detect(
        transaction_frame=frame,
        min_distinct_incoming_senders=3,
        min_distinct_outgoing_destinations=1,
        max_time_window_hours=2,
    ) == []


def test_gather_scatter_configurable_thresholds() -> None:
    detector = GatherScatterDetector()
    frame = _gather_scatter_frame()

    assert detector.detect(
        transaction_frame=frame,
        min_distinct_incoming_senders=4,
        min_distinct_outgoing_destinations=2,
        max_time_window_hours=2,
    ) == []
    assert detector.detect(
        transaction_frame=frame,
        min_distinct_incoming_senders=3,
        min_distinct_outgoing_destinations=2,
        max_time_window_hours=2,
    )


def test_scatter_gather_positive_detection() -> None:
    evidence = ScatterGatherDetector().detect(
        transaction_frame=_scatter_gather_frame(),
        min_intermediaries=2,
        max_time_window_hours=2,
    )

    assert len(evidence) == 1
    assert evidence[0].primary_account_id == "A"
    assert evidence[0].metadata["origin_account"] == "A"
    assert evidence[0].metadata["intermediate_accounts"] == ["C", "D"]
    assert evidence[0].metadata["common_destination"] == "X"
    assert evidence[0].reasons
    assert evidence[0].detector_parameters == {"min_intermediaries": 2, "max_time_window_hours": 2}


def test_scatter_gather_negative_detection() -> None:
    frame = _base_frame(
        [
            {"sender_account": "A", "receiver_account": "C"},
            {"sender_account": "A", "receiver_account": "D"},
            {"sender_account": "C", "receiver_account": "Y"},
            {"sender_account": "D", "receiver_account": "Z"},
        ]
    )

    assert ScatterGatherDetector().detect(transaction_frame=frame, min_intermediaries=2, max_time_window_hours=2) == []


def test_scatter_gather_insufficient_intermediaries() -> None:
    frame = _base_frame(
        [
            {"sender_account": "A", "receiver_account": "C"},
            {"sender_account": "C", "receiver_account": "X"},
        ]
    )

    assert ScatterGatherDetector().detect(transaction_frame=frame, min_intermediaries=2, max_time_window_hours=2) == []


def test_scatter_gather_no_common_destination() -> None:
    frame = _base_frame(
        [
            {"sender_account": "A", "receiver_account": "C"},
            {"sender_account": "A", "receiver_account": "D"},
            {"sender_account": "C", "receiver_account": "X"},
            {"sender_account": "D", "receiver_account": "Y"},
        ]
    )

    assert ScatterGatherDetector().detect(transaction_frame=frame, min_intermediaries=2, max_time_window_hours=2) == []


def test_scatter_gather_wrong_temporal_order_rejected() -> None:
    frame = _base_frame(
        [
            {"sender_account": "C", "receiver_account": "X"},
            {"sender_account": "D", "receiver_account": "X"},
            {"sender_account": "A", "receiver_account": "C"},
            {"sender_account": "A", "receiver_account": "D"},
        ]
    )

    assert ScatterGatherDetector().detect(transaction_frame=frame, min_intermediaries=2, max_time_window_hours=2) == []


def test_scatter_gather_configurable_thresholds() -> None:
    detector = ScatterGatherDetector()
    frame = _scatter_gather_frame()

    assert detector.detect(transaction_frame=frame, min_intermediaries=3, max_time_window_hours=2) == []
    assert detector.detect(transaction_frame=frame, min_intermediaries=2, max_time_window_hours=2)


@pytest.mark.parametrize(
    ("detector", "frame", "kwargs", "label_column"),
    [
        (CycleDetector(), _cycle_frame(), {"min_hops": 4, "max_hops": 4, "max_elapsed_hours": 2}, "is_laundering"),
        (GatherScatterDetector(), _gather_scatter_frame(), {"min_distinct_incoming_senders": 3, "min_distinct_outgoing_destinations": 2, "max_time_window_hours": 2}, "Is Laundering"),
        (ScatterGatherDetector(), _scatter_gather_frame(), {"min_intermediaries": 2, "max_time_window_hours": 2}, "annotation_typology"),
    ],
)
def test_label_columns_do_not_influence_detector_decisions(detector, frame: pd.DataFrame, kwargs: dict[str, object], label_column: str) -> None:
    baseline = detector.detect(transaction_frame=frame, **kwargs)
    with_labels = frame.copy()
    with_labels[label_column] = ["1"] * len(with_labels)
    if label_column == "annotation_typology":
        with_labels[label_column] = ["CYCLE"] * len(with_labels)
    varied = detector.detect(transaction_frame=with_labels, **kwargs)

    assert [item.model_dump() for item in baseline] == [item.model_dump() for item in varied]


@pytest.mark.parametrize(
    ("query", "expected_steps", "expected_pattern"),
    [
        ("Find circular transaction cycles", ["cycle_detector"], AMLPattern.cycle),
        ("Find gather-scatter laundering patterns", ["gather_scatter_detector"], AMLPattern.gather_scatter),
        ("Find scatter-gather transaction patterns", ["scatter_gather_detector"], AMLPattern.scatter_gather),
        ("Find fan-out laundering patterns", ["fan_out_detector"], AMLPattern.fan_out),
        ("Find accounts receiving money from many different accounts", ["fan_in_detector"], AMLPattern.fan_in),
        ("Find customers with unusually high transaction velocity", ["feature_engineering", "velocity_detector", "risk_scoring", "explanation"], AMLPattern.velocity),
    ],
)
def test_parser_and_planner_route_targeted_queries_correctly(query: str, expected_steps: list[str], expected_pattern: AMLPattern) -> None:
    parsed = QueryParser().parse(query)
    plan = AgentPlanner().plan(parsed)

    assert parsed.aml_pattern == expected_pattern
    assert [step.tool_name for step in plan.steps] == expected_steps
    for tool_name in expected_steps:
        assert tool_name not in plan.skipped_tools


def test_parser_and_planner_avoid_unrelated_detectors_for_targeted_queries() -> None:
    for query, forbidden in [
        ("Find circular transaction cycles", {"gather_scatter_detector", "scatter_gather_detector", "fan_out_detector", "fan_in_detector", "velocity_detector"}),
        ("Find gather-scatter laundering patterns", {"cycle_detector", "scatter_gather_detector", "fan_out_detector", "fan_in_detector", "velocity_detector"}),
        ("Find scatter-gather transaction patterns", {"cycle_detector", "gather_scatter_detector", "fan_out_detector", "fan_in_detector", "velocity_detector"}),
    ]:
        plan = AgentPlanner().plan(QueryParser().parse(query))
        step_names = {step.tool_name for step in plan.steps}
        assert forbidden.isdisjoint(step_names)


def test_default_registry_wires_phase5b_detectors() -> None:
    registry = build_default_tool_registry()

    assert registry.get("cycle_detector").metadata.availability.value == "implemented"
    assert registry.get("gather_scatter_detector").metadata.availability.value == "implemented"
    assert registry.get("scatter_gather_detector").metadata.availability.value == "implemented"
