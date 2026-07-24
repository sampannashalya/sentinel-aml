from datetime import date

import pytest

from agent.query_parser import QueryParser
from agent.query_schema import AMLPattern, QueryIntent, RequestedOutput


@pytest.fixture()
def parser() -> QueryParser:
    return QueryParser(reference_date=date(2026, 7, 24))


@pytest.mark.parametrize(
    ("query", "intent", "customer_id", "pattern", "requested_output", "amount_value", "count_value"),
    [
        (
            "Analyse this dataset for suspicious activity",
            QueryIntent.suspicious_activity_search,
            None,
            AMLPattern.suspicious_activity,
            RequestedOutput.investigation_summary,
            None,
            None,
        ),
        (
            "Find structuring patterns in the last 30 days",
            QueryIntent.structuring_detection,
            None,
            AMLPattern.structuring,
            RequestedOutput.pattern_summary,
            None,
            None,
        ),
        (
            "Which customers made 10 or more transactions under $10,000?",
            QueryIntent.threshold_analysis,
            None,
            None,
            RequestedOutput.customer_list,
            10000.0,
            10.0,
        ),
        (
            "Is customer CUST_4521 suspicious?",
            QueryIntent.customer_investigation,
            "CUST_4521",
            None,
            RequestedOutput.case_file,
            None,
            None,
        ),
        (
            "Why is customer CUST_4521 high risk?",
            QueryIntent.risk_explanation,
            "CUST_4521",
            None,
            RequestedOutput.explanation,
            None,
            None,
        ),
        (
            "Show high-risk customers",
            QueryIntent.high_risk_search,
            None,
            None,
            RequestedOutput.customer_list,
            None,
            None,
        ),
        (
            "Find customers with unusually high transaction velocity",
            QueryIntent.velocity_analysis,
            None,
            AMLPattern.velocity,
            RequestedOutput.pattern_summary,
            None,
            None,
        ),
    ],
)
def test_parser_supports_demo_queries(
    parser: QueryParser,
    query: str,
    intent: QueryIntent,
    customer_id: str | None,
    pattern: AMLPattern | None,
    requested_output: RequestedOutput,
    amount_value: float | None,
    count_value: float | None,
) -> None:
    parsed = parser.parse(query)

    assert parsed.raw_query == " ".join(query.strip().split())
    assert parsed.intent == intent
    assert parsed.customer_id == customer_id
    assert parsed.aml_pattern == pattern
    assert parsed.requested_output == requested_output

    if query == "Find structuring patterns in the last 30 days":
        assert parsed.date_range is not None
        assert parsed.date_range.relative_days == 30
        assert parsed.date_range.start_date == date(2026, 6, 24)
        assert parsed.date_range.end_date == date(2026, 7, 24)

    if amount_value is None:
        assert parsed.amount_threshold is None
    else:
        assert parsed.amount_threshold is not None
        assert parsed.amount_threshold.value == amount_value
        assert parsed.amount_threshold.operator == "<"

    if count_value is None:
        assert parsed.transaction_count_threshold is None
    else:
        assert parsed.transaction_count_threshold is not None
        assert parsed.transaction_count_threshold.value == count_value
        assert parsed.transaction_count_threshold.operator == ">="


def test_parser_does_not_attach_execution_logic(parser: QueryParser) -> None:
    parsed = parser.parse("Find structuring patterns in the last 30 days")

    assert parsed.intent == QueryIntent.structuring_detection
    assert parsed.amount_threshold is None
    assert parsed.transaction_count_threshold is None
    assert parsed.transaction_type is None
    assert parsed.country is None