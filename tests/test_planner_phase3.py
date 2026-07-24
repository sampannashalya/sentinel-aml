from agent.planner import AgentPlanner
from agent.query_parser import QueryParser


def test_broad_analysis_selects_broad_tools() -> None:
    parsed = QueryParser().parse("Analyse this dataset for suspicious activity")
    plan = AgentPlanner().plan(parsed)

    assert plan.query_intent.value == "suspicious_activity_search"
    assert [step.tool_name for step in plan.steps] == [
        "dataset_profiler",
        "eda",
        "feature_engineering",
        "structuring_detector",
        "smurfing_detector",
        "velocity_detector",
        "behavior_deviation_detector",
        "anomaly_detector",
        "risk_scoring",
        "explanation",
    ]


def test_structuring_query_selects_structuring_path_only() -> None:
    parsed = QueryParser().parse("Find structuring patterns in the last 30 days")
    plan = AgentPlanner().plan(parsed)

    assert [step.tool_name for step in plan.steps] == [
        "transaction_filter",
        "feature_engineering",
        "structuring_detector",
        "risk_scoring",
        "explanation",
    ]
    assert "smurfing_detector" in plan.skipped_tools
    assert "eda" in plan.skipped_tools


def test_threshold_query_skips_anomaly_detection() -> None:
    parsed = QueryParser().parse("Which customers made 10 or more transactions under $10,000?")
    plan = AgentPlanner().plan(parsed)

    assert [step.tool_name for step in plan.steps] == ["transaction_filter", "aggregation", "threshold_rule"]
    assert all(step.tool_name != "anomaly_detector" for step in plan.steps)


def test_velocity_query_avoids_structuring_path() -> None:
    parsed = QueryParser().parse("Find customers with unusually high transaction velocity")
    plan = AgentPlanner().plan(parsed)

    assert [step.tool_name for step in plan.steps] == [
        "feature_engineering",
        "velocity_detector",
        "risk_scoring",
        "explanation",
    ]
    assert "structuring_detector" in plan.skipped_tools


def test_customer_investigation_skips_full_eda() -> None:
    parsed = QueryParser().parse("Is customer CUST_4521 suspicious?")
    plan = AgentPlanner().plan(parsed)

    assert [step.tool_name for step in plan.steps] == [
        "customer_lookup",
        "feature_engineering",
        "behavior_deviation_detector",
        "anomaly_detector",
        "risk_scoring",
        "explanation",
    ]
    assert "eda" in plan.skipped_tools


def test_risk_explanation_uses_minimum_tools() -> None:
    parsed = QueryParser().parse("Why is customer CUST_4521 high risk?")
    plan = AgentPlanner().plan(parsed)

    assert [step.tool_name for step in plan.steps] == ["customer_lookup", "risk_lookup", "explanation"]
    assert "risk_scoring" in plan.skipped_tools


def test_different_queries_produce_different_plans() -> None:
    parser = QueryParser()
    planner = AgentPlanner()

    structuring_plan = planner.plan(parser.parse("Find structuring patterns in the last 30 days"))
    velocity_plan = planner.plan(parser.parse("Find customers with unusually high transaction velocity"))

    assert [step.tool_name for step in structuring_plan.steps] != [step.tool_name for step in velocity_plan.steps]