import time

import pytest

from agent.execution_models import ExecutionPlan, ExecutionContext, PlanStep
from agent.orchestrator import AgentOrchestrator
from agent.planner import AgentPlanner
from agent.query_parser import QueryParser
from agent.tool_contracts import ExecutionStatus, PlaceholderTool, ToolMetadata
from agent.tool_registry import ToolRegistry


class RecordingTool:
    def __init__(self, name: str, delay_seconds: float = 0.0, should_fail: bool = False) -> None:
        self.metadata = ToolMetadata(name=name, description=f"Recording tool for {name}", availability="implemented")
        self.delay_seconds = delay_seconds
        self.should_fail = should_fail

    def execute(self, context: ExecutionContext, parameters: dict[str, object] | None = None):
        sequence = context.metadata.setdefault("sequence", [])
        sequence.append(self.metadata.name)
        if self.should_fail:
            raise RuntimeError(f"{self.metadata.name} failed")
        if self.delay_seconds:
            time.sleep(self.delay_seconds)
        return type(
            "Result",
            (),
            {
                "status": ExecutionStatus.SUCCESS,
                "summary": f"{self.metadata.name} executed",
                "data": {"tool": self.metadata.name},
                "error": None,
            },
        )()


def test_orchestrator_preserves_execution_order() -> None:
    registry = ToolRegistry([RecordingTool("first"), RecordingTool("second")])
    orchestrator = AgentOrchestrator(planner=AgentPlanner(), registry=registry)
    parsed = QueryParser().parse("Show high-risk customers")
    plan = ExecutionPlan(
        query_intent=parsed.intent,
        steps=[
            PlanStep(order=1, tool_name="first", reason="first"),
            PlanStep(order=2, tool_name="second", reason="second"),
        ],
        skipped_tools=[],
        planning_summary="Test plan",
    )
    context = ExecutionContext(original_query=parsed.raw_query, parsed_query=parsed)

    result = orchestrator.execute(plan, context)

    assert context.metadata["sequence"] == ["first", "second"]
    assert [entry.tool_name for entry in result.trace] == ["first", "second"]
    assert [entry.order for entry in result.trace] == [1, 2]


def test_tool_exceptions_generate_failed_trace_entries() -> None:
    registry = ToolRegistry([RecordingTool("failing", should_fail=True)])
    orchestrator = AgentOrchestrator(planner=AgentPlanner(), registry=registry)
    parsed = QueryParser().parse("Show high-risk customers")
    plan = ExecutionPlan(
        query_intent=parsed.intent,
        steps=[PlanStep(order=1, tool_name="failing", reason="boom")],
        skipped_tools=[],
        planning_summary="Test failure path",
    )
    context = ExecutionContext(original_query=parsed.raw_query, parsed_query=parsed)

    result = orchestrator.execute(plan, context)

    assert result.trace[0].status == ExecutionStatus.FAILED
    assert "failed" in (result.trace[0].error or "")


def test_execution_time_is_measured_not_hardcoded() -> None:
    registry = ToolRegistry([RecordingTool("timed", delay_seconds=0.03)])
    orchestrator = AgentOrchestrator(planner=AgentPlanner(), registry=registry)
    parsed = QueryParser().parse("Show high-risk customers")
    plan = ExecutionPlan(
        query_intent=parsed.intent,
        steps=[PlanStep(order=1, tool_name="timed", reason="time it")],
        skipped_tools=[],
        planning_summary="Timing test",
    )
    context = ExecutionContext(original_query=parsed.raw_query, parsed_query=parsed)

    result = orchestrator.execute(plan, context)

    assert result.trace[0].status == ExecutionStatus.SUCCESS
    assert result.trace[0].execution_time_ms > 0


def test_unknown_tool_handling_is_graceful() -> None:
    registry = ToolRegistry()
    orchestrator = AgentOrchestrator(planner=AgentPlanner(), registry=registry)
    parsed = QueryParser().parse("Show high-risk customers")
    plan = ExecutionPlan(
        query_intent=parsed.intent,
        steps=[PlanStep(order=1, tool_name="missing_tool", reason="missing")],
        skipped_tools=[],
        planning_summary="Unknown tool test",
    )
    context = ExecutionContext(original_query=parsed.raw_query, parsed_query=parsed)

    result = orchestrator.execute(plan, context)

    assert result.trace[0].status == ExecutionStatus.NOT_IMPLEMENTED


def test_orchestrator_run_builds_plan_from_query() -> None:
    orchestrator = AgentOrchestrator()
    parsed = QueryParser().parse("Find structuring patterns in the last 30 days")

    result = orchestrator.run(parsed)

    assert [step.tool_name for step in result.plan.steps] == [
        "transaction_filter",
        "feature_engineering",
        "structuring_detector",
        "risk_scoring",
        "explanation",
    ]
    assert result.trace[0].status == ExecutionStatus.NOT_IMPLEMENTED