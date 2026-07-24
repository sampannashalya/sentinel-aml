from __future__ import annotations

import time
from typing import Any

from agent.execution_models import (
    ExecutionContext,
    ExecutionPlan,
    ExecutionTraceEntry,
    OrchestrationResult,
)
from agent.planner import AgentPlanner
from agent.query_schema import QueryRequest
from agent.tool_contracts import ExecutionStatus, ToolResult
from agent.tool_registry import ToolRegistry, build_default_tool_registry


class AgentOrchestrator:
    def __init__(self, planner: AgentPlanner | None = None, registry: ToolRegistry | None = None) -> None:
        self.planner = planner or AgentPlanner()
        self.registry = registry or build_default_tool_registry()

    def run(self, query_request: QueryRequest, dataset_reference: str | None = None) -> OrchestrationResult:
        plan = self.planner.plan(query_request)
        context = ExecutionContext(
            original_query=query_request.raw_query,
            parsed_query=query_request,
            dataset_reference=dataset_reference,
        )
        return self.execute(plan, context)

    def execute(self, plan: ExecutionPlan, context: ExecutionContext) -> OrchestrationResult:
        trace: list[ExecutionTraceEntry] = []
        tool_outputs: dict[str, Any] = {}
        blocked = False

        for step in sorted(plan.steps, key=lambda item: item.order):
            branch_start = time.perf_counter()
            if blocked and step.required:
                elapsed_ms = (time.perf_counter() - branch_start) * 1000
                trace.append(
                    ExecutionTraceEntry(
                        tool_name=step.tool_name,
                        order=step.order,
                        status=ExecutionStatus.SKIPPED,
                        execution_time_ms=elapsed_ms,
                        input_scope=self._summarize_scope(step.parameters),
                        summary="Skipped because a required upstream step failed.",
                        error=None,
                    )
                )
                continue

            if not self.registry.exists(step.tool_name):
                elapsed_ms = (time.perf_counter() - branch_start) * 1000
                trace.append(
                    ExecutionTraceEntry(
                        tool_name=step.tool_name,
                        order=step.order,
                        status=ExecutionStatus.NOT_IMPLEMENTED,
                        execution_time_ms=elapsed_ms,
                        input_scope=self._summarize_scope(step.parameters),
                        summary="Tool is not registered in the current registry.",
                        error=None,
                    )
                )
                continue

            tool = self.registry.get(step.tool_name)
            start = time.perf_counter()
            try:
                result = tool.execute(context, step.parameters)
                elapsed_ms = (time.perf_counter() - start) * 1000
                trace.append(
                    ExecutionTraceEntry(
                        tool_name=step.tool_name,
                        order=step.order,
                        status=result.status,
                        execution_time_ms=elapsed_ms,
                        input_scope=self._summarize_scope(step.parameters),
                        summary=result.summary,
                        error=result.error,
                    )
                )
                if result.status == ExecutionStatus.SUCCESS:
                    if result.data:
                        context.intermediate_results[step.tool_name] = result.data
                        tool_outputs[step.tool_name] = result.data
                elif result.status == ExecutionStatus.FAILED and step.required:
                    blocked = True
            except Exception as exc:
                elapsed_ms = (time.perf_counter() - start) * 1000
                trace.append(
                    ExecutionTraceEntry(
                        tool_name=step.tool_name,
                        order=step.order,
                        status=ExecutionStatus.FAILED,
                        execution_time_ms=elapsed_ms,
                        input_scope=self._summarize_scope(step.parameters),
                        summary="Tool execution raised an exception.",
                        error=str(exc),
                    )
                )
                if step.required:
                    blocked = True

        final_status = self._final_status(trace)
        return OrchestrationResult(
            plan=plan,
            trace=trace,
            context=context,
            final_status=final_status,
            summary=self._build_summary(plan, trace),
            tool_outputs=tool_outputs,
        )

    def _final_status(self, trace: list[ExecutionTraceEntry]) -> ExecutionStatus:
        if any(entry.status == ExecutionStatus.FAILED for entry in trace):
            return ExecutionStatus.FAILED
        if any(entry.status == ExecutionStatus.NOT_IMPLEMENTED for entry in trace):
            return ExecutionStatus.NOT_IMPLEMENTED
        if any(entry.status == ExecutionStatus.SKIPPED for entry in trace):
            return ExecutionStatus.SKIPPED
        return ExecutionStatus.SUCCESS

    def _build_summary(self, plan: ExecutionPlan, trace: list[ExecutionTraceEntry]) -> str:
        status_counts = {status: sum(1 for item in trace if item.status == status) for status in ExecutionStatus}
        return (
            f"Planned {len(plan.steps)} steps for {plan.query_intent.value}; "
            f"executed {status_counts[ExecutionStatus.SUCCESS]} successful tool(s), "
            f"{status_counts[ExecutionStatus.NOT_IMPLEMENTED]} not-implemented tool(s), "
            f"{status_counts[ExecutionStatus.FAILED]} failed tool(s), and "
            f"{status_counts[ExecutionStatus.SKIPPED]} skipped tool(s)."
        )

    def _summarize_scope(self, parameters: dict[str, Any]) -> str:
        if not parameters:
            return "entire_request"
        parts = []
        for key in sorted(parameters):
            parts.append(f"{key}={parameters[key]}")
        return "; ".join(parts)