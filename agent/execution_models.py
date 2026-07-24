from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from agent.query_schema import QueryIntent, QueryRequest
from agent.tool_contracts import ExecutionStatus


class PlanStep(BaseModel):
    model_config = ConfigDict(extra="forbid")

    order: int
    tool_name: str
    reason: str
    required: bool = True
    parameters: dict[str, Any] = Field(default_factory=dict)
    dependencies: list[str] = Field(default_factory=list)


class ExecutionPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query_intent: QueryIntent
    steps: list[PlanStep]
    skipped_tools: list[str] = Field(default_factory=list)
    planning_summary: str


class ExecutionTraceEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool_name: str
    order: int
    status: ExecutionStatus
    execution_time_ms: float
    input_scope: str
    summary: str
    error: str | None = None


class ExecutionContext(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    original_query: str
    parsed_query: QueryRequest
    dataset_reference: str | None = None
    intermediate_results: dict[str, Any] = Field(default_factory=dict)
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class OrchestrationResult(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    plan: ExecutionPlan
    trace: list[ExecutionTraceEntry]
    context: ExecutionContext
    final_status: ExecutionStatus
    summary: str
    tool_outputs: dict[str, Any] = Field(default_factory=dict)