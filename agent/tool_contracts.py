from __future__ import annotations

from enum import Enum
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field


class ExecutionStatus(str, Enum):
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"
    NOT_IMPLEMENTED = "NOT_IMPLEMENTED"


class ToolAvailability(str, Enum):
    IMPLEMENTED = "implemented"
    PLACEHOLDER = "placeholder"
    DISABLED = "disabled"


class ToolMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
    availability: ToolAvailability = ToolAvailability.PLACEHOLDER
    input_type: str | None = None
    output_type: str | None = None


class ToolResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: ExecutionStatus
    summary: str
    data: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


@runtime_checkable
class ToolContract(Protocol):
    metadata: ToolMetadata

    def execute(self, context: Any, parameters: dict[str, Any] | None = None) -> ToolResult:
        ...


class PlaceholderTool:
    def __init__(
        self,
        name: str,
        description: str,
        input_type: str | None = None,
        output_type: str | None = None,
    ) -> None:
        self.metadata = ToolMetadata(
            name=name,
            description=description,
            availability=ToolAvailability.PLACEHOLDER,
            input_type=input_type,
            output_type=output_type,
        )

    def execute(self, context: Any, parameters: dict[str, Any] | None = None) -> ToolResult:
        return ToolResult(
            status=ExecutionStatus.NOT_IMPLEMENTED,
            summary=f"{self.metadata.name} is registered as a placeholder for later implementation.",
        )