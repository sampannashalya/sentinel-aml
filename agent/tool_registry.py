from __future__ import annotations

from collections.abc import Iterable

from pydantic import BaseModel, ConfigDict, Field

from agent.tool_contracts import PlaceholderTool, ToolContract, ToolMetadata
from detection import FanInDetector, FanOutDetector, VelocityDetector
from tools.feature_engineering import FeatureEngineeringTool


class ToolRegistry:
    def __init__(self, tools: Iterable[ToolContract] | None = None) -> None:
        self._tools: dict[str, ToolContract] = {}
        for tool in tools or []:
            self.register(tool)

    def register(self, tool: ToolContract, replace: bool = False) -> None:
        name = tool.metadata.name
        if name in self._tools and not replace:
            raise ValueError(f"Tool '{name}' is already registered.")
        self._tools[name] = tool

    def get(self, tool_name: str) -> ToolContract:
        try:
            return self._tools[tool_name]
        except KeyError as exc:
            raise KeyError(f"Tool '{tool_name}' is not registered.") from exc

    def exists(self, tool_name: str) -> bool:
        return tool_name in self._tools

    def list_tools(self) -> list[ToolMetadata]:
        return [self._tools[name].metadata for name in sorted(self._tools)]


def build_default_tool_registry() -> ToolRegistry:
    tools = [
        PlaceholderTool("dataset_profiler", "Profile the dataset and report overall shape and quality."),
        PlaceholderTool("eda", "Run exploratory analysis and summarize patterns."),
        PlaceholderTool("transaction_filter", "Filter transactions by query constraints and scope."),
        PlaceholderTool("aggregation", "Aggregate transactions into query-specific summaries."),
        PlaceholderTool("threshold_rule", "Evaluate threshold-based suspicious activity rules."),
        PlaceholderTool("customer_lookup", "Retrieve customer-scoped records and profile information."),
        FeatureEngineeringTool(),
        PlaceholderTool("structuring_detector", "Detect structuring-oriented suspicious patterns."),
        PlaceholderTool("smurfing_detector", "Detect smurfing-oriented suspicious patterns."),
        FanOutDetector(),
        FanInDetector(),
        VelocityDetector(),
        PlaceholderTool("behavior_deviation_detector", "Detect deviations from historical behavior."),
        PlaceholderTool("anomaly_detector", "Score transactions or customers using anomaly signals."),
        PlaceholderTool("risk_scoring", "Combine signals into an explainable risk score."),
        PlaceholderTool("explanation", "Turn evidence into human-readable reasoning."),
        PlaceholderTool("risk_lookup", "Retrieve existing risk evidence for a customer."),
        PlaceholderTool("result_filtering", "Filter results into the requested output slice."),
    ]
    return ToolRegistry(tools)
