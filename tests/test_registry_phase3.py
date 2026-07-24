import pytest

from agent.tool_contracts import PlaceholderTool
from agent.tool_registry import ToolRegistry, build_default_tool_registry


def test_registry_registers_and_retrieves_tools() -> None:
    registry = ToolRegistry()
    tool = PlaceholderTool("sample_tool", "Sample description")

    registry.register(tool)

    assert registry.exists("sample_tool") is True
    assert registry.get("sample_tool").metadata.name == "sample_tool"
    assert [metadata.name for metadata in registry.list_tools()] == ["sample_tool"]


def test_duplicate_tool_registration_is_handled() -> None:
    registry = ToolRegistry()
    tool = PlaceholderTool("sample_tool", "Sample description")

    registry.register(tool)

    with pytest.raises(ValueError, match="already registered"):
        registry.register(tool)


def test_default_registry_contains_expected_placeholder_tools() -> None:
    registry = build_default_tool_registry()

    assert registry.exists("transaction_filter")
    assert registry.exists("risk_scoring")
    assert registry.get("feature_engineering").metadata.availability.value == "implemented"
    assert registry.get("risk_scoring").metadata.availability.value == "placeholder"