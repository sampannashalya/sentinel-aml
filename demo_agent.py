from __future__ import annotations

import json
import sys

from agent.execution_models import ExecutionContext
from agent.orchestrator import AgentOrchestrator
from agent.planner import AgentPlanner
from agent.query_parser import QueryParser


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if not args:
        print('Usage: python demo_agent.py "<query>"')
        return 1

    query = " ".join(args)
    parser = QueryParser()
    parsed = parser.parse(query)
    planner = AgentPlanner()
    plan = planner.plan(parsed)
    orchestrator = AgentOrchestrator(planner=planner)
    context = ExecutionContext(original_query=parsed.raw_query, parsed_query=parsed)
    result = orchestrator.execute(plan, context)

    print("QUERY UNDERSTANDING")
    print(f"Intent: {parsed.intent.value}")
    print(f"Pattern: {parsed.aml_pattern.value if parsed.aml_pattern else 'None'}")
    print(f"Customer ID: {parsed.customer_id or 'None'}")
    if parsed.date_range and parsed.date_range.relative_days:
        print(f"Date Range: last {parsed.date_range.relative_days} days")
    else:
        print("Date Range: None")
    print(f"Amount Threshold: {json.dumps(parsed.amount_threshold.model_dump(mode='json')) if parsed.amount_threshold else 'None'}")
    print(f"Transaction Count Threshold: {json.dumps(parsed.transaction_count_threshold.model_dump(mode='json')) if parsed.transaction_count_threshold else 'None'}")
    print()
    print("AGENT PLAN")
    for step in plan.steps:
        print(f"{step.order}. {step.tool_name}")
    print("SKIPPED")
    for tool_name in plan.skipped_tools:
        print(tool_name)
    print()
    print("EXECUTION TRACE")
    for entry in result.trace:
        print(f"{entry.order}. {entry.tool_name} [{entry.status.value}] {entry.execution_time_ms:.2f} ms - {entry.summary}")
        if entry.error:
            print(f"   error: {entry.error}")
    print()
    print("SUMMARY")
    print(result.summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())