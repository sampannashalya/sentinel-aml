from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd

from agent.planner import AgentPlanner
from agent.query_parser import QueryParser
from risk import RiskAssessment, RiskScoreBreakdown


def format_amount(value: float) -> str:
    return f"${float(value):,.2f}"


def format_amount_compact(value: float) -> str:
    amount = float(value)
    magnitude = abs(amount)
    if magnitude >= 1_000_000:
        return f"${amount / 1_000_000:.1f}M"
    if magnitude >= 1_000:
        return f"${amount / 1_000:.1f}K"
    return f"${amount:.2f}"


def format_risk_score(value: float) -> str:
    return f"{float(value):.1f}"


def ibm_dataset_status_label(is_available: bool) -> str:
    return "IBM AML dataset: Available" if is_available else "IBM AML dataset: Not found"


def breakdown_rows(breakdown: RiskScoreBreakdown) -> list[dict[str, object]]:
    return [
        {"component": "Evidence strength", "points": breakdown.evidence_strength_score},
        {"component": "Severity", "points": breakdown.severity_score},
        {"component": "Typology diversity", "points": breakdown.typology_diversity_score},
        {"component": "Repeated evidence", "points": breakdown.repeated_evidence_score},
        {"component": "Activity magnitude", "points": breakdown.activity_magnitude_score},
    ]


def evidence_rows(assessment: RiskAssessment) -> pd.DataFrame:
    records = []
    for evidence in sorted(
        assessment.contributing_evidence,
        key=lambda item: (-float(item.evidence_strength), item.detector_name, item.typology, item.start_time.isoformat() if item.start_time else "", item.end_time.isoformat() if item.end_time else ""),
    ):
        records.append(
            {
                "Detector": evidence.detector_name,
                "Typology": evidence.typology,
                "Severity": evidence.severity.upper(),
                "Evidence Strength": round(float(evidence.evidence_strength), 3),
                "Time Range": _time_range(evidence.start_time, evidence.end_time),
                "Transactions": int(evidence.transaction_count),
                "Amount": float(evidence.total_amount),
                "Involved Accounts": ", ".join(evidence.involved_account_ids[:8]),
                "Reasons": " | ".join(evidence.reasons[:3]),
            }
        )
    return pd.DataFrame.from_records(records)


def timeline_rows(report: Any) -> pd.DataFrame:
    records = []
    for item in getattr(report, "timeline", []):
        records.append(
            {
                "Start": item.timestamp,
                "End": item.end_time,
                "Typology": item.typology,
                "Detector": item.detector,
                "Transactions": item.transaction_count,
                "Amount": item.amount,
                "Description": item.short_description,
            }
        )
    return pd.DataFrame.from_records(records)


def parse_and_plan_query(query: str) -> dict[str, Any]:
    parsed = QueryParser().parse(query)
    plan = AgentPlanner().plan(parsed)
    return {
        "query": parsed.raw_query,
        "intent": parsed.intent.value,
        "aml_pattern": parsed.aml_pattern.value if parsed.aml_pattern else None,
        "requested_output": parsed.requested_output.value,
        "tool_names": [step.tool_name for step in plan.steps],
        "planning_summary": plan.planning_summary,
        "parsed": parsed,
        "plan": plan,
    }


def metric_value(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return format_risk_score(value)
    return str(value)


def _time_range(start: datetime | None, end: datetime | None) -> str:
    if start and end:
        return f"{start.isoformat()} -> {end.isoformat()}"
    if start:
        return f"from {start.isoformat()}"
    if end:
        return f"through {end.isoformat()}"
    return "not available"
