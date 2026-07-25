from __future__ import annotations

from datetime import date
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class QueryIntent(str, Enum):
    broad_analysis = "broad_analysis"
    suspicious_activity_search = "suspicious_activity_search"
    structuring_detection = "structuring_detection"
    smurfing_detection = "smurfing_detection"
    fan_out_detection = "fan_out_detection"
    fan_in_detection = "fan_in_detection"
    velocity_analysis = "velocity_analysis"
    cycle_detection = "cycle_detection"
    gather_scatter_detection = "gather_scatter_detection"
    scatter_gather_detection = "scatter_gather_detection"
    customer_investigation = "customer_investigation"
    threshold_analysis = "threshold_analysis"
    risk_explanation = "risk_explanation"
    high_risk_search = "high_risk_search"
    eda_request = "eda_request"


class AMLPattern(str, Enum):
    suspicious_activity = "suspicious_activity"
    structuring = "structuring"
    smurfing = "smurfing"
    fan_out = "fan_out"
    fan_in = "fan_in"
    velocity = "velocity"
    cycle = "cycle"
    gather_scatter = "gather_scatter"
    scatter_gather = "scatter_gather"
    behavior_deviation = "behavior_deviation"


class RequestedOutput(str, Enum):
    investigation_summary = "investigation_summary"
    pattern_summary = "pattern_summary"
    customer_list = "customer_list"
    case_file = "case_file"
    explanation = "explanation"
    summary = "summary"
    eda_summary = "eda_summary"


class QueryDateRange(BaseModel):
    model_config = ConfigDict(extra="forbid")

    start_date: date | None = Field(default=None)
    end_date: date | None = Field(default=None)
    relative_days: int | None = Field(default=None, ge=1)


class NumericThreshold(BaseModel):
    model_config = ConfigDict(extra="forbid")

    operator: str = Field(..., pattern=r"^(<|<=|=|>=|>|between)$")
    value: float = Field(..., gt=0)
    upper_value: float | None = Field(default=None, gt=0)
    currency: str | None = None


class QueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    raw_query: str
    intent: QueryIntent
    customer_id: str | None = None
    date_range: QueryDateRange | None = None
    amount_threshold: NumericThreshold | None = None
    transaction_count_threshold: NumericThreshold | None = None
    transaction_type: str | None = None
    country: str | None = None
    aml_pattern: AMLPattern | None = None
    requested_output: RequestedOutput = RequestedOutput.summary
