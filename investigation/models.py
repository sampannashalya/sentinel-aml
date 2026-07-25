from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class InvestigationEvidenceSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    detector: str
    typology: str
    severity: str
    evidence_strength: float
    time_window: str
    transaction_count: int
    amount: float
    involved_accounts: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)


class InvestigationTimelineEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    timestamp: datetime | None = None
    end_time: datetime | None = None
    typology: str
    detector: str
    short_description: str
    transaction_count: int
    amount: float


class InvestigationReport(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    primary_account_id: str = Field(..., min_length=1)
    report_id: str = Field(..., min_length=1)
    generated_at: datetime
    risk_score: float = Field(ge=0.0, le=100.0)
    risk_level: str
    executive_summary: str
    typologies_detected: list[str] = Field(default_factory=list)
    detector_names: list[str] = Field(default_factory=list)
    evidence_count: int = Field(ge=0)
    involved_account_ids: list[str] = Field(default_factory=list)
    entity_ids: list[str] = Field(default_factory=list)
    assessment_start_time: datetime | None = None
    assessment_end_time: datetime | None = None
    suspicious_transaction_count: int = Field(ge=0)
    suspicious_amount: float = Field(ge=0.0)
    key_findings: list[str] = Field(default_factory=list)
    evidence_summary: list[InvestigationEvidenceSummary] = Field(default_factory=list)
    timeline: list[InvestigationTimelineEntry] = Field(default_factory=list)
    score_explanation: list[str] = Field(default_factory=list)
    recommended_actions: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    compliance_disclaimer: str
    metadata: dict[str, Any] = Field(default_factory=dict)
