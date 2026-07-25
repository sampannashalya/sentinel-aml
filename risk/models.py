from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from detection.evidence import DetectionEvidence


class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class RiskScoringConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    medium_threshold: float = 25.0
    high_threshold: float = 50.0
    critical_threshold: float = 75.0
    strength_weight: float = 32.0
    severity_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "low": 4.0,
            "medium": 10.0,
            "high": 16.0,
        }
    )
    typology_diversity_weight: float = 16.0
    repeated_evidence_weight: float = 16.0
    activity_magnitude_weight: float = 16.0
    transaction_scale: float = 25.0
    amount_scale: float = 10_000.0

    def level_for_score(self, score: float) -> RiskLevel:
        if score >= self.critical_threshold:
            return RiskLevel.CRITICAL
        if score >= self.high_threshold:
            return RiskLevel.HIGH
        if score >= self.medium_threshold:
            return RiskLevel.MEDIUM
        return RiskLevel.LOW


class RiskScoreBreakdown(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence_strength_score: float = 0.0
    severity_score: float = 0.0
    typology_diversity_score: float = 0.0
    repeated_evidence_score: float = 0.0
    activity_magnitude_score: float = 0.0
    total_before_clamp: float = 0.0
    total_after_clamp: float = 0.0
    component_reasons: list[str] = Field(default_factory=list)


class RiskAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    primary_account_id: str = Field(..., min_length=1)
    risk_score: float = Field(default=0.0, ge=0.0, le=100.0)
    risk_level: RiskLevel = RiskLevel.LOW
    evidence_count: int = Field(default=0, ge=0)
    typologies_detected: list[str] = Field(default_factory=list)
    detector_names: list[str] = Field(default_factory=list)
    involved_account_ids: list[str] = Field(default_factory=list)
    entity_ids: list[str] = Field(default_factory=list)
    assessment_start_time: datetime | None = None
    assessment_end_time: datetime | None = None
    total_suspicious_amount: float = Field(default=0.0, ge=0.0)
    total_suspicious_transactions: int = Field(default=0, ge=0)
    contributing_evidence: list[DetectionEvidence] = Field(default_factory=list)
    score_breakdown: RiskScoreBreakdown = Field(default_factory=RiskScoreBreakdown)
    reasons: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
