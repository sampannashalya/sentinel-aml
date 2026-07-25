from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class DetectorParameterSet(BaseModel):
    model_config = ConfigDict(extra="forbid")

    parameters: dict[str, Any] = Field(default_factory=dict)


class DetectionEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    detector_name: str = Field(..., min_length=1)
    typology: str = Field(..., min_length=1)
    primary_account_id: str = Field(..., min_length=1)
    involved_account_ids: list[str] = Field(default_factory=list)
    entity_ids: list[str] = Field(default_factory=list)
    start_time: datetime | None = None
    end_time: datetime | None = None
    transaction_count: int = Field(default=0, ge=0)
    total_amount: float = Field(default=0.0, ge=0.0)
    evidence_strength: float = Field(default=0.0, ge=0.0, le=1.0)
    severity: Literal["low", "medium", "high"] = "medium"
    reasons: list[str] = Field(default_factory=list)
    detector_parameters: dict[str, Any] = Field(default_factory=dict)
    transaction_references: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    assessment_type: Literal["suspicious_pattern_evidence"] = "suspicious_pattern_evidence"

    @field_validator("typology")
    @classmethod
    def normalize_typology(cls, value: str) -> str:
        return value.strip().upper().replace(" ", "-")

    @field_validator("involved_account_ids", "entity_ids", "transaction_references")
    @classmethod
    def remove_empty_values(cls, values: list[str]) -> list[str]:
        return [str(value) for value in values if str(value)]
