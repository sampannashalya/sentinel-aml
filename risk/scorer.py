from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

from agent.tool_contracts import ExecutionStatus, ToolAvailability, ToolMetadata, ToolResult
from detection.evidence import DetectionEvidence

from .aggregator import EvidenceAggregator
from .models import RiskAssessment, RiskLevel, RiskScoreBreakdown, RiskScoringConfig


class RiskScorer:
    def __init__(self, config: RiskScoringConfig | None = None) -> None:
        self.config = config or RiskScoringConfig()

    def score(self, assessments: list[RiskAssessment]) -> list[RiskAssessment]:
        if not assessments:
            return []
        return [self.score_assessment(assessment) for assessment in sorted(assessments, key=self._assessment_sort_key)]

    def score_assessment(self, assessment: RiskAssessment) -> RiskAssessment:
        evidence = list(assessment.contributing_evidence)
        if not evidence:
            breakdown = RiskScoreBreakdown(component_reasons=["No contributing evidence was supplied."])
            return assessment.model_copy(
                update={
                    "risk_score": 0.0,
                    "risk_level": RiskLevel.LOW,
                    "score_breakdown": breakdown,
                    "reasons": assessment.reasons + ["No contributing evidence was supplied, so the risk score remains at zero."],
                    "metadata": {
                        **assessment.metadata,
                        "scoring_config": self.config.model_dump(mode="json"),
                    },
                }
            )

        strength = self._weighted_strength(evidence)
        max_severity = self._max_severity_score(evidence)
        typology_count = len({item.typology for item in evidence})
        evidence_count = len(evidence)
        tx_component = self._transaction_component(assessment.total_suspicious_transactions)
        amount_component = self._amount_component(assessment.total_suspicious_amount)

        evidence_strength_score = round(self.config.strength_weight * strength, 4)
        severity_score = round(max_severity, 4)
        typology_diversity_score = round(self._typology_score(typology_count), 4)
        repeated_evidence_score = round(self._repeat_score(evidence_count), 4)
        activity_magnitude_score = round(min(self.config.activity_magnitude_weight, tx_component + amount_component), 4)

        total_before_clamp = (
            evidence_strength_score
            + severity_score
            + typology_diversity_score
            + repeated_evidence_score
            + activity_magnitude_score
        )
        risk_score = round(max(0.0, min(100.0, total_before_clamp)), 4)
        risk_level = self.config.level_for_score(risk_score)

        component_reasons = [
            f"Evidence strength contributed {evidence_strength_score:.2f} points from an average strength of {strength:.2f}.",
            f"Severity contributed {severity_score:.2f} points based on the strongest evidence severity.",
            f"Typology diversity contributed {typology_diversity_score:.2f} points across {typology_count} typology/typologies.",
            f"Repeated evidence contributed {repeated_evidence_score:.2f} points across {evidence_count} evidence item(s).",
            f"Activity magnitude contributed {activity_magnitude_score:.2f} points from {assessment.total_suspicious_transactions} suspicious transaction(s) and {assessment.total_suspicious_amount:.2f} suspicious amount.",
        ]

        reasons = list(assessment.reasons) + component_reasons

        return assessment.model_copy(
            update={
                "risk_score": risk_score,
                "risk_level": risk_level,
                "score_breakdown": RiskScoreBreakdown(
                    evidence_strength_score=evidence_strength_score,
                    severity_score=severity_score,
                    typology_diversity_score=typology_diversity_score,
                    repeated_evidence_score=repeated_evidence_score,
                    activity_magnitude_score=activity_magnitude_score,
                    total_before_clamp=round(total_before_clamp, 4),
                    total_after_clamp=risk_score,
                    component_reasons=component_reasons,
                ),
                "reasons": reasons,
                "metadata": {
                    **assessment.metadata,
                    "scoring_config": self.config.model_dump(mode="json"),
                },
            }
        )

    def _assessment_sort_key(self, assessment: RiskAssessment) -> tuple[str, str]:
        return (
            assessment.primary_account_id,
            assessment.assessment_start_time.isoformat() if assessment.assessment_start_time else "",
        )

    def _weighted_strength(self, evidence: list[DetectionEvidence]) -> float:
        weights = [max(item.transaction_count, 1) for item in evidence]
        numerator = sum(item.evidence_strength * weight for item, weight in zip(evidence, weights))
        denominator = sum(weights)
        return numerator / denominator if denominator else 0.0

    def _max_severity_score(self, evidence: list[DetectionEvidence]) -> float:
        strongest = 0.0
        for item in evidence:
            strongest = max(strongest, self.config.severity_weights.get(item.severity, 0.0))
        return strongest

    def _typology_score(self, typology_count: int) -> float:
        if typology_count <= 0:
            return 0.0
        return min(self.config.typology_diversity_weight, 4.0 + (typology_count - 1) * 4.0)

    def _repeat_score(self, evidence_count: int) -> float:
        if evidence_count <= 1:
            return 0.0
        return min(self.config.repeated_evidence_weight, (evidence_count - 1) * 5.0)

    def _transaction_component(self, transaction_count: int) -> float:
        if transaction_count <= 0:
            return 0.0
        normalized = math.log1p(transaction_count) / math.log1p(self.config.transaction_scale)
        return min(self.config.activity_magnitude_weight / 2, normalized * (self.config.activity_magnitude_weight / 2))

    def _amount_component(self, total_amount: float) -> float:
        if total_amount <= 0:
            return 0.0
        normalized = math.log1p(total_amount / self.config.amount_scale) / math.log1p(100.0)
        return min(self.config.activity_magnitude_weight / 2, normalized * (self.config.activity_magnitude_weight / 2))


class RiskScorerTool:
    def __init__(self, scorer: RiskScorer | None = None, aggregator: EvidenceAggregator | None = None) -> None:
        self.scorer = scorer or RiskScorer()
        self.aggregator = aggregator or EvidenceAggregator()
        self.metadata = ToolMetadata(
            name="risk_scorer",
            description="Convert aggregated suspicious-pattern evidence into a deterministic risk assessment.",
            availability=ToolAvailability.IMPLEMENTED,
            input_type="ExecutionContext",
            output_type="list[RiskAssessment]",
        )

    def execute(self, context: Any, parameters: dict[str, Any] | None = None) -> ToolResult:
        params = parameters or {}
        assessments = self._coerce_assessments(params)
        scored = self.scorer.score(assessments)
        return ToolResult(
            status=ExecutionStatus.SUCCESS,
            summary=f"Scored {len(scored)} risk assessment(s).",
            data={"assessments": scored},
        )

    def _coerce_assessments(self, params: dict[str, Any]) -> list[RiskAssessment]:
        if params.get("assessments") is not None:
            return [assessment if isinstance(assessment, RiskAssessment) else RiskAssessment.model_validate(assessment) for assessment in params["assessments"]]
        evidence_items = params.get("evidence") or params.get("evidence_items") or []
        evidence = [item if isinstance(item, DetectionEvidence) else DetectionEvidence.model_validate(item) for item in evidence_items]
        return self.aggregator.aggregate(evidence)
