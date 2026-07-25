from __future__ import annotations

import hashlib
from collections import OrderedDict
from datetime import datetime, timezone
from typing import Any

from agent.tool_contracts import ExecutionStatus, ToolAvailability, ToolMetadata, ToolResult
from risk.models import RiskAssessment, RiskLevel, RiskScoreBreakdown

from .models import InvestigationEvidenceSummary, InvestigationReport, InvestigationTimelineEntry


class InvestigationReportBuilder:
    LABEL_KEYS = {
        "is_laundering",
        "Is Laundering",
        "laundering_event_count",
        "laundering_event_rate",
        "annotation_typology",
        "expected_typology",
        "expected_ground_truth",
    }

    def build(self, assessment: RiskAssessment, generated_at: datetime | None = None) -> InvestigationReport:
        generated_at_provided = generated_at is not None
        generated_at = generated_at or datetime.now(timezone.utc)
        evidence = self._sorted_evidence(assessment)
        report_id = self._report_id(assessment)
        evidence_summary = [self._summarize_evidence(item) for item in evidence[:8]]
        timeline = self._build_timeline(evidence)
        key_findings = self._key_findings(assessment, evidence, timeline)
        score_explanation = self._score_explanation(assessment.score_breakdown)
        recommended_actions = self._recommended_actions(assessment.risk_level)

        return InvestigationReport(
            primary_account_id=assessment.primary_account_id,
            report_id=report_id,
            generated_at=generated_at,
            risk_score=assessment.risk_score,
            risk_level=assessment.risk_level.value if isinstance(assessment.risk_level, RiskLevel) else str(assessment.risk_level),
            executive_summary=self._executive_summary(assessment),
            typologies_detected=list(assessment.typologies_detected),
            detector_names=list(assessment.detector_names),
            evidence_count=assessment.evidence_count,
            involved_account_ids=list(assessment.involved_account_ids),
            entity_ids=list(assessment.entity_ids),
            assessment_start_time=assessment.assessment_start_time,
            assessment_end_time=assessment.assessment_end_time,
            suspicious_transaction_count=assessment.total_suspicious_transactions,
            suspicious_amount=assessment.total_suspicious_amount,
            key_findings=key_findings,
            evidence_summary=evidence_summary,
            timeline=timeline,
            score_explanation=score_explanation,
            recommended_actions=recommended_actions,
            limitations=self._limitations(),
            compliance_disclaimer=self._disclaimer(),
            metadata=self._sanitize_metadata(
                {
                    **assessment.metadata,
                    "report_version": "phase7",
                    "generated_at_source": "injected" if generated_at_provided else "runtime",
                }
            ),
        )

    def _report_id(self, assessment: RiskAssessment) -> str:
        parts = [
            assessment.primary_account_id,
            assessment.assessment_start_time.isoformat() if assessment.assessment_start_time else "",
            assessment.assessment_end_time.isoformat() if assessment.assessment_end_time else "",
            f"{assessment.risk_score:.4f}",
            "|".join(sorted(assessment.typologies_detected)),
        ]
        digest = hashlib.sha1("::".join(parts).encode("utf-8")).hexdigest().upper()[:10]
        return f"IR-{assessment.primary_account_id}-{digest}"

    def _executive_summary(self, assessment: RiskAssessment) -> str:
        typologies = ", ".join(assessment.typologies_detected) if assessment.typologies_detected else "no specific typology"
        count = assessment.evidence_count
        level = assessment.risk_level.value if isinstance(assessment.risk_level, RiskLevel) else str(assessment.risk_level)
        if assessment.risk_score >= 75:
            tail = "The combined evidence warrants immediate analyst review."
        elif assessment.risk_score >= 50:
            tail = "The evidence warrants prioritized analyst review."
        elif assessment.risk_score >= 25:
            tail = "The pattern merits analyst review and continued monitoring."
        else:
            tail = "The evidence supports routine documentation and ongoing monitoring."
        return (
            f"Account {assessment.primary_account_id} received a {level} AML risk score of {assessment.risk_score:.1f} "
            f"based on {count} suspicious-pattern evidence item(s) spanning {typologies}. {tail}"
        )

    def _key_findings(self, assessment: RiskAssessment, evidence: list[Any], timeline: list[InvestigationTimelineEntry]) -> list[str]:
        findings: list[str] = []
        if len(assessment.typologies_detected) > 1:
            findings.append("Multiple independent suspicious typologies were detected.")
        if any(item.evidence_strength >= 0.8 for item in evidence):
            findings.append("At least one detector produced strong evidence.")
        if assessment.evidence_count > 1:
            findings.append("More than one evidence item contributes to the assessment.")
        if "VELOCITY" in assessment.typologies_detected:
            findings.append("Elevated transaction velocity was observed.")
        if "CYCLE" in assessment.typologies_detected:
            findings.append("A circular transaction path was observed.")
        if "FAN-OUT" in assessment.typologies_detected or "FAN-IN" in assessment.typologies_detected:
            findings.append("Counterparty concentration behavior was observed.")
        if "GATHER-SCATTER" in assessment.typologies_detected:
            findings.append("A gather-scatter sequence was observed.")
        if "SCATTER-GATHER" in assessment.typologies_detected:
            findings.append("A scatter-gather sequence was observed.")
        if timeline and (assessment.assessment_start_time or assessment.assessment_end_time):
            findings.append("Suspicious activity occurred within a bounded time window.")
        return findings[:8]

    def _summarize_evidence(self, evidence: Any) -> InvestigationEvidenceSummary:
        time_window = self._format_time_window(evidence.start_time, evidence.end_time)
        return InvestigationEvidenceSummary(
            detector=evidence.detector_name,
            typology=evidence.typology,
            severity=evidence.severity,
            evidence_strength=float(evidence.evidence_strength),
            time_window=time_window,
            transaction_count=int(evidence.transaction_count),
            amount=float(evidence.total_amount),
            involved_accounts=list(evidence.involved_account_ids[:8]),
            reasons=list(evidence.reasons[:3]),
        )

    def _build_timeline(self, evidence: list[Any]) -> list[InvestigationTimelineEntry]:
        entries: list[InvestigationTimelineEntry] = []
        for item in evidence:
            entries.append(
                InvestigationTimelineEntry(
                    timestamp=item.start_time,
                    end_time=item.end_time,
                    typology=item.typology,
                    detector=item.detector_name,
                    short_description=self._timeline_description(item),
                    transaction_count=int(item.transaction_count),
                    amount=float(item.total_amount),
                )
            )
        return sorted(entries, key=lambda item: (item.timestamp or datetime.min, item.end_time or datetime.min, item.typology, item.detector))

    def _timeline_description(self, evidence: Any) -> str:
        if evidence.reasons:
            return evidence.reasons[0]
        return f"{evidence.detector_name} flagged {evidence.typology} evidence."

    def _score_explanation(self, breakdown: RiskScoreBreakdown) -> list[str]:
        if not breakdown.component_reasons:
            return ["No score breakdown was available."]
        return list(breakdown.component_reasons)

    def _recommended_actions(self, risk_level: RiskLevel | str) -> list[str]:
        level = risk_level.value if isinstance(risk_level, RiskLevel) else str(risk_level)
        if level == RiskLevel.CRITICAL.value:
            return [
                "Prioritize immediate analyst review.",
                "Investigate related accounts and entities.",
                "Preserve relevant evidence and chronology.",
                "Escalate according to institutional AML procedures.",
                "Consider whether regulatory reporting is warranted under applicable policy and law.",
            ]
        if level == RiskLevel.HIGH.value:
            return [
                "Prioritize analyst review.",
                "Examine counterparties and related accounts.",
                "Review transaction chronology and customer expectations.",
                "Consider escalation under internal AML procedures.",
            ]
        if level == RiskLevel.MEDIUM.value:
            return [
                "Review recent account activity.",
                "Verify customer and account context.",
                "Consider enhanced monitoring.",
            ]
        return [
            "Document the assessment.",
            "Continue routine monitoring.",
        ]

    def _limitations(self) -> list[str]:
        return [
            "The report is generated from deterministic heuristic evidence and scoring.",
            "It reflects the available transaction and account data only.",
            "The risk score is not proof of illicit activity or legal liability.",
            "External KYC or customer context is not included unless explicitly provided.",
        ]

    def _disclaimer(self) -> str:
        return (
            "This assessment identifies suspicious behavioral patterns for analyst review. "
            "It does not establish money laundering, fraud, criminal conduct, or legal liability. "
            "Final determinations require appropriate human investigation and applicable institutional or regulatory procedures."
        )

    def _sanitize_metadata(self, metadata: dict[str, Any]) -> dict[str, Any]:
        sanitized: dict[str, Any] = {}
        for key, value in sorted(metadata.items()):
            if key in self.LABEL_KEYS:
                continue
            sanitized[key] = self._sanitize_value(value)
        return sanitized

    def _sanitize_value(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: self._sanitize_value(subvalue)
                for key, subvalue in sorted(value.items())
                if key not in self.LABEL_KEYS
            }
        if isinstance(value, list):
            return [self._sanitize_value(item) for item in value]
        return value

    def _sorted_evidence(self, assessment: RiskAssessment) -> list[Any]:
        return sorted(
            list(assessment.contributing_evidence),
            key=lambda item: (
                -float(item.evidence_strength),
                item.detector_name,
                item.typology,
                self._datetime_key(item.start_time),
                self._datetime_key(item.end_time),
            ),
        )

    def _format_time_window(self, start: datetime | None, end: datetime | None) -> str:
        if start and end:
            return f"{start.isoformat()} -> {end.isoformat()}"
        if start:
            return f"from {start.isoformat()}"
        if end:
            return f"through {end.isoformat()}"
        return "not available"

    def _datetime_key(self, value: datetime | None) -> str:
        return value.isoformat() if value is not None else ""


class InvestigationReportBuilderTool:
    def __init__(self, builder: InvestigationReportBuilder | None = None) -> None:
        self.builder = builder or InvestigationReportBuilder()
        self.metadata = ToolMetadata(
            name="investigation_report",
            description="Build a deterministic AML investigation report from a risk assessment.",
            availability=ToolAvailability.IMPLEMENTED,
            input_type="ExecutionContext",
            output_type="InvestigationReport",
        )

    def execute(self, context: Any, parameters: dict[str, Any] | None = None) -> ToolResult:
        params = parameters or {}
        assessment = self._coerce_assessment(params)
        report = self.builder.build(assessment, generated_at=params.get("generated_at"))
        return ToolResult(status=ExecutionStatus.SUCCESS, summary="Built an investigation report.", data={"report": report})

    def _coerce_assessment(self, params: dict[str, Any]) -> RiskAssessment:
        assessment = params.get("assessment") or params.get("risk_assessment")
        if isinstance(assessment, RiskAssessment):
            return assessment
        if assessment is None:
            raise ValueError("investigation_report requires an assessment or risk_assessment parameter.")
        return RiskAssessment.model_validate(assessment)
