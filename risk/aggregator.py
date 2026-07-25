from __future__ import annotations

import json
from collections import OrderedDict
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any

from agent.tool_contracts import ExecutionStatus, ToolAvailability, ToolMetadata, ToolResult
from detection.evidence import DetectionEvidence

from .models import RiskAssessment


class EvidenceAggregator:
    LABEL_KEYS = {
        "is_laundering",
        "Is Laundering",
        "laundering_event_count",
        "laundering_event_rate",
        "annotation_typology",
        "expected_typology",
        "expected_ground_truth",
    }

    def aggregate(self, evidence_items: list[DetectionEvidence]) -> list[RiskAssessment]:
        if not evidence_items:
            return []

        grouped: dict[str, list[DetectionEvidence]] = {}
        for evidence in evidence_items:
            grouped.setdefault(evidence.primary_account_id, []).append(self._sanitize_evidence(evidence))

        assessments = [self._build_assessment(account_id, items) for account_id, items in sorted(grouped.items())]
        return assessments

    def _build_assessment(self, primary_account_id: str, items: list[DetectionEvidence]) -> RiskAssessment:
        deduplicated = self._deduplicate(items)
        ordered_evidence = sorted(deduplicated, key=self._evidence_signature)

        involved_accounts = self._ordered_union(
            value
            for evidence in ordered_evidence
            for value in [evidence.primary_account_id, *evidence.involved_account_ids]
        )
        entity_ids = self._ordered_union(
            value for evidence in ordered_evidence for value in evidence.entity_ids
        )
        detector_names = self._ordered_union(evidence.detector_name for evidence in ordered_evidence)
        typologies = self._ordered_union(evidence.typology for evidence in ordered_evidence)

        assessment_start_time = self._min_datetime(evidence.start_time for evidence in ordered_evidence)
        assessment_end_time = self._max_datetime(evidence.end_time for evidence in ordered_evidence)

        total_transactions = 0.0
        total_amount = 0.0
        seen_transaction_refs: set[str] = set()
        unique_transaction_refs: list[str] = []

        for evidence in ordered_evidence:
            refs = self._normalized_refs(evidence.transaction_references)
            if refs:
                new_refs = [ref for ref in refs if ref not in seen_transaction_refs]
                if not new_refs:
                    continue
                ratio = len(new_refs) / len(refs)
                seen_transaction_refs.update(new_refs)
                unique_transaction_refs.extend(new_refs)
                total_transactions += evidence.transaction_count * ratio
                total_amount += evidence.total_amount * ratio
            else:
                total_transactions += evidence.transaction_count
                total_amount += evidence.total_amount

        reasons = [
            f"Aggregated {len(ordered_evidence)} unique evidence item(s) for account {primary_account_id}.",
        ]
        if len(typologies) > 1:
            reasons.append(f"Evidence spans {len(typologies)} distinct suspicious typology types.")
        if len(detector_names) > 1:
            reasons.append(f"Signals came from {len(detector_names)} detector(s).")

        return RiskAssessment(
            primary_account_id=primary_account_id,
            evidence_count=len(ordered_evidence),
            typologies_detected=typologies,
            detector_names=detector_names,
            involved_account_ids=involved_accounts,
            entity_ids=entity_ids,
            assessment_start_time=assessment_start_time,
            assessment_end_time=assessment_end_time,
            total_suspicious_amount=float(total_amount),
            total_suspicious_transactions=int(round(total_transactions)),
            contributing_evidence=ordered_evidence,
            reasons=reasons,
            metadata={
                "raw_evidence_count": len(items),
                "deduplicated_evidence_count": len(ordered_evidence),
                "unique_transaction_reference_count": len(unique_transaction_refs),
                "unique_transaction_references": unique_transaction_refs[:50],
            },
        )

    def _deduplicate(self, items: list[DetectionEvidence]) -> list[DetectionEvidence]:
        deduped: list[DetectionEvidence] = []
        seen: set[str] = set()
        for evidence in items:
            signature = self._evidence_signature(evidence)
            if signature in seen:
                continue
            seen.add(signature)
            deduped.append(evidence)
        return deduped

    def _sanitize_evidence(self, evidence: DetectionEvidence) -> DetectionEvidence:
        cleaned_metadata = self._clean_value(evidence.metadata)
        cleaned_parameters = self._clean_value(evidence.detector_parameters)
        return evidence.model_copy(
            update={
                "metadata": cleaned_metadata,
                "detector_parameters": cleaned_parameters,
            }
        )

    def _evidence_signature(self, evidence: DetectionEvidence) -> str:
        payload = {
            "detector_name": evidence.detector_name,
            "typology": evidence.typology,
            "primary_account_id": evidence.primary_account_id,
            "involved_account_ids": sorted(str(value) for value in evidence.involved_account_ids),
            "entity_ids": sorted(str(value) for value in evidence.entity_ids),
            "start_time": self._datetime_key(evidence.start_time),
            "end_time": self._datetime_key(evidence.end_time),
            "transaction_count": evidence.transaction_count,
            "total_amount": round(float(evidence.total_amount), 6),
            "evidence_strength": round(float(evidence.evidence_strength), 6),
            "severity": evidence.severity,
            "reasons": tuple(evidence.reasons),
            "detector_parameters": self._clean_value(evidence.detector_parameters),
            "transaction_references": sorted(self._normalized_refs(evidence.transaction_references)),
        }
        return json.dumps(payload, sort_keys=True, default=str)

    def _normalized_refs(self, values: Iterable[str]) -> list[str]:
        return [str(value) for value in values if str(value)]

    def _ordered_union(self, values: Iterable[str]) -> list[str]:
        ordered = OrderedDict()
        for value in values:
            text = str(value)
            if text:
                ordered.setdefault(text, None)
        return list(ordered.keys())

    def _clean_value(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: self._clean_value(subvalue)
                for key, subvalue in sorted(value.items())
                if key not in self.LABEL_KEYS
            }
        if isinstance(value, list):
            return [self._clean_value(item) for item in value]
        return value

    def _datetime_key(self, value: datetime | None) -> str:
        return value.isoformat() if value is not None else ""

    def _min_datetime(self, values: Iterable[datetime | None]) -> datetime | None:
        timestamps = [value for value in values if value is not None]
        return min(timestamps) if timestamps else None

    def _max_datetime(self, values: Iterable[datetime | None]) -> datetime | None:
        timestamps = [value for value in values if value is not None]
        return max(timestamps) if timestamps else None


class EvidenceAggregatorTool:
    def __init__(self, aggregator: EvidenceAggregator | None = None) -> None:
        self.aggregator = aggregator or EvidenceAggregator()
        self.metadata = ToolMetadata(
            name="evidence_aggregator",
            description="Group detector evidence into account-level risk assessments.",
            availability=ToolAvailability.IMPLEMENTED,
            input_type="ExecutionContext",
            output_type="list[RiskAssessment]",
        )

    def execute(self, context: Any, parameters: dict[str, Any] | None = None) -> ToolResult:
        params = parameters or {}
        evidence = self._coerce_evidence(params.get("evidence") or params.get("evidence_items") or [])
        assessments = self.aggregator.aggregate(evidence)
        return ToolResult(
            status=ExecutionStatus.SUCCESS,
            summary=f"Aggregated {len(evidence)} evidence item(s) into {len(assessments)} assessment(s).",
            data={"assessments": assessments},
        )

    def _coerce_evidence(self, items: Iterable[Any]) -> list[DetectionEvidence]:
        evidence: list[DetectionEvidence] = []
        for item in items:
            if isinstance(item, DetectionEvidence):
                evidence.append(item)
            else:
                evidence.append(DetectionEvidence.model_validate(item))
        return evidence
