from __future__ import annotations

from typing import Any

import pandas as pd

from agent.tool_contracts import ExecutionStatus, ToolAvailability, ToolMetadata, ToolResult
from detection.evidence import DetectionEvidence
from tools.feature_store import FeatureStore
from config.settings import IBM_FEATURE_CACHE_DIR


class VelocityDetector:
    LABEL_COLUMNS = {"is_laundering", "Is Laundering", "laundering_event_count", "has_laundering_label", "laundering_event_rate"}

    def __init__(self, feature_store: FeatureStore | None = None) -> None:
        self.feature_store = feature_store or FeatureStore(IBM_FEATURE_CACHE_DIR)
        self.metadata = ToolMetadata(
            name="velocity_detector",
            description="Detect accounts with unusually high transaction velocity using cached account features.",
            availability=ToolAvailability.IMPLEMENTED,
            input_type="ExecutionContext",
            output_type="list[DetectionEvidence]",
        )

    def execute(self, context: Any, parameters: dict[str, Any] | None = None) -> ToolResult:
        params = parameters or {}
        evidence = self.detect(
            feature_frame=params.get("feature_frame"),
            min_max_transactions_in_hour=int(params.get("min_max_transactions_in_hour", 10)),
            min_rapid_gap_count=int(params.get("min_rapid_gap_count", 5)),
            min_transactions_per_active_hour=float(params.get("min_transactions_per_active_hour", 5.0)),
        )
        return ToolResult(status=ExecutionStatus.SUCCESS, summary=f"Detected {len(evidence)} velocity evidence item(s).", data={"evidence": evidence})

    def detect(
        self,
        *,
        feature_frame: pd.DataFrame | None = None,
        min_max_transactions_in_hour: int = 10,
        min_rapid_gap_count: int = 5,
        min_transactions_per_active_hour: float = 5.0,
    ) -> list[DetectionEvidence]:
        features = feature_frame if feature_frame is not None else self._load_features()
        features = self._prepare_features(features)
        if features.empty:
            return []

        evidence: list[DetectionEvidence] = []
        triggered_features = features[
            (features["max_transactions_in_hour"] >= min_max_transactions_in_hour)
            | (features["rapid_gap_count"] >= min_rapid_gap_count)
            | (features["transactions_per_active_hour"] >= min_transactions_per_active_hour)
        ]

        for row in triggered_features.itertuples(index=False):
            values = row._asdict()
            reasons: list[str] = []
            if values["max_transactions_in_hour"] >= min_max_transactions_in_hour:
                reasons.append(
                    f"Account generated {int(values['max_transactions_in_hour'])} transactions within its busiest hour, exceeding the configured threshold of {min_max_transactions_in_hour}."
                )
            if values["rapid_gap_count"] >= min_rapid_gap_count:
                reasons.append(
                    f"Account recorded {int(values['rapid_gap_count'])} rapid transaction gaps, exceeding the configured threshold of {min_rapid_gap_count}."
                )
            if values["transactions_per_active_hour"] >= min_transactions_per_active_hour:
                reasons.append(
                    f"Account averaged {float(values['transactions_per_active_hour']):.2f} transactions per active hour, exceeding the configured threshold of {min_transactions_per_active_hour}."
                )
            raw_strength = max(
                float(values["max_transactions_in_hour"]) / float(max(min_max_transactions_in_hour, 1)),
                float(values["rapid_gap_count"]) / float(max(min_rapid_gap_count, 1)),
                float(values["transactions_per_active_hour"]) / float(max(min_transactions_per_active_hour, 1e-9)),
            )
            strength = min(1.0, raw_strength)

            evidence.append(
                DetectionEvidence(
                    detector_name="velocity_detector",
                    typology="VELOCITY",
                    primary_account_id=str(values["account_id"]),
                    involved_account_ids=[str(values["account_id"])],
                    entity_ids=[str(values["entity_id"])] if pd.notna(values.get("entity_id")) and str(values.get("entity_id")) else [],
                    start_time=None,
                    end_time=None,
                    transaction_count=int(values.get("transaction_count", 0)),
                    total_amount=float(values.get("total_outgoing_amount", 0.0)) + float(values.get("total_incoming_amount", 0.0)),
                    evidence_strength=strength,
                    severity=self._severity(raw_strength),
                    reasons=reasons,
                    detector_parameters={
                        "min_max_transactions_in_hour": min_max_transactions_in_hour,
                        "min_rapid_gap_count": min_rapid_gap_count,
                        "min_transactions_per_active_hour": min_transactions_per_active_hour,
                    },
                    transaction_references=[],
                    metadata={
                        "max_transactions_in_hour": int(values["max_transactions_in_hour"]),
                        "rapid_gap_count": int(values["rapid_gap_count"]),
                        "transactions_per_active_hour": float(values["transactions_per_active_hour"]),
                    },
                )
            )
        return evidence

    def _prepare_features(self, features: pd.DataFrame) -> pd.DataFrame:
        if features.empty:
            return pd.DataFrame()

        prepared = features.drop(columns=[column for column in self.LABEL_COLUMNS if column in features.columns], errors="ignore").copy()
        required_columns = {
            "account_id",
            "max_transactions_in_hour",
            "rapid_gap_count",
            "transactions_per_active_hour",
        }
        missing_columns = required_columns - set(prepared.columns)
        if missing_columns:
            raise ValueError(f"Velocity detection requires feature columns: {sorted(missing_columns)}")

        defaults = {
            "transaction_count": 0,
            "total_outgoing_amount": 0.0,
            "total_incoming_amount": 0.0,
            "entity_id": "",
        }
        for column, default in defaults.items():
            if column not in prepared.columns:
                prepared[column] = default

        numeric_columns = [
            "max_transactions_in_hour",
            "rapid_gap_count",
            "transactions_per_active_hour",
            "transaction_count",
            "total_outgoing_amount",
            "total_incoming_amount",
        ]
        for column in numeric_columns:
            prepared[column] = pd.to_numeric(prepared[column], errors="coerce").fillna(0)
        return prepared

    def _load_features(self) -> pd.DataFrame:
        if not self.feature_store.exists():
            return pd.DataFrame()
        return self.feature_store.load_features()

    def _severity(self, strength: float) -> str:
        if strength >= 1.75:
            return "high"
        if strength >= 1.25:
            return "medium"
        return "low"
