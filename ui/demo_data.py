from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

from agent.planner import AgentPlanner
from agent.query_parser import QueryParser
from detection import CycleDetector, DetectionEvidence, FanInDetector, FanOutDetector, GatherScatterDetector, ScatterGatherDetector, VelocityDetector
from investigation import InvestigationReport, InvestigationReportBuilder
from risk import EvidenceAggregator, RiskAssessment, RiskLevel, RiskScorer
from tools.feature_store import FeatureStore
from tools.ibm_dataset_adapter import IBMDatasetAdapter
from config.settings import IBM_FEATURE_CACHE_DIR, IBM_TRANSACTION_PATH

INVESTIGATION_SOURCE_LABEL = "Demo / Synthetic Evidence"
DEMO_ACCOUNT_ID = "ACCT-DEMO-1"
ALL_TYPOLOGIES = [
    "FAN-OUT",
    "FAN-IN",
    "VELOCITY",
    "CYCLE",
    "GATHER-SCATTER",
    "SCATTER-GATHER",
]


@dataclass(frozen=True)
class InvestigationBundle:
    source_label: str
    is_synthetic: bool
    account_id: str
    evidence: list[DetectionEvidence]
    assessment: RiskAssessment | None
    report: InvestigationReport | None
    notes: list[str]
    query_intent: str | None = None
    planned_tools: list[str] | None = None


def build_demo_evidence(selected_typologies: Sequence[str] | None = None) -> list[DetectionEvidence]:
    selected = set(selected_typologies or ALL_TYPOLOGIES)
    base = datetime(2022, 9, 1, 9, 0)
    evidence: list[DetectionEvidence] = []

    if "FAN-OUT" in selected:
        evidence.append(
            DetectionEvidence(
                detector_name="fan_out_detector",
                typology="FAN-OUT",
                primary_account_id=DEMO_ACCOUNT_ID,
                involved_account_ids=[DEMO_ACCOUNT_ID, "ACCT-B", "ACCT-C", "ACCT-D"],
                entity_ids=["ENT-DEMO-1"],
                start_time=base,
                end_time=base + timedelta(minutes=45),
                transaction_count=4,
                total_amount=12500.0,
                evidence_strength=0.82,
                severity="high",
                reasons=["Demo source account sent funds to multiple distinct receivers in a short window."],
                detector_parameters={"min_distinct_receivers": 4, "time_window_hours": 24},
                transaction_references=["demo-fo-1", "demo-fo-2", "demo-fo-3", "demo-fo-4"],
                metadata={"source": "synthetic_demo"},
            )
        )

    if "FAN-IN" in selected:
        evidence.append(
            DetectionEvidence(
                detector_name="fan_in_detector",
                typology="FAN-IN",
                primary_account_id=DEMO_ACCOUNT_ID,
                involved_account_ids=[DEMO_ACCOUNT_ID, "ACCT-E", "ACCT-F", "ACCT-G"],
                entity_ids=["ENT-DEMO-2"],
                start_time=base + timedelta(minutes=10),
                end_time=base + timedelta(minutes=40),
                transaction_count=4,
                total_amount=9200.0,
                evidence_strength=0.77,
                severity="medium",
                reasons=["Demo destination account received funds from multiple distinct senders."],
                detector_parameters={"min_distinct_senders": 4, "time_window_hours": 24},
                transaction_references=["demo-fi-1", "demo-fi-2", "demo-fi-3", "demo-fi-4"],
                metadata={"source": "synthetic_demo"},
            )
        )

    if "VELOCITY" in selected:
        evidence.append(
            DetectionEvidence(
                detector_name="velocity_detector",
                typology="VELOCITY",
                primary_account_id=DEMO_ACCOUNT_ID,
                involved_account_ids=[DEMO_ACCOUNT_ID],
                entity_ids=["ENT-DEMO-1"],
                start_time=base + timedelta(hours=1),
                end_time=base + timedelta(hours=1, minutes=15),
                transaction_count=18,
                total_amount=28500.0,
                evidence_strength=0.74,
                severity="medium",
                reasons=["Demo account generated elevated transaction velocity across its busiest hour."],
                detector_parameters={"min_max_transactions_in_hour": 10, "min_rapid_gap_count": 5},
                transaction_references=[],
                metadata={"source": "synthetic_demo"},
            )
        )

    if "GATHER-SCATTER" in selected:
        evidence.append(
            DetectionEvidence(
                detector_name="gather_scatter_detector",
                typology="GATHER-SCATTER",
                primary_account_id=DEMO_ACCOUNT_ID,
                involved_account_ids=[DEMO_ACCOUNT_ID, "ACCT-H", "ACCT-I", "ACCT-J"],
                entity_ids=["ENT-DEMO-1"],
                start_time=base + timedelta(hours=1, minutes=20),
                end_time=base + timedelta(hours=2),
                transaction_count=5,
                total_amount=14200.0,
                evidence_strength=0.79,
                severity="high",
                reasons=["Demo account first gathered funds from several senders and then scattered to destinations."],
                detector_parameters={"min_distinct_incoming_senders": 3, "min_distinct_outgoing_destinations": 2, "max_time_window_hours": 24},
                transaction_references=["demo-gs-1", "demo-gs-2", "demo-gs-3", "demo-gs-4", "demo-gs-5"],
                metadata={"source": "synthetic_demo"},
            )
        )

    if "CYCLE" in selected:
        evidence.append(
            DetectionEvidence(
                detector_name="cycle_detector",
                typology="CYCLE",
                primary_account_id=DEMO_ACCOUNT_ID,
                involved_account_ids=[DEMO_ACCOUNT_ID, "ACCT-B", "ACCT-C", "ACCT-D"],
                entity_ids=["ENT-DEMO-1"],
                start_time=base + timedelta(hours=2),
                end_time=base + timedelta(hours=3),
                transaction_count=4,
                total_amount=9800.0,
                evidence_strength=0.91,
                severity="high",
                reasons=["Demo account participated in a bounded circular transaction path."],
                detector_parameters={"min_hops": 4, "max_hops": 6, "max_elapsed_hours": 24},
                transaction_references=["demo-cy-1", "demo-cy-2", "demo-cy-3", "demo-cy-4"],
                metadata={"source": "synthetic_demo"},
            )
        )

    if "SCATTER-GATHER" in selected:
        evidence.append(
            DetectionEvidence(
                detector_name="scatter_gather_detector",
                typology="SCATTER-GATHER",
                primary_account_id=DEMO_ACCOUNT_ID,
                involved_account_ids=[DEMO_ACCOUNT_ID, "ACCT-K", "ACCT-L", "ACCT-M"],
                entity_ids=["ENT-DEMO-3"],
                start_time=base + timedelta(hours=3, minutes=15),
                end_time=base + timedelta(hours=4),
                transaction_count=5,
                total_amount=16800.0,
                evidence_strength=0.81,
                severity="high",
                reasons=["Demo origin account scattered to intermediaries that later gathered toward a common destination."],
                detector_parameters={"min_intermediaries": 2, "max_time_window_hours": 24},
                transaction_references=["demo-sg-1", "demo-sg-2", "demo-sg-3", "demo-sg-4", "demo-sg-5"],
                metadata={"source": "synthetic_demo"},
            )
        )

    return evidence


def build_demo_bundle(selected_typologies: Sequence[str] | None = None, generated_at: datetime | None = None) -> InvestigationBundle:
    evidence = build_demo_evidence(selected_typologies)
    assessment = _score_evidence(evidence)
    report = InvestigationReportBuilder().build(assessment, generated_at=generated_at or datetime(2022, 9, 1, 12, 0))
    notes = ["Synthetic demo evidence is shown for demonstration only."]
    return InvestigationBundle(
        source_label=INVESTIGATION_SOURCE_LABEL,
        is_synthetic=True,
        account_id=DEMO_ACCOUNT_ID,
        evidence=evidence,
        assessment=assessment,
        report=report,
        notes=notes,
    )


def build_account_investigation_bundle(
    account_id: str,
    selected_typologies: Sequence[str] | None = None,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
    chunksize: int = 100_000,
) -> InvestigationBundle:
    selected = set(selected_typologies or ALL_TYPOLOGIES)
    notes: list[str] = []
    evidence: list[DetectionEvidence] = []
    adapter = IBMDatasetAdapter()

    if not Path(IBM_TRANSACTION_PATH).exists():
        notes.append("IBM transaction data is not available locally, so the account investigation could not load raw transactions.")
        return InvestigationBundle(INVESTIGATION_SOURCE_LABEL, False, account_id, [], None, None, notes)

    transactions = adapter.load_transactions(
        account_ids=[account_id],
        start_date=start_date,
        end_date=end_date,
        chunksize=chunksize,
    )
    if transactions.empty:
        notes.append("No matching transactions were found for the selected account and filters.")
    else:
        if "FAN-OUT" in selected:
            evidence.extend(FanOutDetector().detect(transaction_frame=transactions, min_distinct_receivers=4, min_transaction_count=4, time_window_hours=24))
        if "FAN-IN" in selected:
            evidence.extend(FanInDetector().detect(transaction_frame=transactions, min_distinct_senders=4, min_transaction_count=4, time_window_hours=24))
        if "CYCLE" in selected:
            evidence.extend(CycleDetector().detect(transaction_frame=transactions, min_hops=4, max_hops=6, max_elapsed_hours=24))
        if "GATHER-SCATTER" in selected:
            evidence.extend(GatherScatterDetector().detect(transaction_frame=transactions, min_distinct_incoming_senders=3, min_distinct_outgoing_destinations=1, max_time_window_hours=24))
        if "SCATTER-GATHER" in selected:
            evidence.extend(ScatterGatherDetector().detect(transaction_frame=transactions, min_intermediaries=2, max_time_window_hours=24))

        if "VELOCITY" in selected:
            feature_frame = _load_velocity_features(account_id)
            if feature_frame.empty:
                notes.append("No cached velocity features were available for the selected account.")
            else:
                evidence.extend(
                    VelocityDetector().detect(
                        feature_frame=feature_frame,
                        min_max_transactions_in_hour=10,
                        min_rapid_gap_count=5,
                        min_transactions_per_active_hour=5,
                    )
                )

    assessment = _score_evidence(evidence) if evidence else None
    report = InvestigationReportBuilder().build(assessment, generated_at=datetime(2022, 9, 1, 12, 0)) if assessment is not None else None
    if assessment is None:
        notes.append("No suspicious-pattern evidence was produced for the selected account and filters.")
    return InvestigationBundle(
        source_label=f"Account Investigation: {account_id}",
        is_synthetic=False,
        account_id=account_id,
        evidence=evidence,
        assessment=assessment,
        report=report,
        notes=notes,
    )


def summarize_query_plan(query: str) -> dict[str, object]:
    parsed = QueryParser().parse(query)
    plan = AgentPlanner().plan(parsed)
    return {
        "query": parsed.raw_query,
        "intent": parsed.intent.value,
        "aml_pattern": parsed.aml_pattern.value if parsed.aml_pattern else None,
        "tools": [step.tool_name for step in plan.steps],
        "summary": plan.planning_summary,
    }


def _score_evidence(evidence: list[DetectionEvidence]) -> RiskAssessment:
    assessment = EvidenceAggregator().aggregate(evidence)
    scored = RiskScorer().score(assessment)
    if not scored:
        return RiskAssessment(
            primary_account_id=DEMO_ACCOUNT_ID,
            risk_score=0.0,
            risk_level=RiskLevel.LOW,
            evidence_count=0,
            typologies_detected=[],
            detector_names=[],
            involved_account_ids=[DEMO_ACCOUNT_ID],
            entity_ids=[],
            assessment_start_time=None,
            assessment_end_time=None,
            total_suspicious_amount=0.0,
            total_suspicious_transactions=0,
            contributing_evidence=[],
        )
    return scored[0]


def _load_velocity_features(account_id: str) -> pd.DataFrame:
    store = FeatureStore(IBM_FEATURE_CACHE_DIR)
    if not store.exists():
        return pd.DataFrame()
    features = store.load_features()
    if "account_id" not in features.columns:
        return pd.DataFrame()
    return features[features["account_id"].astype(str) == str(account_id)].reset_index(drop=True)
