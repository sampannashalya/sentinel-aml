from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd

from agent.tool_contracts import ExecutionStatus, ToolAvailability, ToolMetadata, ToolResult
from detection.evidence import DetectionEvidence
from tools.ibm_dataset_adapter import IBMDatasetAdapter


class GatherScatterDetector:
    LABEL_COLUMNS = {"is_laundering", "Is Laundering", "label", "expected_typology", "annotation_typology"}

    def __init__(
        self,
        transaction_path: str | Path | None = None,
        account_path: str | Path | None = None,
        default_parameters: dict[str, Any] | None = None,
    ) -> None:
        self.adapter = IBMDatasetAdapter(transaction_path=transaction_path, account_path=account_path)
        self.default_parameters = default_parameters or {
            "min_distinct_incoming_senders": 3,
            "min_distinct_outgoing_destinations": 1,
            "max_time_window_hours": 24,
        }
        self.metadata = ToolMetadata(
            name="gather_scatter_detector",
            description="Detect accounts that first gather from many senders and then scatter to destinations within a bounded time window.",
            availability=ToolAvailability.IMPLEMENTED,
            input_type="ExecutionContext",
            output_type="list[DetectionEvidence]",
        )

    def execute(self, context: Any, parameters: dict[str, Any] | None = None) -> ToolResult:
        params = {**self.default_parameters, **(parameters or {})}
        evidence = self.detect(
            transaction_frame=params.get("transaction_frame"),
            max_rows=params.get("max_rows"),
            chunksize=int(params.get("chunksize", 100_000)),
            account_ids=params.get("account_ids"),
            start_date=params.get("start_date"),
            end_date=params.get("end_date"),
            min_distinct_incoming_senders=int(params.get("min_distinct_incoming_senders", 3)),
            min_distinct_outgoing_destinations=int(params.get("min_distinct_outgoing_destinations", 1)),
            max_time_window_hours=float(params.get("max_time_window_hours", 24)),
        )
        return ToolResult(status=ExecutionStatus.SUCCESS, summary=f"Detected {len(evidence)} gather-scatter evidence item(s).", data={"evidence": evidence})

    def detect(
        self,
        *,
        transaction_frame: pd.DataFrame | None = None,
        max_rows: int | None = None,
        chunksize: int = 100_000,
        account_ids: Iterable[str] | None = None,
        start_date: str | pd.Timestamp | None = None,
        end_date: str | pd.Timestamp | None = None,
        min_distinct_incoming_senders: int = 3,
        min_distinct_outgoing_destinations: int = 1,
        max_time_window_hours: float = 24,
    ) -> list[DetectionEvidence]:
        if min_distinct_incoming_senders < 1 or min_distinct_outgoing_destinations < 1:
            return []

        transactions = self._prepare_transactions(
            transaction_frame
            if transaction_frame is not None
            else self.adapter.load_transactions(
                max_rows=max_rows,
                chunksize=chunksize,
                start_date=start_date,
                end_date=end_date,
                account_ids=account_ids,
            )
        )
        if transactions.empty:
            return []

        max_window = pd.to_timedelta(float(max_time_window_hours), unit="h")
        account_events = self._build_account_events(transactions)
        evidence: list[DetectionEvidence] = []

        for account_id, group in account_events.groupby("account_id", sort=False):
            incoming = group[group["direction"] == "in"].sort_values("timestamp").reset_index(drop=True)
            outgoing = group[group["direction"] == "out"].sort_values("timestamp").reset_index(drop=True)
            if incoming.empty or outgoing.empty:
                continue

            gather_result = self._find_gather_prefix(incoming, min_distinct_incoming_senders)
            if gather_result is None:
                continue
            gather_rows, gather_end_time = gather_result
            gather_start_time = gather_rows.iloc[0]["timestamp"]
            scatter_rows = outgoing[(outgoing["timestamp"] > gather_end_time) & (outgoing["timestamp"] <= gather_start_time + max_window)].copy()
            if scatter_rows.empty:
                continue

            outgoing_destinations = self._bounded_unique(scatter_rows["counterparty_account"].astype(str).tolist())
            if len(outgoing_destinations) < min_distinct_outgoing_destinations:
                continue

            relevant_rows = pd.concat([gather_rows, scatter_rows], ignore_index=True).sort_values("timestamp")
            total_amount = float(relevant_rows["amount"].sum())
            incoming_senders = self._bounded_unique(gather_rows["counterparty_account"].astype(str).tolist())
            references = relevant_rows["transaction_reference"].astype(str).tolist()

            evidence.append(
                DetectionEvidence(
                    detector_name="gather_scatter_detector",
                    typology="GATHER-SCATTER",
                    primary_account_id=str(account_id),
                    involved_account_ids=self._bounded_unique([str(account_id), *incoming_senders, *outgoing_destinations]),
                    entity_ids=self._collect_entities(relevant_rows),
                    start_time=relevant_rows["timestamp"].min().to_pydatetime(),
                    end_time=relevant_rows["timestamp"].max().to_pydatetime(),
                    transaction_count=int(len(relevant_rows)),
                    total_amount=total_amount,
                    evidence_strength=min(
                        1.0,
                        max(
                            len(incoming_senders) / float(max(min_distinct_incoming_senders, 1)),
                            len(outgoing_destinations) / float(max(min_distinct_outgoing_destinations, 1)),
                        ),
                    ),
                    severity=self._severity(len(incoming_senders), len(outgoing_destinations), min_distinct_incoming_senders, min_distinct_outgoing_destinations),
                    reasons=[
                        f"Central account {account_id} first gathered funds from {len(incoming_senders)} distinct senders and then scattered to {len(outgoing_destinations)} destinations within {max_time_window_hours:g} hours.",
                        "Gathering occurred before scattering in the selected transaction window.",
                    ],
                    detector_parameters={
                        "min_distinct_incoming_senders": min_distinct_incoming_senders,
                        "min_distinct_outgoing_destinations": min_distinct_outgoing_destinations,
                        "max_time_window_hours": max_time_window_hours,
                    },
                    transaction_references=references,
                    metadata={
                        "central_account": str(account_id),
                        "incoming_senders": incoming_senders,
                        "outgoing_destinations": outgoing_destinations,
                        "gather_start_time": gather_start_time.to_pydatetime(),
                        "gather_end_time": gather_end_time.to_pydatetime(),
                        "scatter_start_time": scatter_rows["timestamp"].min().to_pydatetime(),
                        "scatter_end_time": scatter_rows["timestamp"].max().to_pydatetime(),
                    },
                )
            )

        return evidence

    def _find_gather_prefix(self, incoming: pd.DataFrame, min_distinct_incoming_senders: int) -> tuple[pd.DataFrame, pd.Timestamp] | None:
        seen_senders: set[str] = set()
        rows: list[pd.Series] = []
        for _, row in incoming.iterrows():
            rows.append(row)
            seen_senders.add(str(row["counterparty_account"]))
            if len(seen_senders) >= min_distinct_incoming_senders:
                prefix = pd.DataFrame(rows)
                return prefix, row["timestamp"]
        return None

    def _build_account_events(self, transactions: pd.DataFrame) -> pd.DataFrame:
        incoming = pd.DataFrame(
            {
                "account_id": transactions["receiver_account"],
                "counterparty_account": transactions["sender_account"],
                "direction": "in",
                "timestamp": transactions["timestamp"],
                "amount": transactions["amount_received"] if "amount_received" in transactions.columns else transactions.get("amount_paid", 0.0),
                "transaction_reference": transactions["transaction_reference"],
            }
        )
        outgoing = pd.DataFrame(
            {
                "account_id": transactions["sender_account"],
                "counterparty_account": transactions["receiver_account"],
                "direction": "out",
                "timestamp": transactions["timestamp"],
                "amount": transactions["amount_paid"] if "amount_paid" in transactions.columns else transactions.get("amount_received", 0.0),
                "transaction_reference": transactions["transaction_reference"],
            }
        )
        events = pd.concat([incoming, outgoing], ignore_index=True)
        events["amount"] = pd.to_numeric(events["amount"], errors="coerce").fillna(0.0)
        events["account_id"] = events["account_id"].astype("string").str.strip()
        events["counterparty_account"] = events["counterparty_account"].astype("string").str.strip()
        return events.dropna(subset=["timestamp", "account_id", "counterparty_account"]).reset_index(drop=True)

    def _prepare_transactions(self, transactions: pd.DataFrame) -> pd.DataFrame:
        if transactions.empty:
            return pd.DataFrame()

        prepared = transactions.drop(columns=[column for column in self.LABEL_COLUMNS if column in transactions.columns], errors="ignore").copy()
        required_columns = {"timestamp", "sender_account", "receiver_account"}
        missing_columns = required_columns - set(prepared.columns)
        if missing_columns:
            raise ValueError(f"Gather-scatter detection requires columns: {sorted(missing_columns)}")

        prepared["timestamp"] = pd.to_datetime(prepared["timestamp"], errors="coerce")
        prepared["sender_account"] = prepared["sender_account"].astype("string").str.strip()
        prepared["receiver_account"] = prepared["receiver_account"].astype("string").str.strip()
        if "amount_paid" not in prepared.columns:
            prepared["amount_paid"] = 0.0
        if "amount_received" not in prepared.columns:
            prepared["amount_received"] = 0.0
        prepared["amount_paid"] = pd.to_numeric(prepared["amount_paid"], errors="coerce").fillna(0.0)
        prepared["amount_received"] = pd.to_numeric(prepared["amount_received"], errors="coerce").fillna(0.0)

        if "transaction_reference" not in prepared.columns:
            prepared["transaction_reference"] = (
                prepared["timestamp"].astype("string")
                + "|"
                + prepared["sender_account"].astype("string")
                + "|"
                + prepared["receiver_account"].astype("string")
            )

        return prepared.dropna(subset=["timestamp", "sender_account", "receiver_account"]).reset_index(drop=True)

    def _collect_entities(self, frame: pd.DataFrame) -> list[str]:
        entity_columns = [column for column in ("sender_entity_id", "receiver_entity_id", "entity_id") if column in frame.columns]
        if not entity_columns:
            return []
        values: list[str] = []
        for column in entity_columns:
            values.extend(str(value) for value in frame[column].dropna().tolist() if str(value))
        return self._bounded_unique(values)

    def _bounded_unique(self, values: list[str], limit: int = 20) -> list[str]:
        return list(dict.fromkeys(str(value) for value in values if pd.notna(value)))[:limit]

    def _severity(
        self,
        incoming_sender_count: int,
        outgoing_destination_count: int,
        min_incoming: int,
        min_outgoing: int,
    ) -> str:
        ratio = max(
            incoming_sender_count / max(min_incoming, 1),
            outgoing_destination_count / max(min_outgoing, 1),
        )
        if ratio >= 2:
            return "high"
        if ratio >= 1.25:
            return "medium"
        return "low"
