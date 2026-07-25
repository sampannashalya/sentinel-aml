from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd

from agent.tool_contracts import ExecutionStatus, ToolAvailability, ToolMetadata, ToolResult
from detection.evidence import DetectionEvidence
from tools.ibm_dataset_adapter import IBMDatasetAdapter


class ScatterGatherDetector:
    LABEL_COLUMNS = {"is_laundering", "Is Laundering", "label", "expected_typology", "annotation_typology"}

    def __init__(
        self,
        transaction_path: str | Path | None = None,
        account_path: str | Path | None = None,
        default_parameters: dict[str, Any] | None = None,
    ) -> None:
        self.adapter = IBMDatasetAdapter(transaction_path=transaction_path, account_path=account_path)
        self.default_parameters = default_parameters or {
            "min_intermediaries": 2,
            "max_time_window_hours": 24,
        }
        self.metadata = ToolMetadata(
            name="scatter_gather_detector",
            description="Detect accounts that scatter to intermediaries which then gather toward a common destination.",
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
            min_intermediaries=int(params.get("min_intermediaries", 2)),
            max_time_window_hours=float(params.get("max_time_window_hours", 24)),
        )
        return ToolResult(status=ExecutionStatus.SUCCESS, summary=f"Detected {len(evidence)} scatter-gather evidence item(s).", data={"evidence": evidence})

    def detect(
        self,
        *,
        transaction_frame: pd.DataFrame | None = None,
        max_rows: int | None = None,
        chunksize: int = 100_000,
        account_ids: Iterable[str] | None = None,
        start_date: str | pd.Timestamp | None = None,
        end_date: str | pd.Timestamp | None = None,
        min_intermediaries: int = 2,
        max_time_window_hours: float = 24,
    ) -> list[DetectionEvidence]:
        if min_intermediaries < 2:
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
        outgoing_map = self._build_outgoing_map(transactions)
        evidence: list[DetectionEvidence] = []

        for origin_account, origin_rows in outgoing_map.items():
            scatter_result = self._select_scatter_prefix(origin_rows, min_intermediaries)
            if scatter_result is None:
                continue
            scatter_rows, intermediaries = scatter_result
            scatter_start_time = scatter_rows.iloc[0]["timestamp"]
            scatter_end_time = scatter_rows.iloc[-1]["timestamp"]
            window_end = scatter_start_time + max_window

            downstream_matches = self._find_common_destination(outgoing_map, intermediaries, origin_account, scatter_start_time, window_end)
            if downstream_matches is None:
                continue
            common_destination, downstream_rows, downstream_info = downstream_matches

            relevant_rows = pd.concat([scatter_rows, *downstream_rows.values()], ignore_index=True).sort_values("timestamp")
            total_amount = float(relevant_rows["amount"].sum())
            references = relevant_rows["transaction_reference"].astype(str).tolist()
            intermediate_accounts = self._bounded_unique(list(intermediaries.keys()))
            inbound_reach_count = len(intermediate_accounts)

            evidence.append(
                DetectionEvidence(
                    detector_name="scatter_gather_detector",
                    typology="SCATTER-GATHER",
                    primary_account_id=str(origin_account),
                    involved_account_ids=self._bounded_unique([str(origin_account), *intermediate_accounts, str(common_destination)]),
                    entity_ids=self._collect_entities(relevant_rows),
                    start_time=relevant_rows["timestamp"].min().to_pydatetime(),
                    end_time=relevant_rows["timestamp"].max().to_pydatetime(),
                    transaction_count=int(len(relevant_rows)),
                    total_amount=total_amount,
                    evidence_strength=min(1.0, inbound_reach_count / float(max(min_intermediaries, 1))),
                    severity=self._severity(inbound_reach_count, min_intermediaries),
                    reasons=[
                        f"Origin account {origin_account} scattered funds to {inbound_reach_count} intermediaries that later gathered toward common destination {common_destination} within {max_time_window_hours:g} hours.",
                        "Intermediaries were connected to both the scatter and gather stages.",
                    ],
                    detector_parameters={
                        "min_intermediaries": min_intermediaries,
                        "max_time_window_hours": max_time_window_hours,
                    },
                    transaction_references=references,
                    metadata={
                        "origin_account": str(origin_account),
                        "intermediate_accounts": intermediate_accounts,
                        "common_destination": str(common_destination),
                        "scatter_start_time": scatter_start_time.to_pydatetime(),
                        "scatter_end_time": scatter_end_time.to_pydatetime(),
                        "gather_end_time": relevant_rows["timestamp"].max().to_pydatetime(),
                        "intermediate_paths": {
                            intermediary: {
                                "receipt_time": info["receipt_time"].to_pydatetime(),
                                "downstream_time": info["downstream_row"].timestamp.to_pydatetime(),
                            }
                            for intermediary, info in downstream_info.items()
                        },
                    },
                )
            )

        return evidence

    def _select_scatter_prefix(self, origin_rows: list[Any], min_intermediaries: int) -> tuple[pd.DataFrame, dict[str, pd.Timestamp]] | None:
        selected_intermediaries: dict[str, pd.Timestamp] = {}
        selected_rows: list[Any] = []
        for row in origin_rows:
            if pd.isna(row.timestamp):
                continue
            selected_rows.append(row)
            receiver = str(row.receiver_account)
            if receiver not in selected_intermediaries:
                selected_intermediaries[receiver] = row.timestamp
            if len(selected_intermediaries) >= min_intermediaries:
                return pd.DataFrame(selected_rows), selected_intermediaries
        return None

    def _find_common_destination(
        self,
        outgoing_map: dict[str, list[Any]],
        intermediaries: dict[str, pd.Timestamp],
        origin_account: str,
        scatter_start_time: pd.Timestamp,
        window_end: pd.Timestamp,
    ) -> tuple[str, dict[str, pd.DataFrame], dict[str, dict[str, Any]]] | None:
        per_intermediary_destinations: dict[str, dict[str, dict[str, Any]]] = {}
        for intermediary, receipt_time in intermediaries.items():
            downstream = self._downstream_rows(
                outgoing_map.get(intermediary, []),
                receipt_time,
                window_end,
                forbidden_accounts={origin_account, *intermediaries.keys()},
            )
            if not downstream:
                return None
            per_intermediary_destinations[intermediary] = downstream

        common_destinations = set.intersection(*(set(destinations) for destinations in per_intermediary_destinations.values()))
        if not common_destinations:
            return None

        def score_destination(destination: str) -> tuple[pd.Timestamp, str]:
            latest_time = max(
                destinations[destination]["downstream_row"].timestamp
                for destinations in per_intermediary_destinations.values()
            )
            return latest_time, destination

        common_destination = min(common_destinations, key=score_destination)
        downstream_rows = {
            intermediary: pd.DataFrame([destinations[common_destination]["downstream_row"]])
            for intermediary, destinations in per_intermediary_destinations.items()
        }
        downstream_info = {
            intermediary: destinations[common_destination]
            for intermediary, destinations in per_intermediary_destinations.items()
        }
        return common_destination, downstream_rows, downstream_info

    def _downstream_rows(
        self,
        rows: list[Any],
        receipt_time: pd.Timestamp,
        window_end: pd.Timestamp,
        *,
        forbidden_accounts: set[str],
    ) -> dict[str, dict[str, Any]]:
        matches: dict[str, dict[str, Any]] = {}
        for row in rows:
            if row.timestamp <= receipt_time:
                continue
            if row.timestamp > window_end:
                break
            destination = str(row.receiver_account)
            if not destination or destination in forbidden_accounts:
                continue
            matches.setdefault(destination, {"downstream_row": row, "receipt_time": receipt_time})
        return matches

    def _build_outgoing_map(self, transactions: pd.DataFrame) -> dict[str, list[Any]]:
        outgoing_map: dict[str, list[Any]] = {}
        for sender_account, group in transactions.sort_values(["sender_account", "timestamp"]).groupby("sender_account", sort=False):
            outgoing = group.copy()
            outgoing["amount"] = outgoing["amount_paid"] if "amount_paid" in outgoing.columns else outgoing["amount_received"]
            outgoing_map[str(sender_account)] = list(outgoing.itertuples(index=False))
        return outgoing_map

    def _prepare_transactions(self, transactions: pd.DataFrame) -> pd.DataFrame:
        if transactions.empty:
            return pd.DataFrame()

        prepared = transactions.drop(columns=[column for column in self.LABEL_COLUMNS if column in transactions.columns], errors="ignore").copy()
        required_columns = {"timestamp", "sender_account", "receiver_account"}
        missing_columns = required_columns - set(prepared.columns)
        if missing_columns:
            raise ValueError(f"Scatter-gather detection requires columns: {sorted(missing_columns)}")

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

    def _severity(self, observed: int, threshold: int) -> str:
        ratio = observed / max(threshold, 1)
        if ratio >= 2:
            return "high"
        if ratio >= 1.25:
            return "medium"
        return "low"
