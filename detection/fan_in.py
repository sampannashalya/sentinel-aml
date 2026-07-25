from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd

from agent.tool_contracts import ExecutionStatus, ToolAvailability, ToolMetadata, ToolResult
from detection.evidence import DetectionEvidence
from tools.ibm_dataset_adapter import IBMDatasetAdapter


class FanInDetector:
    LABEL_COLUMNS = {"is_laundering", "Is Laundering", "label", "expected_typology", "annotation_typology"}

    def __init__(
        self,
        transaction_path: str | Path | None = None,
        account_path: str | Path | None = None,
        default_parameters: dict[str, Any] | None = None,
    ) -> None:
        self.adapter = IBMDatasetAdapter(transaction_path=transaction_path, account_path=account_path)
        self.default_parameters = default_parameters or {
            "min_distinct_senders": 5,
            "time_window_hours": 24,
            "min_transaction_count": 5,
        }
        self.metadata = ToolMetadata(
            name="fan_in_detector",
            description="Detect accounts receiving from many distinct senders within a configurable time window.",
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
            min_distinct_senders=int(params.get("min_distinct_senders", 5)),
            time_window_hours=float(params.get("time_window_hours", 24)),
            min_transaction_count=int(params.get("min_transaction_count", 5)),
        )
        return ToolResult(status=ExecutionStatus.SUCCESS, summary=f"Detected {len(evidence)} fan-in evidence item(s).", data={"evidence": evidence})

    def detect(
        self,
        *,
        transaction_frame: pd.DataFrame | None = None,
        max_rows: int | None = None,
        chunksize: int = 100_000,
        account_ids: Iterable[str] | None = None,
        start_date: str | pd.Timestamp | None = None,
        end_date: str | pd.Timestamp | None = None,
        min_distinct_senders: int = 5,
        time_window_hours: float = 24,
        min_transaction_count: int = 5,
    ) -> list[DetectionEvidence]:
        window = pd.to_timedelta(float(time_window_hours) * 3600.0, unit="s")
        if transaction_frame is None:
            transactions = self.adapter.load_transactions(
                max_rows=max_rows,
                chunksize=chunksize,
                start_date=start_date,
                end_date=end_date,
                account_ids=account_ids,
            )
        else:
            transactions = transaction_frame.copy()
        transactions = self._prepare_transactions(transactions)
        if transactions.empty:
            return []

        candidate_slices = self._candidate_windows(
            transactions=transactions,
            account_column="receiver_account",
            counterparty_column="sender_account",
            window=window,
            min_distinct_counterparties=min_distinct_senders,
            min_transaction_count=min_transaction_count,
        )
        evidence: list[DetectionEvidence] = []
        for account_id, slice_frame in candidate_slices:
            sender_count = int(slice_frame["sender_account"].nunique())
            transaction_count = int(len(slice_frame))
            total_amount = float(slice_frame["amount_received"].sum())

            evidence.append(
                DetectionEvidence(
                    detector_name="fan_in_detector",
                    typology="FAN-IN",
                    primary_account_id=account_id,
                    involved_account_ids=self._bounded_unique(slice_frame["sender_account"].tolist()),
                    entity_ids=self._bounded_unique(slice_frame.get("sender_entity_id", pd.Series(dtype=object)).dropna().tolist()),
                    start_time=slice_frame["timestamp"].min().to_pydatetime(),
                    end_time=slice_frame["timestamp"].max().to_pydatetime(),
                    transaction_count=transaction_count,
                    total_amount=total_amount,
                    evidence_strength=min(1.0, sender_count / float(min_distinct_senders)),
                    severity=self._severity(sender_count, min_distinct_senders),
                    reasons=[
                        f"Destination account received funds from {sender_count} distinct senders within a {time_window_hours:g} hour window, meeting the configured threshold of {min_distinct_senders}.",
                        f"{transaction_count} incoming transactions totaled {total_amount:.2f}.",
                    ],
                    detector_parameters={
                        "min_distinct_senders": min_distinct_senders,
                        "time_window_hours": time_window_hours,
                        "min_transaction_count": min_transaction_count,
                    },
                    transaction_references=self._bounded_unique(slice_frame["transaction_reference"].tolist()),
                    metadata={"sender_count": sender_count},
                )
            )
        return evidence

    def _prepare_transactions(self, transactions: pd.DataFrame) -> pd.DataFrame:
        if transactions.empty:
            return pd.DataFrame()

        prepared = transactions.drop(columns=[column for column in self.LABEL_COLUMNS if column in transactions.columns], errors="ignore").copy()
        required_columns = {"timestamp", "sender_account", "receiver_account"}
        missing_columns = required_columns - set(prepared.columns)
        if missing_columns:
            raise ValueError(f"Fan-in detection requires columns: {sorted(missing_columns)}")

        prepared["timestamp"] = pd.to_datetime(prepared["timestamp"], errors="coerce")
        prepared["sender_account"] = prepared["sender_account"].astype("string").str.strip()
        prepared["receiver_account"] = prepared["receiver_account"].astype("string").str.strip()
        if "amount_received" not in prepared.columns:
            prepared["amount_received"] = 0.0
        prepared["amount_received"] = pd.to_numeric(prepared["amount_received"], errors="coerce").fillna(0.0)
        if "transaction_reference" not in prepared.columns:
            prepared["transaction_reference"] = "row-" + prepared.index.astype(str)
        prepared = prepared.dropna(subset=["timestamp", "sender_account", "receiver_account"])
        return prepared.reset_index(drop=True)

    def _candidate_windows(
        self,
        *,
        transactions: pd.DataFrame,
        account_column: str,
        counterparty_column: str,
        window: pd.Timedelta,
        min_distinct_counterparties: int,
        min_transaction_count: int,
    ) -> list[tuple[str, pd.DataFrame]]:
        if transactions.empty:
            return []
        incoming = transactions.sort_values([account_column, "timestamp"]).reset_index(drop=True)
        candidates: list[tuple[str, pd.DataFrame]] = []
        for account_id, group in incoming.groupby(account_column, sort=False):
            if group.empty:
                continue
            best_window = self._best_counterparty_window(
                group=group.reset_index(drop=True),
                counterparty_column=counterparty_column,
                window=window,
                min_distinct_counterparties=min_distinct_counterparties,
                min_transaction_count=min_transaction_count,
            )
            if best_window is None:
                continue
            start_index, end_index = best_window
            candidates.append((str(account_id), group.iloc[start_index : end_index + 1].copy()))
        return candidates

    def _best_counterparty_window(
        self,
        *,
        group: pd.DataFrame,
        counterparty_column: str,
        window: pd.Timedelta,
        min_distinct_counterparties: int,
        min_transaction_count: int,
    ) -> tuple[int, int] | None:
        timestamps = group["timestamp"].tolist()
        counterparties = group[counterparty_column].astype(str).tolist()
        counts: Counter[str] = Counter()
        left = 0
        best: tuple[int, int, int, int] | None = None

        for right, timestamp in enumerate(timestamps):
            counts[counterparties[right]] += 1
            while timestamp - timestamps[left] > window:
                left_counterparty = counterparties[left]
                counts[left_counterparty] -= 1
                if counts[left_counterparty] <= 0:
                    del counts[left_counterparty]
                left += 1

            transaction_count = right - left + 1
            distinct_count = len(counts)
            if distinct_count < min_distinct_counterparties or transaction_count < min_transaction_count:
                continue

            candidate = (distinct_count, transaction_count, left, right)
            if best is None or candidate[:2] > best[:2]:
                best = candidate

        if best is None:
            return None
        return best[2], best[3]

    def _bounded_unique(self, values: list[str], limit: int = 20) -> list[str]:
        unique_values = list(dict.fromkeys(str(value) for value in values if pd.notna(value)))
        return unique_values[:limit]

    def _severity(self, observed: int, threshold: int) -> str:
        ratio = observed / max(threshold, 1)
        if ratio >= 2:
            return "high"
        if ratio >= 1.25:
            return "medium"
        return "low"
