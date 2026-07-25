from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd

from agent.tool_contracts import ExecutionStatus, ToolAvailability, ToolMetadata, ToolResult
from detection.evidence import DetectionEvidence
from tools.ibm_dataset_adapter import IBMDatasetAdapter


class CycleDetector:
    LABEL_COLUMNS = {"is_laundering", "Is Laundering", "label", "expected_typology", "annotation_typology"}

    def __init__(
        self,
        transaction_path: str | Path | None = None,
        account_path: str | Path | None = None,
        default_parameters: dict[str, Any] | None = None,
    ) -> None:
        self.adapter = IBMDatasetAdapter(transaction_path=transaction_path, account_path=account_path)
        self.default_parameters = default_parameters or {
            "min_hops": 3,
            "max_hops": 6,
            "max_elapsed_hours": 24,
        }
        self.metadata = ToolMetadata(
            name="cycle_detector",
            description="Detect bounded circular transaction paths within a configurable time window.",
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
            min_hops=int(params.get("min_hops", 3)),
            max_hops=int(params.get("max_hops", 6)),
            max_elapsed_hours=float(params.get("max_elapsed_hours", 24)),
        )
        return ToolResult(status=ExecutionStatus.SUCCESS, summary=f"Detected {len(evidence)} cycle evidence item(s).", data={"evidence": evidence})

    def detect(
        self,
        *,
        transaction_frame: pd.DataFrame | None = None,
        max_rows: int | None = None,
        chunksize: int = 100_000,
        account_ids: Iterable[str] | None = None,
        start_date: str | pd.Timestamp | None = None,
        end_date: str | pd.Timestamp | None = None,
        min_hops: int = 3,
        max_hops: int = 6,
        max_elapsed_hours: float = 24,
    ) -> list[DetectionEvidence]:
        if min_hops < 2 or max_hops < min_hops:
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

        max_elapsed = pd.to_timedelta(float(max_elapsed_hours), unit="h")
        outgoing_map = self._build_outgoing_map(transactions)
        seen_cycles: set[tuple[str, ...]] = set()
        evidence: list[DetectionEvidence] = []

        for start_row in transactions.itertuples(index=False):
            if pd.isna(start_row.timestamp):
                continue
            start_account = str(start_row.sender_account)
            next_account = str(start_row.receiver_account)
            if not start_account or not next_account or start_account == next_account:
                continue

            self._dfs(
                start_account=start_account,
                start_time=start_row.timestamp,
                current_account=next_account,
                current_time=start_row.timestamp,
                path_accounts=[start_account, next_account],
                path_rows=[start_row],
                max_elapsed=max_elapsed,
                max_elapsed_hours=float(max_elapsed_hours),
                min_hops=min_hops,
                max_hops=max_hops,
                outgoing_map=outgoing_map,
                seen_cycles=seen_cycles,
                evidence=evidence,
            )

        return evidence

    def _dfs(
        self,
        *,
        start_account: str,
        start_time: pd.Timestamp,
        current_account: str,
        current_time: pd.Timestamp,
        path_accounts: list[str],
        path_rows: list[Any],
        max_elapsed: pd.Timedelta,
        max_elapsed_hours: float,
        min_hops: int,
        max_hops: int,
        outgoing_map: dict[str, list[Any]],
        seen_cycles: set[tuple[str, ...]],
        evidence: list[DetectionEvidence],
    ) -> None:
        if len(path_rows) > max_hops:
            return

        for row in outgoing_map.get(current_account, []):
            row_time = row.timestamp
            if row_time <= current_time:
                continue
            if row_time - start_time > max_elapsed:
                break

            next_account = str(row.receiver_account)
            if not next_account:
                continue

            if next_account == start_account:
                hop_count = len(path_rows) + 1
                if hop_count < min_hops or hop_count > max_hops:
                    continue
                canonical_key = self._canonical_cycle_key(path_accounts)
                if canonical_key in seen_cycles:
                    continue
                seen_cycles.add(canonical_key)
                cycle_rows = path_rows + [row]
                cycle_path = path_accounts + [start_account]
                evidence.append(self._build_evidence(start_account, cycle_path, cycle_rows, start_time, row_time, min_hops, max_hops, float(max_elapsed_hours)))
                continue

            if next_account in path_accounts:
                continue
            if len(path_rows) + 1 >= max_hops:
                continue

            self._dfs(
                start_account=start_account,
                start_time=start_time,
                current_account=next_account,
                current_time=row_time,
                path_accounts=path_accounts + [next_account],
                path_rows=path_rows + [row],
                max_elapsed=max_elapsed,
                max_elapsed_hours=max_elapsed_hours,
                min_hops=min_hops,
                max_hops=max_hops,
                outgoing_map=outgoing_map,
                seen_cycles=seen_cycles,
                evidence=evidence,
            )

    def _build_evidence(
        self,
        start_account: str,
        cycle_path: list[str],
        cycle_rows: list[Any],
        start_time: pd.Timestamp,
        end_time: pd.Timestamp,
        min_hops: int,
        max_hops: int,
        max_elapsed_hours: float,
    ) -> DetectionEvidence:
        hop_count = len(cycle_rows)
        amounts = [float(self._amount_from_row(row)) for row in cycle_rows]
        references = [str(getattr(row, "transaction_reference")) for row in cycle_rows]
        involved_accounts = self._bounded_unique(cycle_path[:-1])
        entities = self._collect_entities(cycle_rows)
        elapsed_hours = (end_time - start_time).total_seconds() / 3600.0
        ratio = hop_count / max(min_hops, 1)

        return DetectionEvidence(
            detector_name="cycle_detector",
            typology="CYCLE",
            primary_account_id=start_account,
            involved_account_ids=involved_accounts,
            entity_ids=entities,
            start_time=start_time.to_pydatetime(),
            end_time=end_time.to_pydatetime(),
            transaction_count=hop_count,
            total_amount=float(sum(amounts)),
            evidence_strength=min(1.0, ratio),
            severity=self._severity(ratio),
            reasons=[
                f"Observed a {hop_count}-hop circular path within {elapsed_hours:.2f} hours.",
                f"Ordered path: {' -> '.join(cycle_path)}.",
            ],
            detector_parameters={
                "min_hops": min_hops,
                "max_hops": max_hops,
                "max_elapsed_hours": max_elapsed_hours,
            },
            transaction_references=references,
            metadata={
                "cycle_path": cycle_path,
                "ordered_accounts": cycle_path[:-1],
                "hop_count": hop_count,
            },
        )

    def _prepare_transactions(self, transactions: pd.DataFrame) -> pd.DataFrame:
        if transactions.empty:
            return pd.DataFrame()

        prepared = transactions.drop(columns=[column for column in self.LABEL_COLUMNS if column in transactions.columns], errors="ignore").copy()
        required_columns = {"timestamp", "sender_account", "receiver_account"}
        missing_columns = required_columns - set(prepared.columns)
        if missing_columns:
            raise ValueError(f"Cycle detection requires columns: {sorted(missing_columns)}")

        prepared["timestamp"] = pd.to_datetime(prepared["timestamp"], errors="coerce")
        prepared["sender_account"] = prepared["sender_account"].astype("string").str.strip()
        prepared["receiver_account"] = prepared["receiver_account"].astype("string").str.strip()

        if "amount_paid" in prepared.columns:
            prepared["amount_paid"] = pd.to_numeric(prepared["amount_paid"], errors="coerce").fillna(0.0)
        elif "amount_received" in prepared.columns:
            prepared["amount_received"] = pd.to_numeric(prepared["amount_received"], errors="coerce").fillna(0.0)
        else:
            prepared["amount_paid"] = 0.0

        if "transaction_reference" not in prepared.columns:
            prepared["transaction_reference"] = (
                prepared["timestamp"].astype("string")
                + "|"
                + prepared["sender_account"].astype("string")
                + "|"
                + prepared["receiver_account"].astype("string")
            )

        prepared = prepared.dropna(subset=["timestamp", "sender_account", "receiver_account"])
        return prepared.sort_values("timestamp").reset_index(drop=True)

    def _build_outgoing_map(self, transactions: pd.DataFrame) -> dict[str, list[Any]]:
        outgoing_map: dict[str, list[Any]] = {}
        for sender_account, group in transactions.sort_values(["sender_account", "timestamp"]).groupby("sender_account", sort=False):
            outgoing_map[str(sender_account)] = list(group.itertuples(index=False))
        return outgoing_map

    def _amount_from_row(self, row: Any) -> float:
        if hasattr(row, "amount_paid") and pd.notna(getattr(row, "amount_paid")):
            return float(getattr(row, "amount_paid"))
        if hasattr(row, "amount_received") and pd.notna(getattr(row, "amount_received")):
            return float(getattr(row, "amount_received"))
        return 0.0

    def _collect_entities(self, rows: list[Any]) -> list[str]:
        entities: list[str] = []
        for row in rows:
            for field in ("sender_entity_id", "receiver_entity_id", "entity_id"):
                value = getattr(row, field, None)
                if value is not None and pd.notna(value):
                    entities.append(str(value))
        return self._bounded_unique(entities)

    def _bounded_unique(self, values: list[str], limit: int = 20) -> list[str]:
        return list(dict.fromkeys(str(value) for value in values if pd.notna(value)))[:limit]

    def _canonical_cycle_key(self, accounts: list[str]) -> tuple[str, ...]:
        rotations = [tuple(accounts[index:] + accounts[:index]) for index in range(len(accounts))]
        return min(rotations)

    def _severity(self, ratio: float) -> str:
        if ratio >= 2.0:
            return "high"
        if ratio >= 1.25:
            return "medium"
        return "low"
