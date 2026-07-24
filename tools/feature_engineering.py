from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import math

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from agent.tool_contracts import ExecutionStatus, ToolAvailability, ToolMetadata, ToolResult
from config.settings import IBM_ACCOUNT_PATH, IBM_TRANSACTION_PATH
from tools.ibm_dataset_adapter import IBMDatasetAdapter


@dataclass(frozen=True, slots=True)
class FeatureEngineeringConfig:
    low_value_threshold: float = 10_000.0
    near_threshold_band_ratio: float = 0.05
    round_amount_multiple: float = 100.0
    round_tolerance: float = 0.01
    rapid_gap_minutes: float = 60.0
    default_chunksize: int = 100_000
    default_sample_rows: int = 2_000
    include_median: bool = True
    median_exact_threshold: int = 5


class FeatureSet(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    features: pd.DataFrame
    labels: pd.DataFrame
    metadata: dict[str, Any] = Field(default_factory=dict)


@dataclass(slots=True)
class AccountAccumulator:
    transaction_count: int = 0
    outgoing_count: int = 0
    incoming_count: int = 0
    total_outgoing_amount: float = 0.0
    total_incoming_amount: float = 0.0
    max_outgoing_amount: float = 0.0
    max_incoming_amount: float = 0.0
    outgoing_counterparties: set[str] = field(default_factory=set)
    incoming_counterparties: set[str] = field(default_factory=set)
    outgoing_banks: set[str] = field(default_factory=set)
    incoming_banks: set[str] = field(default_factory=set)
    payment_formats: set[str] = field(default_factory=set)
    payment_currencies: set[str] = field(default_factory=set)
    receiving_currencies: set[str] = field(default_factory=set)
    hour_counts: dict[int, int] = field(default_factory=dict)
    day_counts: dict[int, int] = field(default_factory=dict)
    total_gap_seconds: float = 0.0
    gap_count: int = 0
    last_timestamp: pd.Timestamp | None = None
    rapid_gap_count: int = 0
    low_value_outgoing_count: int = 0
    low_value_outgoing_amount: float = 0.0
    near_threshold_outgoing_count: int = 0
    near_threshold_outgoing_value: float = 0.0
    round_amount_count: int = 0
    cross_bank_transaction_count: int = 0
    self_transfer_count: int = 0
    labeled_event_count: int = 0
    laundering_event_count: int = 0
    outgoing_median: "P2MedianEstimator" = field(default_factory=lambda: P2MedianEstimator())
    incoming_median: "P2MedianEstimator" = field(default_factory=lambda: P2MedianEstimator())
    bank_name: str | None = None
    bank_id: str | None = None
    entity_id: str | None = None
    entity_name: str | None = None


@dataclass(slots=True)
class P2MedianEstimator:
    count: int = 0
    initial_values: list[float] = field(default_factory=list)
    q: list[float] = field(default_factory=list)
    n: list[float] = field(default_factory=list)
    np: list[float] = field(default_factory=list)
    dn: list[float] = field(default_factory=list)

    def add(self, value: float) -> None:
        if math.isnan(value):
            return
        value = float(value)
        self.count += 1
        if self.count <= 5:
            self.initial_values.append(value)
            if self.count == 5:
                self.initial_values.sort()
                self.q = self.initial_values[:]
                self.n = [1.0, 2.0, 3.0, 4.0, 5.0]
                self.np = [1.0, 2.0, 3.0, 4.0, 5.0]
                self.dn = [0.0, 0.25, 0.5, 0.75, 1.0]
            return

        self._update(value)

    def add_many(self, values: Any) -> None:
        if values is None:
            return
        for value in values:
            self.add(float(value))

    def estimate(self) -> float:
        if self.count == 0:
            return math.nan
        if self.count <= 5:
            return float(np.median(self.initial_values))
        return float(self.q[2])

    def _update(self, value: float) -> None:
        if value < self.q[0]:
            self.q[0] = value
            cell = 0
        elif value < self.q[1]:
            cell = 0
        elif value < self.q[2]:
            cell = 1
        elif value < self.q[3]:
            cell = 2
        elif value <= self.q[4]:
            cell = 3
        else:
            self.q[4] = value
            cell = 3

        for index in range(cell + 1, 5):
            self.n[index] += 1
        for index in range(5):
            self.np[index] += self.dn[index]

        for index in range(1, 4):
            delta = self.np[index] - self.n[index]
            direction = 0
            if delta >= 1 and self.n[index + 1] - self.n[index] > 1:
                direction = 1
            elif delta <= -1 and self.n[index - 1] - self.n[index] < -1:
                direction = -1
            if direction == 0:
                continue

            q_prev, q_curr, q_next = self.q[index - 1], self.q[index], self.q[index + 1]
            n_prev, n_curr, n_next = self.n[index - 1], self.n[index], self.n[index + 1]
            numerator = (n_curr - n_prev + direction) * (q_next - q_curr) / (n_next - n_curr)
            numerator += (n_next - n_curr - direction) * (q_curr - q_prev) / (n_curr - n_prev)
            q_new = q_curr + direction * numerator / (n_next - n_prev)
            if q_prev < q_new < q_next:
                self.q[index] = q_new
            else:
                step_index = index + direction
                self.q[index] = q_curr + direction * (self.q[step_index] - q_curr) / (self.n[step_index] - n_curr)
            self.n[index] += direction


class FeatureEngineeringTool:
    def __init__(self, config: FeatureEngineeringConfig | None = None) -> None:
        self.config = config or FeatureEngineeringConfig()
        self.metadata = ToolMetadata(
            name="feature_engineering",
            description="Create account-level IBM AML features from transaction behavior while keeping labels separate.",
            availability=ToolAvailability.IMPLEMENTED,
            input_type="ExecutionContext",
            output_type="FeatureSet",
        )

    def execute(self, context: Any, parameters: dict[str, Any] | None = None) -> ToolResult:
        parameters = parameters or {}
        transaction_path = parameters.get("transaction_path") or getattr(context, "dataset_reference", None) or IBM_TRANSACTION_PATH
        account_path = parameters.get("account_path") or IBM_ACCOUNT_PATH
        max_rows = parameters.get("max_rows", self.config.default_sample_rows)
        chunksize = int(parameters.get("chunksize", self.config.default_chunksize))
        low_value_threshold = float(parameters.get("low_value_threshold", self.config.low_value_threshold))
        near_threshold_band_ratio = float(parameters.get("near_threshold_band_ratio", self.config.near_threshold_band_ratio))
        round_amount_multiple = float(parameters.get("round_amount_multiple", self.config.round_amount_multiple))
        round_tolerance = float(parameters.get("round_tolerance", self.config.round_tolerance))
        rapid_gap_minutes = float(parameters.get("rapid_gap_minutes", self.config.rapid_gap_minutes))
        include_median = bool(parameters.get("include_median", self.config.include_median))

        builder = IBMFeatureEngineer(
            transaction_path=transaction_path,
            account_path=account_path,
            config=FeatureEngineeringConfig(
                low_value_threshold=low_value_threshold,
                near_threshold_band_ratio=near_threshold_band_ratio,
                round_amount_multiple=round_amount_multiple,
                round_tolerance=round_tolerance,
                rapid_gap_minutes=rapid_gap_minutes,
                default_chunksize=chunksize,
                default_sample_rows=max_rows if max_rows is not None else self.config.default_sample_rows,
                include_median=include_median,
            ),
        )

        feature_set = builder.build_feature_set(max_rows=max_rows, chunksize=chunksize)
        preview = feature_set.features.head(5).to_dict(orient="records")
        return ToolResult(
            status=ExecutionStatus.SUCCESS,
            summary=f"Built features for {len(feature_set.features)} accounts from {feature_set.metadata.get('transactions_processed', 0)} transactions.",
            data={
                "feature_set": feature_set,
                "feature_preview": preview,
                "feature_columns": list(feature_set.features.columns),
                "label_columns": list(feature_set.labels.columns),
                "metadata": feature_set.metadata,
            },
        )


class IBMFeatureEngineer:
    def __init__(
        self,
        transaction_path: str | Path = IBM_TRANSACTION_PATH,
        account_path: str | Path = IBM_ACCOUNT_PATH,
        config: FeatureEngineeringConfig | None = None,
    ) -> None:
        self.config = config or FeatureEngineeringConfig()
        self.adapter = IBMDatasetAdapter(transaction_path=transaction_path, account_path=account_path)

    def build_feature_set(
        self,
        max_rows: int | None = None,
        chunksize: int | None = None,
        start_date: str | pd.Timestamp | None = None,
        end_date: str | pd.Timestamp | None = None,
        account_ids: list[str] | None = None,
    ) -> FeatureSet:
        effective_chunksize = chunksize or self.config.default_chunksize
        transactions_processed = 0
        acc: dict[str, AccountAccumulator] = defaultdict(AccountAccumulator)
        label_acc: dict[str, dict[str, int]] = defaultdict(lambda: {"event_count": 0, "laundering_event_count": 0})

        account_metadata = self.adapter.load_account_metadata().set_index("account_number")

        for chunk in self.adapter.iter_transaction_chunks(
            chunksize=effective_chunksize,
            max_rows=max_rows,
            start_date=start_date,
            end_date=end_date,
            account_ids=account_ids,
        ):
            if chunk.empty:
                continue

            transactions_processed += len(chunk)
            self._process_chunk(chunk, acc, label_acc)

        features = self._finalize_features(acc, account_metadata)
        labels = self._finalize_labels(label_acc)

        metadata = {
            "source": "IBM HI-Small synthetic AML dataset",
            "transactions_processed": int(transactions_processed),
            "accounts_processed": int(len(features)),
            "feature_count": int(len(features.columns)),
            "output_memory_bytes": int(features.memory_usage(deep=True).sum() + labels.memory_usage(deep=True).sum()),
            "label_policy": "is_laundering retained only in separate labels; never included in engineered features",
            "chunked_processing": True,
            "chunksize": int(effective_chunksize),
            "max_rows": max_rows,
        }
        return FeatureSet(features=features, labels=labels, metadata=metadata)

    def _process_chunk(
        self,
        chunk: pd.DataFrame,
        acc: dict[str, AccountAccumulator],
        label_acc: dict[str, dict[str, int]],
    ) -> None:
        chunk = chunk.sort_values("timestamp").reset_index(drop=True)
        outgoing = self._build_event_frame(chunk, role="outgoing")
        incoming = self._build_event_frame(chunk, role="incoming")
        self._update_grouped_accumulators(outgoing, acc, label_acc)
        self._update_grouped_accumulators(incoming, acc, label_acc)
        time_frames = [frame[["account_id", "timestamp"]] for frame in (outgoing, incoming) if not frame.empty]
        bucket_frames = [frame for frame in (outgoing, incoming) if not frame.empty]
        if time_frames:
            self._update_time_gaps(pd.concat(time_frames, ignore_index=True), acc)
        if bucket_frames:
            self._update_bucket_counts(pd.concat(bucket_frames, ignore_index=True), acc)

    def _build_event_frame(self, chunk: pd.DataFrame, role: str) -> pd.DataFrame:
        if role not in {"outgoing", "incoming"}:
            raise ValueError("role must be outgoing or incoming")

        if role == "outgoing":
            self_transfer_mask = chunk["sender_account"] == chunk["receiver_account"]
            events = pd.DataFrame(
                {
                    "account_id": chunk["sender_account"],
                    "counterparty_account": chunk["receiver_account"],
                    "counterparty_bank": chunk["to_bank"],
                    "timestamp": chunk["timestamp"],
                    "amount": chunk["amount_paid"],
                    "payment_format": chunk["payment_format"],
                    "payment_currency": chunk["payment_currency"],
                    "receiving_currency": chunk["receiving_currency"],
                    "cross_bank_flag": (chunk["from_bank"] != chunk["to_bank"]).astype(int),
                    "self_transfer_flag": self_transfer_mask.astype(int),
                    "low_value_outgoing_flag": (chunk["amount_paid"] < self.config.low_value_threshold).astype(int),
                    "low_value_outgoing_amount": chunk["amount_paid"].where(chunk["amount_paid"] < self.config.low_value_threshold, 0.0),
                    "near_threshold_outgoing_flag": self._near_threshold_mask(chunk["amount_paid"]),
                    "near_threshold_outgoing_value": self._near_threshold_value(chunk["amount_paid"]),
                    "round_amount_flag": self._round_amount_mask(chunk["amount_paid"]),
                    "outgoing_flag": 1,
                    "incoming_flag": self_transfer_mask.astype(int),
                    "outgoing_amount": chunk["amount_paid"],
                    "incoming_amount": chunk["amount_paid"].where(self_transfer_mask, 0.0),
                    "is_laundering": chunk["is_laundering"],
                }
            )
        else:
            self_transfer_mask = chunk["sender_account"] == chunk["receiver_account"]
            if self_transfer_mask.any():
                chunk = chunk[~self_transfer_mask].copy()
            if chunk.empty:
                return pd.DataFrame(columns=[
                    "account_id",
                    "counterparty_account",
                    "counterparty_bank",
                    "timestamp",
                    "amount",
                    "payment_format",
                    "payment_currency",
                    "receiving_currency",
                    "cross_bank_flag",
                    "self_transfer_flag",
                    "low_value_outgoing_flag",
                    "low_value_outgoing_amount",
                    "near_threshold_outgoing_flag",
                    "near_threshold_outgoing_value",
                    "round_amount_flag",
                    "outgoing_flag",
                    "incoming_flag",
                    "outgoing_amount",
                    "incoming_amount",
                    "is_laundering",
                ])
            events = pd.DataFrame(
                {
                    "account_id": chunk["receiver_account"],
                    "counterparty_account": chunk["sender_account"],
                    "counterparty_bank": chunk["from_bank"],
                    "timestamp": chunk["timestamp"],
                    "amount": chunk["amount_received"],
                    "payment_format": chunk["payment_format"],
                    "payment_currency": chunk["payment_currency"],
                    "receiving_currency": chunk["receiving_currency"],
                    "cross_bank_flag": (chunk["from_bank"] != chunk["to_bank"]).astype(int),
                    "self_transfer_flag": 0,
                    "low_value_outgoing_flag": 0,
                    "low_value_outgoing_amount": 0.0,
                    "near_threshold_outgoing_flag": 0,
                    "near_threshold_outgoing_value": 0.0,
                    "round_amount_flag": 0,
                    "outgoing_flag": 0,
                    "incoming_flag": 1,
                    "outgoing_amount": 0.0,
                    "incoming_amount": chunk["amount_received"],
                    "is_laundering": chunk["is_laundering"],
                }
            )

        return events.reset_index(drop=True)

    def _update_grouped_accumulators(
        self,
        events: pd.DataFrame,
        acc: dict[str, AccountAccumulator],
        label_acc: dict[str, dict[str, int]],
    ) -> None:
        if events.empty:
            return

        grouped = events.groupby("account_id", sort=False)
        event_summary = grouped.agg(
            transaction_count=("account_id", "size"),
            outgoing_count=("outgoing_flag", "sum"),
            incoming_count=("incoming_flag", "sum"),
            total_outgoing_amount=("outgoing_amount", "sum"),
            total_incoming_amount=("incoming_amount", "sum"),
            max_outgoing_amount=("outgoing_amount", "max"),
            max_incoming_amount=("incoming_amount", "max"),
            low_value_outgoing_count=("low_value_outgoing_flag", "sum"),
            low_value_outgoing_amount=("low_value_outgoing_amount", "sum"),
            near_threshold_outgoing_count=("near_threshold_outgoing_flag", "sum"),
            near_threshold_outgoing_value=("near_threshold_outgoing_value", "sum"),
            round_amount_count=("round_amount_flag", "sum"),
            cross_bank_transaction_count=("cross_bank_flag", "sum"),
            self_transfer_count=("self_transfer_flag", "sum"),
            labeled_event_count=("is_laundering", "size"),
            laundering_event_count=("is_laundering", "sum"),
        )

        outgoing_events = events[events["outgoing_flag"] == 1]
        incoming_events = events[events["incoming_flag"] == 1]
        outgoing_counterparties = outgoing_events.groupby("account_id", sort=False)["counterparty_account"].unique()
        incoming_counterparties = incoming_events.groupby("account_id", sort=False)["counterparty_account"].unique()
        outgoing_banks = outgoing_events.groupby("account_id", sort=False)["counterparty_bank"].unique()
        incoming_banks = incoming_events.groupby("account_id", sort=False)["counterparty_bank"].unique()
        payment_formats = grouped["payment_format"].unique()
        payment_currencies = grouped["payment_currency"].unique()
        receiving_currencies = grouped["receiving_currency"].unique()

        outgoing_medians = outgoing_events.groupby("account_id", sort=False)["outgoing_amount"].agg(list) if self.config.include_median else pd.Series(dtype=object)
        incoming_medians = incoming_events.groupby("account_id", sort=False)["incoming_amount"].agg(list) if self.config.include_median else pd.Series(dtype=object)

        for row in event_summary.itertuples():
            account_id = row.Index
            state = acc[account_id]
            state.transaction_count += int(row.transaction_count)
            state.outgoing_count += int(row.outgoing_count)
            state.incoming_count += int(row.incoming_count)
            state.total_outgoing_amount += float(row.total_outgoing_amount)
            state.total_incoming_amount += float(row.total_incoming_amount)
            state.max_outgoing_amount = max(state.max_outgoing_amount, float(row.max_outgoing_amount))
            state.max_incoming_amount = max(state.max_incoming_amount, float(row.max_incoming_amount))
            state.low_value_outgoing_count += int(row.low_value_outgoing_count)
            state.low_value_outgoing_amount += float(row.low_value_outgoing_amount)
            state.near_threshold_outgoing_count += int(row.near_threshold_outgoing_count)
            state.near_threshold_outgoing_value += float(row.near_threshold_outgoing_value)
            state.round_amount_count += int(row.round_amount_count)
            state.cross_bank_transaction_count += int(row.cross_bank_transaction_count)
            state.self_transfer_count += int(row.self_transfer_count)
            state.labeled_event_count += int(row.labeled_event_count)
            state.laundering_event_count += int(row.laundering_event_count)

            if account_id in outgoing_counterparties.index:
                state.outgoing_counterparties.update(self._non_null_values(outgoing_counterparties.loc[account_id]))
            if account_id in incoming_counterparties.index:
                state.incoming_counterparties.update(self._non_null_values(incoming_counterparties.loc[account_id]))
            if account_id in outgoing_banks.index:
                state.outgoing_banks.update(self._non_null_values(outgoing_banks.loc[account_id]))
            if account_id in incoming_banks.index:
                state.incoming_banks.update(self._non_null_values(incoming_banks.loc[account_id]))
            state.payment_formats.update(self._non_null_values(payment_formats.get(account_id, [])))
            state.payment_currencies.update(self._non_null_values(payment_currencies.get(account_id, [])))
            state.receiving_currencies.update(self._non_null_values(receiving_currencies.get(account_id, [])))

            if self.config.include_median:
                if account_id in outgoing_medians.index:
                    state.outgoing_median.add_many(outgoing_medians.loc[account_id])
                if account_id in incoming_medians.index:
                    state.incoming_median.add_many(incoming_medians.loc[account_id])

            label_state = label_acc[account_id]
            label_state["event_count"] += int(row.labeled_event_count)
            label_state["laundering_event_count"] += int(row.laundering_event_count)

    def _update_time_gaps(self, events: pd.DataFrame, acc: dict[str, AccountAccumulator]) -> None:
        if events.empty:
            return

        sorted_events = events.dropna(subset=["timestamp"]).sort_values(["account_id", "timestamp"])
        if sorted_events.empty:
            return

        previous_timestamps = pd.Series({account_id: state.last_timestamp for account_id, state in acc.items() if state.last_timestamp is not None})
        grouped = sorted_events.groupby("account_id", sort=False)
        last_timestamps = grouped["timestamp"].last()

        for account_id, group in grouped:
            timestamps = group["timestamp"].reset_index(drop=True)
            prev_timestamp = timestamps.shift(1)
            if account_id in previous_timestamps.index:
                prev_timestamp.iloc[0] = previous_timestamps.loc[account_id]
            gaps = (timestamps - prev_timestamp).dt.total_seconds()
            positive_gaps = gaps[gaps > 0]
            if positive_gaps.empty:
                acc[account_id].last_timestamp = last_timestamps.loc[account_id]
                continue

            state = acc[account_id]
            rapid_threshold_seconds = self.config.rapid_gap_minutes * 60.0
            state.total_gap_seconds += float(positive_gaps.sum())
            state.gap_count += int(positive_gaps.count())
            state.rapid_gap_count += int((positive_gaps <= rapid_threshold_seconds).sum())
            state.last_timestamp = last_timestamps.loc[account_id]

    def _update_bucket_counts(self, events: pd.DataFrame, acc: dict[str, AccountAccumulator]) -> None:
        if events.empty:
            return

        valid_events = events.dropna(subset=["timestamp"])
        if valid_events.empty:
            return

        hour_keys = valid_events["timestamp"].astype("int64") // (60 * 60 * 1_000_000_000)
        day_keys = valid_events["timestamp"].astype("int64") // (24 * 60 * 60 * 1_000_000_000)

        hour_groups = valid_events.assign(hour_key=hour_keys).groupby(["account_id", "hour_key"]).size()
        day_groups = valid_events.assign(day_key=day_keys).groupby(["account_id", "day_key"]).size()

        for (account_id, bucket), count in hour_groups.items():
            state = acc[account_id]
            state.hour_counts[bucket] = state.hour_counts.get(bucket, 0) + int(count)

        for (account_id, bucket), count in day_groups.items():
            state = acc[account_id]
            state.day_counts[bucket] = state.day_counts.get(bucket, 0) + int(count)

    def _finalize_features(self, acc: dict[str, AccountAccumulator], account_metadata: pd.DataFrame) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for account_id, state in acc.items():
            unique_outgoing_counterparties = self._non_self_count(state.outgoing_counterparties, account_id)
            unique_incoming_counterparties = self._non_self_count(state.incoming_counterparties, account_id)
            unique_counterparties = len((state.outgoing_counterparties | state.incoming_counterparties) - {account_id})
            unique_outgoing_banks = len(state.outgoing_banks)
            unique_incoming_banks = len(state.incoming_banks)
            active_hours = len(state.hour_counts)
            active_days = len(state.day_counts)
            mean_outgoing_amount = state.total_outgoing_amount / state.outgoing_count if state.outgoing_count else 0.0
            mean_incoming_amount = state.total_incoming_amount / state.incoming_count if state.incoming_count else 0.0
            average_transactions_per_active_day = state.transaction_count / active_days if active_days else 0.0
            transactions_per_active_hour = state.transaction_count / active_hours if active_hours else 0.0
            max_transactions_in_hour = max(state.hour_counts.values()) if state.hour_counts else 0
            max_transactions_in_day = max(state.day_counts.values()) if state.day_counts else 0
            mean_time_between_transactions_minutes = state.total_gap_seconds / state.gap_count / 60.0 if state.gap_count else np.nan

            row = {
                "account_id": account_id,
                "transaction_count": state.transaction_count,
                "outgoing_count": state.outgoing_count,
                "incoming_count": state.incoming_count,
                "total_outgoing_amount": state.total_outgoing_amount,
                "total_incoming_amount": state.total_incoming_amount,
                "mean_outgoing_amount": mean_outgoing_amount,
                "mean_incoming_amount": mean_incoming_amount,
                "median_outgoing_amount": state.outgoing_median.estimate() if self.config.include_median else np.nan,
                "median_incoming_amount": state.incoming_median.estimate() if self.config.include_median else np.nan,
                "max_outgoing_amount": state.max_outgoing_amount,
                "max_incoming_amount": state.max_incoming_amount,
                "unique_outgoing_counterparties": unique_outgoing_counterparties,
                "unique_incoming_counterparties": unique_incoming_counterparties,
                "unique_counterparties": unique_counterparties,
                "unique_outgoing_banks": unique_outgoing_banks,
                "unique_incoming_banks": unique_incoming_banks,
                "fan_out_ratio": unique_outgoing_counterparties / state.outgoing_count if state.outgoing_count else 0.0,
                "fan_in_ratio": unique_incoming_counterparties / state.incoming_count if state.incoming_count else 0.0,
                "active_hours": active_hours,
                "active_days": active_days,
                "max_transactions_in_hour": max_transactions_in_hour,
                "max_transactions_in_day": max_transactions_in_day,
                "average_transactions_per_active_day": average_transactions_per_active_day,
                "transactions_per_active_hour": transactions_per_active_hour,
                "mean_time_between_transactions_minutes": mean_time_between_transactions_minutes,
                "payment_format_count": len(state.payment_formats),
                "payment_currency_count": len(state.payment_currencies),
                "receiving_currency_count": len(state.receiving_currencies),
                "low_value_outgoing_count": state.low_value_outgoing_count,
                "low_value_outgoing_amount": state.low_value_outgoing_amount,
                "near_threshold_outgoing_count": state.near_threshold_outgoing_count,
                "near_threshold_outgoing_value": state.near_threshold_outgoing_value,
                "round_amount_count": state.round_amount_count,
                "cross_bank_transaction_count": state.cross_bank_transaction_count,
                "self_transfer_count": state.self_transfer_count,
                "rapid_gap_count": state.rapid_gap_count,
                "rapid_gap_ratio": state.rapid_gap_count / max(state.transaction_count - 1, 1),
                "bank_name": state.bank_name,
                "bank_id": state.bank_id,
                "entity_id": state.entity_id,
                "entity_name": state.entity_name,
            }
            rows.append(row)

        features = pd.DataFrame(rows)
        if features.empty:
            return features

        features = features.merge(
            account_metadata.reset_index().rename(
                columns={
                    "account_number": "account_id",
                    "bank_name": "bank_name_meta",
                    "bank_id": "bank_id_meta",
                    "entity_id": "entity_id_meta",
                    "entity_name": "entity_name_meta",
                }
            ),
            on="account_id",
            how="left",
        )
        for base_column, meta_column in [
            ("bank_name", "bank_name_meta"),
            ("bank_id", "bank_id_meta"),
            ("entity_id", "entity_id_meta"),
            ("entity_name", "entity_name_meta"),
        ]:
            features[base_column] = features[base_column].where(features[base_column].notna(), features[meta_column])
        features = features.drop(columns=["bank_name_meta", "bank_id_meta", "entity_id_meta", "entity_name_meta"])

        if not self.config.include_median:
            features["median_outgoing_amount"] = np.nan
            features["median_incoming_amount"] = np.nan

        features = features.sort_values("account_id").reset_index(drop=True)
        return features

    def _finalize_labels(self, label_acc: dict[str, dict[str, int]]) -> pd.DataFrame:
        rows = []
        for account_id, stats in label_acc.items():
            event_count = stats["event_count"]
            laundering_event_count = stats["laundering_event_count"]
            rows.append(
                {
                    "account_id": account_id,
                    "event_count": event_count,
                    "laundering_event_count": laundering_event_count,
                    "laundering_event_rate": laundering_event_count / event_count if event_count else 0.0,
                    "has_laundering_label": laundering_event_count > 0,
                }
            )
        labels = pd.DataFrame(rows)
        if not labels.empty:
            labels = labels.sort_values("account_id").reset_index(drop=True)
        return labels

    def _median_from_state(self, state: AccountAccumulator, direction: str) -> float:
        if not self.config.include_median:
            return np.nan
        estimator = state.outgoing_median if direction == "outgoing" else state.incoming_median
        return estimator.estimate()

    def _non_self_count(self, values: set[str], account_id: str) -> int:
        return len(values - {account_id})

    def _non_null_values(self, values: Any) -> list[str]:
        if values is None:
            return []
        if isinstance(values, str):
            return [values]
        if isinstance(values, (list, tuple, np.ndarray, pd.Series)) or hasattr(values, "tolist"):
            return [str(value) for value in list(values.tolist() if hasattr(values, "tolist") else values) if pd.notna(value)]
        try:
            return [str(values)] if pd.notna(values) else []
        except (TypeError, ValueError):
            return [str(value) for value in list(values) if pd.notna(value)]

    def _near_threshold_mask(self, amounts: pd.Series) -> pd.Series:
        lower_bound = self.config.low_value_threshold * (1 - self.config.near_threshold_band_ratio)
        return ((amounts >= lower_bound) & (amounts < self.config.low_value_threshold)).astype(int)

    def _near_threshold_value(self, amounts: pd.Series) -> pd.Series:
        return amounts.where(self._near_threshold_mask(amounts).astype(bool), 0.0)

    def _round_amount_mask(self, amounts: pd.Series) -> pd.Series:
        multiples = np.mod(amounts, self.config.round_amount_multiple)
        return (np.isclose(multiples, 0.0, atol=self.config.round_tolerance) | np.isclose(multiples, self.config.round_amount_multiple, atol=self.config.round_tolerance)).astype(int)