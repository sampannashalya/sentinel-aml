from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from tools.feature_engineering import FeatureEngineeringConfig, IBMFeatureEngineer
from tools.feature_store import FeatureStore
from tools.ibm_dataset_adapter import IBMDatasetAdapter


TRANSACTION_COLUMNS = [
    "Timestamp",
    "From Bank",
    "Account",
    "To Bank",
    "Account.1",
    "Amount Received",
    "Receiving Currency",
    "Amount Paid",
    "Payment Currency",
    "Payment Format",
    "Is Laundering",
]


ACCOUNT_COLUMNS = ["Bank Name", "Bank ID", "Account Number", "Entity ID", "Entity Name"]


def write_ibm_fixture(tmp_path: Path) -> tuple[Path, Path]:
    transaction_path = tmp_path / "HI-Small_Trans.csv"
    account_path = tmp_path / "HI-Small_accounts.csv"

    transaction_rows = [
        ["2022/09/01 00:00", "001", "A1", "002", "B1", 95, "US Dollar", 95, "US Dollar", "ACH", 1],
        ["2022/09/01 00:30", "001", "A1", "003", "C1", 110, "US Dollar", 110, "US Dollar", "Wire", 0],
        ["2022/09/01 01:00", "001", "A1", "001", "A1", 200, "US Dollar", 200, "US Dollar", "Card", 1],
        ["2022/09/02 12:00", "002", "B1", "001", "A1", 50, "Euro", 50, "Euro", "ACH", 0],
        ["2022/09/02 13:00", "003", "C1", "001", "A1", 90, "Euro", 90, "Euro", "Cash", 0],
        ["2022/09/02 14:00", "001", "A1", "003", "C1", 90, "US Dollar", 90, "US Dollar", "ACH", 0],
    ]
    account_rows = [
        ["Bank #1", "001", "A1", "E1", "Entity A"],
        ["Bank #2", "002", "B1", "E2", "Entity B"],
        ["Bank #3", "003", "C1", "E3", "Entity C"],
    ]

    pd.DataFrame(transaction_rows, columns=TRANSACTION_COLUMNS).to_csv(transaction_path, index=False)
    pd.DataFrame(account_rows, columns=ACCOUNT_COLUMNS).to_csv(account_path, index=False)
    return transaction_path, account_path


def test_ibm_adapter_normalizes_transaction_and_account_columns(tmp_path: Path) -> None:
    transaction_path, account_path = write_ibm_fixture(tmp_path)
    adapter = IBMDatasetAdapter(transaction_path=transaction_path, account_path=account_path)

    raw_transactions = pd.read_csv(transaction_path)
    normalized = adapter.normalize_transaction_frame(raw_transactions)
    accounts = adapter.load_account_metadata()

    assert list(normalized.columns) == [
        "timestamp",
        "from_bank",
        "sender_account",
        "to_bank",
        "receiver_account",
        "amount_received",
        "receiving_currency",
        "amount_paid",
        "payment_currency",
        "payment_format",
        "is_laundering",
    ]
    assert str(normalized["timestamp"].dtype).startswith("datetime64")
    assert list(accounts.columns) == ["bank_name", "bank_id", "account_number", "entity_id", "entity_name"]
    assert accounts.loc[0, "entity_name"] == "Entity A"


def test_ibm_adapter_filters_chunks_by_date_and_account(tmp_path: Path) -> None:
    transaction_path, account_path = write_ibm_fixture(tmp_path)
    adapter = IBMDatasetAdapter(transaction_path=transaction_path, account_path=account_path)

    chunks = list(
        adapter.iter_transaction_chunks(
            chunksize=2,
            max_rows=2,
            start_date="2022-09-02",
            account_ids=["A1"],
        )
    )

    assert len(chunks) == 2
    assert sum(len(chunk) for chunk in chunks) == 2
    for chunk in chunks:
        assert (chunk["sender_account"] == "A1").any() or (chunk["receiver_account"] == "A1").any()
        assert chunk["timestamp"].min() >= pd.Timestamp("2022-09-02")


def test_ibm_feature_engineer_builds_account_features_and_labels(tmp_path: Path) -> None:
    transaction_path, account_path = write_ibm_fixture(tmp_path)
    engineer = IBMFeatureEngineer(
        transaction_path=transaction_path,
        account_path=account_path,
        config=FeatureEngineeringConfig(include_median=True, low_value_threshold=100, near_threshold_band_ratio=0.10, round_amount_multiple=100),
    )

    feature_set = engineer.build_feature_set(chunksize=2)
    features = feature_set.features.set_index("account_id")
    labels = feature_set.labels.set_index("account_id")

    a1 = features.loc["A1"]
    b1 = features.loc["B1"]

    assert a1["transaction_count"] == 6
    assert a1["outgoing_count"] == 4
    assert a1["incoming_count"] == 3
    assert a1["total_outgoing_amount"] == 495
    assert a1["total_incoming_amount"] == 340
    assert a1["unique_outgoing_counterparties"] == 2
    assert a1["unique_incoming_counterparties"] == 2
    assert a1["unique_counterparties"] == 2
    assert a1["active_days"] == 2
    assert a1["max_transactions_in_hour"] == 2
    assert a1["payment_format_count"] == 4
    assert a1["payment_currency_count"] == 2
    assert a1["receiving_currency_count"] == 2
    assert a1["low_value_outgoing_count"] == 2
    assert a1["near_threshold_outgoing_count"] == 2
    assert a1["near_threshold_outgoing_value"] == 185
    assert a1["round_amount_count"] == 1
    assert a1["self_transfer_count"] == 1
    assert a1["median_outgoing_amount"] == 102.5
    assert a1["median_incoming_amount"] == 90.0
    assert a1["bank_name"] == "Bank #1"
    assert a1["entity_id"] == "E1"
    assert a1["entity_name"] == "Entity A"
    assert a1["rapid_gap_count"] >= 1
    assert a1["mean_time_between_transactions_minutes"] > 0

    assert b1["incoming_count"] >= 1
    assert b1["entity_id"] == "E2"

    assert "is_laundering" not in feature_set.features.columns
    assert not feature_set.labels.empty
    assert {"account_id", "event_count", "laundering_event_count", "has_laundering_label"}.issubset(feature_set.labels.columns)
    assert labels.loc["A1", "laundering_event_count"] > 0


def test_chunked_processing_is_stable(tmp_path: Path) -> None:
    transaction_path, account_path = write_ibm_fixture(tmp_path)
    config = FeatureEngineeringConfig(include_median=False, low_value_threshold=100, near_threshold_band_ratio=0.10, round_amount_multiple=100)

    chunky = IBMFeatureEngineer(transaction_path=transaction_path, account_path=account_path, config=config).build_feature_set(chunksize=1)
    larger = IBMFeatureEngineer(transaction_path=transaction_path, account_path=account_path, config=config).build_feature_set(chunksize=4)

    chunky_features = chunky.features.set_index("account_id")
    larger_features = larger.features.set_index("account_id")

    columns_to_compare = [
        "transaction_count",
        "outgoing_count",
        "incoming_count",
        "total_outgoing_amount",
        "total_incoming_amount",
        "unique_counterparties",
        "self_transfer_count",
        "cross_bank_transaction_count",
        "rapid_gap_count",
    ]
    pd.testing.assert_frame_equal(chunky_features[columns_to_compare], larger_features[columns_to_compare])


def test_feature_set_keeps_labels_separate(tmp_path: Path) -> None:
    transaction_path, account_path = write_ibm_fixture(tmp_path)
    feature_set = IBMFeatureEngineer(transaction_path=transaction_path, account_path=account_path).build_feature_set(max_rows=4, chunksize=2)

    assert "is_laundering" not in feature_set.features.columns
    assert "laundering_event_count" in feature_set.labels.columns
    assert feature_set.metadata["label_policy"].startswith("is_laundering retained")


def test_feature_store_round_trip(tmp_path: Path) -> None:
    transaction_path, account_path = write_ibm_fixture(tmp_path)
    feature_set = IBMFeatureEngineer(transaction_path=transaction_path, account_path=account_path, config=FeatureEngineeringConfig(include_median=True)).build_feature_set(max_rows=4, chunksize=2)

    cache = FeatureStore(tmp_path / "cache")
    cache.save(feature_set.features, feature_set.labels, feature_set.metadata)

    assert cache.exists()

    loaded_features, loaded_labels, loaded_metadata = cache.load()
    pd.testing.assert_frame_equal(loaded_features, feature_set.features)
    pd.testing.assert_frame_equal(loaded_labels, feature_set.labels)
    assert loaded_metadata["transactions_processed"] == feature_set.metadata["transactions_processed"]