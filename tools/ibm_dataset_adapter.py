from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from config.settings import IBM_ACCOUNT_PATH, IBM_TRANSACTION_PATH


class IBMDatasetAdapter:
    TRANSACTION_SOURCE_COLUMNS = [
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

    TRANSACTION_RENAME_MAP = {
        "Timestamp": "timestamp",
        "From Bank": "from_bank",
        "Account": "sender_account",
        "To Bank": "to_bank",
        "Account.1": "receiver_account",
        "Amount Received": "amount_received",
        "Receiving Currency": "receiving_currency",
        "Amount Paid": "amount_paid",
        "Payment Currency": "payment_currency",
        "Payment Format": "payment_format",
        "Is Laundering": "is_laundering",
    }

    ACCOUNT_SOURCE_COLUMNS = ["Bank Name", "Bank ID", "Account Number", "Entity ID", "Entity Name"]

    ACCOUNT_RENAME_MAP = {
        "Bank Name": "bank_name",
        "Bank ID": "bank_id",
        "Account Number": "account_number",
        "Entity ID": "entity_id",
        "Entity Name": "entity_name",
    }

    def __init__(self, transaction_path: str | Path | None = None, account_path: str | Path | None = None) -> None:
        self.transaction_path = self._resolve_transaction_path(transaction_path or IBM_TRANSACTION_PATH)
        self.account_path = self._resolve_account_path(account_path or IBM_ACCOUNT_PATH)

    def normalize_transaction_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        normalized = frame.copy()
        normalized = normalized.rename(columns=self.TRANSACTION_RENAME_MAP)
        normalized = normalized.loc[:, list(self.TRANSACTION_RENAME_MAP.values())]

        string_columns = [
            "from_bank",
            "sender_account",
            "to_bank",
            "receiver_account",
            "receiving_currency",
            "payment_currency",
            "payment_format",
        ]
        for column in string_columns:
            normalized[column] = normalized[column].astype("string").str.strip()

        normalized["timestamp"] = pd.to_datetime(normalized["timestamp"], errors="coerce")
        normalized["amount_received"] = pd.to_numeric(normalized["amount_received"], errors="coerce")
        normalized["amount_paid"] = pd.to_numeric(normalized["amount_paid"], errors="coerce")
        normalized["is_laundering"] = pd.to_numeric(normalized["is_laundering"], errors="coerce").astype("Int64")

        normalized = normalized.dropna(subset=["timestamp", "sender_account", "receiver_account"])
        return normalized.reset_index(drop=True)

    def load_account_metadata(self) -> pd.DataFrame:
        frame = pd.read_csv(self.account_path, dtype=str, keep_default_na=False, usecols=self.ACCOUNT_SOURCE_COLUMNS)
        frame = frame.rename(columns=self.ACCOUNT_RENAME_MAP)
        for column in frame.columns:
            frame[column] = frame[column].astype("string").str.strip()
        return frame.drop_duplicates(subset=["account_number"]).reset_index(drop=True)

    def iter_transaction_chunks(
        self,
        chunksize: int = 100_000,
        max_rows: int | None = None,
        start_date: str | pd.Timestamp | None = None,
        end_date: str | pd.Timestamp | None = None,
        account_ids: Iterable[str] | None = None,
    ):
        if chunksize <= 0:
            raise ValueError("chunksize must be positive")

        remaining = max_rows
        account_filter = {str(account_id) for account_id in account_ids} if account_ids is not None else None
        start_ts = pd.Timestamp(start_date) if start_date is not None else None
        end_ts = pd.Timestamp(end_date) if end_date is not None else None

        reader = pd.read_csv(
            self.transaction_path,
            chunksize=chunksize,
            usecols=self.TRANSACTION_SOURCE_COLUMNS,
            dtype=str,
            keep_default_na=False,
            low_memory=False,
        )
        for raw_chunk in reader:
            chunk = self.normalize_transaction_frame(raw_chunk)
            if start_ts is not None:
                chunk = chunk[chunk["timestamp"] >= start_ts]
            if end_ts is not None:
                chunk = chunk[chunk["timestamp"] <= end_ts]
            if account_filter is not None:
                mask = chunk["sender_account"].isin(account_filter) | chunk["receiver_account"].isin(account_filter)
                chunk = chunk[mask]

            if remaining is not None:
                if remaining <= 0:
                    break
                if len(chunk) > remaining:
                    yield chunk.iloc[:remaining].reset_index(drop=True)
                    break
                remaining -= len(chunk)

            if not chunk.empty:
                yield chunk.reset_index(drop=True)

    def load_transactions(
        self,
        max_rows: int | None = None,
        chunksize: int = 100_000,
        start_date: str | pd.Timestamp | None = None,
        end_date: str | pd.Timestamp | None = None,
        account_ids: Iterable[str] | None = None,
    ) -> pd.DataFrame:
        chunks = list(
            self.iter_transaction_chunks(
                chunksize=chunksize,
                max_rows=max_rows,
                start_date=start_date,
                end_date=end_date,
                account_ids=account_ids,
            )
        )
        if not chunks:
            return pd.DataFrame(columns=list(self.TRANSACTION_RENAME_MAP.values()))
        return pd.concat(chunks, ignore_index=True)

    def load_transaction_sample(self, max_rows: int = 1_000) -> pd.DataFrame:
        return self.load_transactions(max_rows=max_rows, chunksize=min(max_rows, 100_000))

    def _resolve_transaction_path(self, path: str | Path) -> Path:
        resolved = Path(path)
        if resolved.is_dir():
            return resolved / IBM_TRANSACTION_PATH.name
        return resolved

    def _resolve_account_path(self, path: str | Path) -> Path:
        resolved = Path(path)
        if resolved.is_dir():
            return resolved / IBM_ACCOUNT_PATH.name
        return resolved