from __future__ import annotations

from pathlib import Path

import pandas as pd

from config.settings import RAW_DATASET_PATH, REQUIRED_COLUMNS


class DatasetLoader:
    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path or RAW_DATASET_PATH)

    def load(self) -> pd.DataFrame:
        df = pd.read_csv(self.path)
        missing = REQUIRED_COLUMNS.difference(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"]).copy()
        df = df.sort_values("timestamp").reset_index(drop=True)
        return df
