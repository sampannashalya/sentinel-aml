from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


class FeatureStore:
    def __init__(self, cache_dir: str | Path) -> None:
        self.cache_dir = Path(cache_dir)

    @property
    def features_path(self) -> Path:
        return self.cache_dir / "features.parquet"

    @property
    def labels_path(self) -> Path:
        return self.cache_dir / "labels.parquet"

    @property
    def metadata_path(self) -> Path:
        return self.cache_dir / "metadata.json"

    def exists(self) -> bool:
        return self.features_path.exists() and self.labels_path.exists() and self.metadata_path.exists()

    def save(self, features: pd.DataFrame, labels: pd.DataFrame, metadata: dict[str, Any]) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        features.to_parquet(self.features_path, index=False)
        labels.to_parquet(self.labels_path, index=False)
        self.metadata_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")

    def load(self) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
        metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        features = pd.read_parquet(self.features_path)
        labels = pd.read_parquet(self.labels_path)
        return features, labels, metadata

    def load_features(self) -> pd.DataFrame:
        return pd.read_parquet(self.features_path)
