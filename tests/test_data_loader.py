from pathlib import Path

import pandas as pd

from tools.data_loader import DatasetLoader


def test_dataset_loader_loads_and_validates(tmp_path: Path) -> None:
    csv_path = tmp_path / "transactions.csv"
    pd.DataFrame(
        [
            {
                "transaction_id": "t1",
                "customer_id": "c1",
                "amount": 100,
                "timestamp": "2024-01-01 00:00:00",
                "transaction_type": "card",
                "counterparty": "m1",
                "country": "US",
            }
        ]
    ).to_csv(csv_path, index=False)

    loader = DatasetLoader(path=csv_path)
    df = loader.load()

    assert not df.empty
    assert "timestamp" in df.columns
    assert df.iloc[0]["customer_id"] == "c1"
