from __future__ import annotations

from tools.data_loader import DatasetLoader


def main() -> None:
    loader = DatasetLoader()
    df = loader.load()
    print(f"Loaded {len(df)} transactions for {df['customer_id'].nunique()} customers.")
    print(df.head(3).to_string(index=False))


if __name__ == "__main__":
    main()
