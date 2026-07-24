from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RAW_DATASET_PATH = RAW_DATA_DIR / "transactions.csv"
PROCESSED_DATASET_PATH = PROCESSED_DATA_DIR / "transactions_processed.csv"

REQUIRED_COLUMNS = {
    "transaction_id",
    "customer_id",
    "amount",
    "timestamp",
    "transaction_type",
    "counterparty",
    "country",
}
