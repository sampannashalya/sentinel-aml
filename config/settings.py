from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RAW_DATASET_PATH = RAW_DATA_DIR / "transactions.csv"
PROCESSED_DATASET_PATH = PROCESSED_DATA_DIR / "transactions_processed.csv"
IBM_RAW_DIR = RAW_DATA_DIR / "ibm_aml"
IBM_TRANSACTION_PATH = IBM_RAW_DIR / "HI-Small_Trans.csv"
IBM_ACCOUNT_PATH = IBM_RAW_DIR / "HI-Small_accounts.csv"
IBM_PATTERNS_PATH = IBM_RAW_DIR / "HI-Small_Patterns.txt"
IBM_FEATURE_CACHE_DIR = PROCESSED_DATA_DIR / "ibm_feature_cache"

REQUIRED_COLUMNS = {
    "transaction_id",
    "customer_id",
    "amount",
    "timestamp",
    "transaction_type",
    "counterparty",
    "country",
}
