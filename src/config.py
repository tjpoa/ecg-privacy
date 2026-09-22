from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SRC_DIR = PROJECT_ROOT / "src"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
OUTPUTS_TABLES_DIR = OUTPUTS_DIR / "tables"
OUTPUTS_FIGURES_DIR = OUTPUTS_DIR / "figures"
OUTPUTS_MODELS_DIR = OUTPUTS_DIR / "models"
OUTPUTS_LOGS_DIR = OUTPUTS_DIR / "logs"

DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW = DATA_DIR / "raw"
DATA_INTERIM = DATA_DIR / "interim"
DATA_METADATA = DATA_DIR / "metadata"
DATA_PROCESSED = DATA_DIR / "processed"

SEGMENT_FEATURES_DATASET_DIR = DATA_INTERIM / "segment_features_dataset"
FINAL_SEGMENT_FEATURES_DIR = DATA_PROCESSED / "final_segment_features_dataset"
FEATURE_SETS_DIR = DATA_PROCESSED / "feature_sets"

WFDB_RECORDS_DIR = DATA_RAW / "WFDBRecords"
