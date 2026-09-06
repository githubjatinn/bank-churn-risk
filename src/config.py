"""
Project-wide constants. Everything here is deliberately static — no environment
variable overrides, no runtime config file. If this project needs multi-environment
config later (dev/staging/prod data paths), that's the time to introduce something
heavier. For now a single Python module is easier to read and diff than YAML.
"""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

RAW_DATA_PATH = ROOT_DIR / "data" / "raw" / "European_Bank.csv"
PROCESSED_DATA_PATH = ROOT_DIR / "data" / "processed" / "features.parquet"

MODEL_DIR = ROOT_DIR / "models"
MODEL_PATH = MODEL_DIR / "churn_model.pkl"
LOGISTIC_MODEL_PATH = MODEL_DIR / "logistic_model.pkl"
PREPROCESSOR_PATH = MODEL_DIR / "preprocessor.pkl"
METADATA_PATH = MODEL_DIR / "metadata.json"

REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

RANDOM_SEED = 42
TARGET_COL = "Exited"

# Columns present in the raw export that carry no predictive signal.
# CustomerId/Surname are identifiers. Year is constant (2025 in every row
# as of this export) — a zero-variance column is worse than useless for a
# tree model, it just adds a split candidate that never helps.
IDENTIFIER_COLS = ["CustomerId", "Surname"]
CONSTANT_COLS = ["Year"]

CATEGORICAL_COLS = ["Geography", "Gender"]
NUMERIC_COLS = [
    "CreditScore",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "EstimatedSalary",
]
BINARY_COLS = ["HasCrCard", "IsActiveMember"]

# Continuous features built in features/engineer.py — get scaled same as the
# raw numeric columns. Binary ones get passed through untouched since a 0/1
# flag scaled to mean-0/unit-variance doesn't mean anything a human can read.
ENGINEERED_NUMERIC_COLS = [
    "balance_salary_ratio",
    "product_density",
    "engagement_product_interaction",
    "age_tenure_interaction",
]
ENGINEERED_BINARY_COLS = ["is_overexposed", "is_zero_balance"]
PASSTHROUGH_COLS = BINARY_COLS + ENGINEERED_BINARY_COLS

# Plausible ranges used for data quality warnings, not hard validation failures.
# A row outside these bounds gets logged, not dropped — bad-but-real data is
# common enough in banking exports that failing the whole pipeline on it is
# the wrong default.
PLAUSIBLE_RANGES = {
    "CreditScore": (300, 850),
    "Age": (18, 100),
    "Tenure": (0, 40),
}

TEST_SIZE = 0.2
CV_FOLDS = 5
