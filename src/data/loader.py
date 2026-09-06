import logging
from pathlib import Path

import pandas as pd

from src.config import TARGET_COL

logger = logging.getLogger(__name__)

EXPECTED_COLUMNS = {
    "Year", "CustomerId", "Surname", "CreditScore", "Geography", "Gender",
    "Age", "Tenure", "Balance", "NumOfProducts", "HasCrCard",
    "IsActiveMember", "EstimatedSalary", "Exited",
}


class SchemaValidationError(ValueError):
    """Raised when the raw export doesn't match the columns the rest of the
    pipeline was built against. Deliberately a distinct type from a generic
    ValueError so callers (and tests) can catch it specifically."""


def load_raw(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"raw data file not found: {path}")

    df = pd.read_csv(path)
    _validate_schema(df)

    logger.info("loaded %d rows from %s", len(df), path.name)
    return df


def _validate_schema(df: pd.DataFrame) -> None:
    missing = EXPECTED_COLUMNS - set(df.columns)
    if missing:
        raise SchemaValidationError(
            f"raw export is missing expected columns: {sorted(missing)}"
        )

    extra = set(df.columns) - EXPECTED_COLUMNS
    if extra:
        # Not fatal — a new column from the source system shouldn't break the
        # pipeline, but it's the kind of thing that should show up in a log
        # rather than silently getting engineered on later.
        logger.warning("raw export has unexpected columns, ignoring: %s", sorted(extra))

    if df[TARGET_COL].isna().any():
        raise SchemaValidationError(
            f"{df[TARGET_COL].isna().sum()} rows have a null target — cannot train on these"
        )

    if not set(df[TARGET_COL].unique()).issubset({0, 1}):
        raise SchemaValidationError(
            f"target column has non-binary values: {sorted(df[TARGET_COL].unique())}"
        )
