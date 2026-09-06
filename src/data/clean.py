import logging

import pandas as pd

from src.config import (
    BINARY_COLS,
    CATEGORICAL_COLS,
    CONSTANT_COLS,
    IDENTIFIER_COLS,
    NUMERIC_COLS,
    PLAUSIBLE_RANGES,
)

logger = logging.getLogger(__name__)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    before = len(df)
    if "CustomerId" in df.columns:
        df = df.drop_duplicates(subset="CustomerId")
    if len(df) != before:
        logger.warning("dropped %d duplicate CustomerId rows", before - len(df))

    df = df.drop(columns=IDENTIFIER_COLS + CONSTANT_COLS, errors="ignore")

    df = _handle_missing(df)
    _flag_implausible_values(df)

    return df.reset_index(drop=True)


def _handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    # This export happens to be complete, but the pipeline shouldn't assume
    # that holds forever. Numeric gaps get median-filled per column rather
    # than dropped — churn is a minority class already, and dropping rows
    # over one missing field would bias the training set for no good reason.
    # Categoricals get an explicit "Unknown" bucket instead of the mode, so
    # the model can learn "we don't know" is itself informative rather than
    # silently pretending every unknown customer is from the majority country.
    null_counts = df.isna().sum()
    if null_counts.sum() == 0:
        return df

    for col in null_counts[null_counts > 0].index:
        logger.info("filling %d nulls in %s", null_counts[col], col)

    for col in NUMERIC_COLS:
        if col in df.columns and df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    for col in CATEGORICAL_COLS:
        if col in df.columns and df[col].isna().any():
            df[col] = df[col].fillna("Unknown")

    for col in BINARY_COLS:
        if col in df.columns and df[col].isna().any():
            df[col] = df[col].fillna(0).astype(int)

    return df


def _flag_implausible_values(df: pd.DataFrame) -> None:
    for col, (low, high) in PLAUSIBLE_RANGES.items():
        if col not in df.columns:
            continue
        out_of_range = ~df[col].between(low, high)
        if out_of_range.any():
            logger.warning(
                "%d rows have %s outside plausible range [%s, %s]",
                out_of_range.sum(), col, low, high,
            )

    if (df["Balance"] < 0).any():
        logger.warning("%d rows have negative Balance", (df["Balance"] < 0).sum())

    if (df["EstimatedSalary"] <= 0).any():
        logger.warning(
            "%d rows have non-positive EstimatedSalary", (df["EstimatedSalary"] <= 0).sum()
        )
