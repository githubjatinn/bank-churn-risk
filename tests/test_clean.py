import pandas as pd

from src.config import CONSTANT_COLS, IDENTIFIER_COLS
from src.data.clean import clean


def test_clean_drops_identifier_and_constant_columns(sample_raw_df):
    cleaned = clean(sample_raw_df)
    for col in IDENTIFIER_COLS + CONSTANT_COLS:
        assert col not in cleaned.columns


def test_clean_removes_duplicate_customer_ids(sample_raw_df):
    # fixture deliberately appends one duplicate row
    assert sample_raw_df["CustomerId"].duplicated().sum() == 1
    cleaned = clean(sample_raw_df)
    # CustomerId itself is dropped by clean(), so check row count instead
    assert len(cleaned) == len(sample_raw_df) - 1


def test_clean_fills_missing_credit_score(sample_raw_df):
    assert sample_raw_df["CreditScore"].isna().any()
    cleaned = clean(sample_raw_df)
    assert not cleaned["CreditScore"].isna().any()


def test_clean_handles_single_row_without_customer_id():
    # This is the exact shape the app's risk calculator hands to clean() —
    # no CustomerId, no Surname, no Year, since it's a manually entered
    # customer rather than a row from the raw export.
    row = pd.DataFrame([{
        "CreditScore": 650, "Geography": "France", "Gender": "Male", "Age": 40,
        "Tenure": 5, "Balance": 75000.0, "NumOfProducts": 1, "HasCrCard": 1,
        "IsActiveMember": 1, "EstimatedSalary": 100000.0,
    }])
    cleaned = clean(row)
    assert len(cleaned) == 1


def test_clean_preserves_row_count_with_no_duplicates_or_nulls(sample_raw_df):
    deduped_no_nulls = sample_raw_df.drop_duplicates(subset="CustomerId").copy()
    deduped_no_nulls["CreditScore"] = deduped_no_nulls["CreditScore"].fillna(600)
    cleaned = clean(deduped_no_nulls)
    assert len(cleaned) == len(deduped_no_nulls)
