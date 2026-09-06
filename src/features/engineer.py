import numpy as np
import pandas as pd

# 3+ products is the single strongest split in this dataset — churn jumps
# from ~8% at 2 products to 83%+ at 3, and hits 100% at 4. That's steep enough
# to look like a labeling artifact of how the source data was assembled, but
# it's what we have, and a Random Forest will find the threshold either way.
# The explicit flag exists so SHAP has a clean, human-readable feature to
# point at instead of forcing the model to reconstruct the cutoff from
# NumOfProducts alone.
OVEREXPOSED_PRODUCT_THRESHOLD = 3


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # +1 epsilon rather than a fixed small constant — salaries in this data
    # range down to double digits, so a fixed epsilon like 1e-6 would still
    # let the ratio blow up. +1 keeps it bounded without distorting anyone
    # with a normal salary.
    #
    # log1p on top of that: a handful of customers with near-zero estimated
    # salary push the raw ratio into the thousands (std of 100 against a
    # median of 0.75), which was overflowing the logistic regression solver
    # during gradient updates. The log compresses that tail without
    # reordering anyone — still monotonic, just no longer capable of
    # dominating every other feature's scale.
    raw_ratio = df["Balance"] / (df["EstimatedSalary"] + 1)
    df["balance_salary_ratio"] = np.log1p(raw_ratio)

    df["product_density"] = df["NumOfProducts"] / (df["Tenure"] + 1)

    df["engagement_product_interaction"] = df["IsActiveMember"] * df["NumOfProducts"]

    df["age_tenure_interaction"] = df["Age"] * df["Tenure"]

    df["is_overexposed"] = (df["NumOfProducts"] >= OVEREXPOSED_PRODUCT_THRESHOLD).astype(int)

    df["is_zero_balance"] = (df["Balance"] == 0).astype(int)

    return df


def split_features_target(df: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y
