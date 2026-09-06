import pandas as pd

from src.features.engineer import OVEREXPOSED_PRODUCT_THRESHOLD, add_derived_features


def test_balance_salary_ratio_is_log_transformed():
    df = pd.DataFrame([{"Balance": 100000.0, "EstimatedSalary": 50000.0, "Tenure": 5,
                         "NumOfProducts": 1, "IsActiveMember": 1, "Age": 30}])
    result = add_derived_features(df)
    # raw ratio would be ~2.0; log1p(2.0) ~ 1.0986
    assert abs(result["balance_salary_ratio"].iloc[0] - 1.0986) < 0.001


def test_balance_salary_ratio_handles_zero_salary_without_error():
    df = pd.DataFrame([{"Balance": 5000.0, "EstimatedSalary": 0.0, "Tenure": 1,
                         "NumOfProducts": 1, "IsActiveMember": 0, "Age": 25}])
    result = add_derived_features(df)
    assert pd.notna(result["balance_salary_ratio"].iloc[0])


def test_product_density_formula():
    df = pd.DataFrame([{"Balance": 0.0, "EstimatedSalary": 1.0, "Tenure": 3,
                         "NumOfProducts": 2, "IsActiveMember": 1, "Age": 40}])
    result = add_derived_features(df)
    assert result["product_density"].iloc[0] == 2 / (3 + 1)


def test_engagement_product_interaction_is_zero_when_inactive():
    df = pd.DataFrame([{"Balance": 0.0, "EstimatedSalary": 1.0, "Tenure": 1,
                         "NumOfProducts": 4, "IsActiveMember": 0, "Age": 40}])
    result = add_derived_features(df)
    assert result["engagement_product_interaction"].iloc[0] == 0


def test_is_overexposed_flag_at_threshold_boundary():
    df = pd.DataFrame([
        {"Balance": 0.0, "EstimatedSalary": 1.0, "Tenure": 1, "NumOfProducts": OVEREXPOSED_PRODUCT_THRESHOLD - 1, "IsActiveMember": 1, "Age": 30},
        {"Balance": 0.0, "EstimatedSalary": 1.0, "Tenure": 1, "NumOfProducts": OVEREXPOSED_PRODUCT_THRESHOLD, "IsActiveMember": 1, "Age": 30},
    ])
    result = add_derived_features(df)
    assert result["is_overexposed"].tolist() == [0, 1]


def test_is_zero_balance_flag():
    df = pd.DataFrame([
        {"Balance": 0.0, "EstimatedSalary": 1.0, "Tenure": 1, "NumOfProducts": 1, "IsActiveMember": 1, "Age": 30},
        {"Balance": 0.01, "EstimatedSalary": 1.0, "Tenure": 1, "NumOfProducts": 1, "IsActiveMember": 1, "Age": 30},
    ])
    result = add_derived_features(df)
    assert result["is_zero_balance"].tolist() == [1, 0]


def test_add_derived_features_does_not_mutate_input():
    df = pd.DataFrame([{"Balance": 100.0, "EstimatedSalary": 50000.0, "Tenure": 5,
                         "NumOfProducts": 1, "IsActiveMember": 1, "Age": 30}])
    original_columns = list(df.columns)
    add_derived_features(df)
    assert list(df.columns) == original_columns  # unchanged — function copies internally


def test_add_derived_features_works_on_single_row():
    df = pd.DataFrame([{"Balance": 100.0, "EstimatedSalary": 50000.0, "Tenure": 5,
                         "NumOfProducts": 3, "IsActiveMember": 0, "Age": 45}])
    result = add_derived_features(df)
    assert len(result) == 1
    assert result["is_overexposed"].iloc[0] == 1
