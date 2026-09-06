import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.config import TARGET_COL
from src.data.clean import clean
from src.features.engineer import add_derived_features
from src.modeling.explain import (
    build_tree_explainer,
    compute_partial_dependence,
    compute_shap_values,
    explain_single_prediction,
    get_base_value,
    logistic_odds_ratios,
    model_importance_table,
    shap_importance_table,
)
from src.modeling.preprocess import build_preprocessor, get_output_feature_names


def _fitted_models(sample_raw_df):
    df = add_derived_features(clean(sample_raw_df))
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    preprocessor = build_preprocessor()
    X_encoded = preprocessor.fit_transform(X)
    feature_names = get_output_feature_names(preprocessor)
    return _fit(X_encoded, y, feature_names)


def _fit(X_encoded, y, feature_names):
    rf = RandomForestClassifier(n_estimators=50, random_state=0).fit(X_encoded, y)
    lr = LogisticRegression(max_iter=200, solver="liblinear").fit(X_encoded, y)
    return rf, lr, X_encoded, feature_names


def test_shap_importance_table_sums_to_reasonable_ranking(sample_raw_df):
    rf, _, X_encoded, feature_names = _fitted_models(sample_raw_df)
    explainer = build_tree_explainer(rf)
    shap_values = compute_shap_values(explainer, X_encoded[:50])

    assert shap_values.shape == (50, len(feature_names))
    table = shap_importance_table(shap_values, feature_names)
    # sorted descending
    assert table["mean_abs_shap"].is_monotonic_decreasing


def test_model_importance_table_sums_to_one(sample_raw_df):
    rf, _, _, feature_names = _fitted_models(sample_raw_df)
    table = model_importance_table(rf, feature_names)
    assert abs(table["importance"].sum() - 1.0) < 1e-6
    assert table["importance"].is_monotonic_decreasing


def test_logistic_odds_ratios_match_exp_of_coefficients(sample_raw_df):
    _, lr, _, feature_names = _fitted_models(sample_raw_df)
    table = logistic_odds_ratios(lr, feature_names)
    for _, row in table.iterrows():
        assert abs(row["odds_ratio"] - np.exp(row["coefficient"])) < 1e-9


def test_explain_single_prediction_returns_top_n(sample_raw_df):
    rf, _, X_encoded, feature_names = _fitted_models(sample_raw_df)
    explainer = build_tree_explainer(rf)
    result = explain_single_prediction(explainer, X_encoded[:1], feature_names, top_n=3)
    assert len(result) == 3
    assert set(result["direction"]).issubset({"increases risk", "decreases risk"})
    # sorted by absolute contribution, descending
    abs_contributions = result["contribution"].abs()
    assert abs_contributions.is_monotonic_decreasing


def test_get_base_value_is_a_probability(sample_raw_df):
    rf, _, _, _ = _fitted_models(sample_raw_df)
    explainer = build_tree_explainer(rf)
    base_value = get_base_value(explainer)
    assert 0.0 <= base_value <= 1.0


def test_partial_dependence_shape_and_range(sample_raw_df):
    rf, _, X_encoded, feature_names = _fitted_models(sample_raw_df)
    age_index = feature_names.index("Age")
    grid, predictions = compute_partial_dependence(rf, X_encoded, age_index, grid_size=10)
    assert len(grid) == 10
    assert len(predictions) == 10
    assert ((predictions >= 0.0) & (predictions <= 1.0)).all()
