import numpy as np
import pandas as pd
import shap
from sklearn.base import ClassifierMixin
from sklearn.linear_model import LogisticRegression

# TreeExplainer on a binary sklearn classifier hands back one array of SHAP
# values per class stacked on the last axis. We only ever care about "what
# pushed this customer toward churning," so every function downstream works
# off a plain 2D array — this constant is the one place that decision lives.
CHURN_CLASS_INDEX = 1


def build_tree_explainer(model: ClassifierMixin) -> shap.TreeExplainer:
    return shap.TreeExplainer(model)


def compute_shap_values(explainer: shap.TreeExplainer, X_encoded: np.ndarray) -> np.ndarray:
    raw = explainer.shap_values(X_encoded)
    if isinstance(raw, list):
        return raw[CHURN_CLASS_INDEX]
    if raw.ndim == 3:
        return raw[:, :, CHURN_CLASS_INDEX]
    return raw


def get_base_value(explainer: shap.TreeExplainer) -> float:
    base = explainer.expected_value
    if isinstance(base, (list, np.ndarray)):
        return float(base[CHURN_CLASS_INDEX])
    return float(base)


def shap_importance_table(shap_values: np.ndarray, feature_names: list) -> pd.DataFrame:
    mean_abs = np.abs(shap_values).mean(axis=0)
    return (
        pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )


def model_importance_table(model: ClassifierMixin, feature_names: list) -> pd.DataFrame:
    # Included for comparison against the SHAP ranking, not as the primary
    # story — RF's impurity-based importance is known to inflate features
    # with more distinct split points (continuous columns like Balance)
    # regardless of whether they're actually more predictive.
    return (
        pd.DataFrame({"feature": feature_names, "importance": model.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def logistic_odds_ratios(logistic_model: LogisticRegression, feature_names: list) -> pd.DataFrame:
    coefficients = logistic_model.coef_[0]
    return (
        pd.DataFrame({
            "feature": feature_names,
            "coefficient": coefficients,
            "odds_ratio": np.exp(coefficients),
        })
        .sort_values("coefficient", key=np.abs, ascending=False)
        .reset_index(drop=True)
    )


def explain_single_prediction(
    explainer: shap.TreeExplainer, x_encoded_row: np.ndarray, feature_names: list, top_n: int = 5,
) -> pd.DataFrame:
    """Breaks down one customer's prediction — this is what the app's risk
    calculator page calls to show which factors moved the needle."""
    shap_row = compute_shap_values(explainer, x_encoded_row)[0]
    contributions = pd.DataFrame({"feature": feature_names, "contribution": shap_row})
    contributions["direction"] = np.where(contributions["contribution"] > 0, "increases risk", "decreases risk")
    ranked = contributions.reindex(contributions["contribution"].abs().sort_values(ascending=False).index)
    return ranked.head(top_n).reset_index(drop=True)


def compute_partial_dependence(
    model: ClassifierMixin, X_encoded: np.ndarray, feature_index: int, grid_size: int = 20,
) -> tuple:
    """Hand-rolled rather than sklearn's PartialDependenceDisplay — that ran
    into a reshape bug inside sklearn's own internals that's sensitive to
    the exact sklearn version installed, and the calculation itself is
    simple enough not to need the dependency: hold every other feature at
    its observed value, sweep the target feature across a grid spanning its
    actual range, and average the predicted churn probability at each point.
    """
    column = X_encoded[:, feature_index]
    grid = np.linspace(column.min(), column.max(), grid_size)

    mean_predictions = np.empty(grid_size)
    for i, value in enumerate(grid):
        X_swept = X_encoded.copy()
        X_swept[:, feature_index] = value
        mean_predictions[i] = model.predict_proba(X_swept)[:, 1].mean()

    return grid, mean_predictions
