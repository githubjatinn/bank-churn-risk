from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless — this runs from a script, never an interactive session

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.base import ClassifierMixin

from src.modeling.explain import compute_partial_dependence


def save_shap_summary_plot(
    shap_values: np.ndarray, X_encoded: np.ndarray, feature_names: list, output_path: Path,
) -> None:
    shap.summary_plot(shap_values, X_encoded, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_importance_bar_plot(
    importance_df: pd.DataFrame, value_col: str, output_path: Path, title: str, top_n: int = 12,
) -> None:
    # Reversed so the highest-importance feature lands at the top of the
    # chart, not the bottom — barh plots bottom-to-top by default.
    top = importance_df.head(top_n).iloc[::-1]
    plt.figure(figsize=(8, 6))
    plt.barh(top["feature"], top[value_col], color="#2563eb")
    plt.xlabel(value_col.replace("_", " ").title())
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_partial_dependence_plot(
    model: ClassifierMixin, X_encoded: np.ndarray, feature_names: list, feature: str, output_path: Path,
) -> None:
    feature_index = feature_names.index(feature)
    grid, mean_predictions = compute_partial_dependence(model, X_encoded, feature_index)

    plt.figure(figsize=(6, 4))
    plt.plot(grid, mean_predictions, color="#2563eb", linewidth=2)
    plt.xlabel(feature)
    plt.ylabel("Predicted churn probability")
    plt.title(f"Partial Dependence — {feature}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
