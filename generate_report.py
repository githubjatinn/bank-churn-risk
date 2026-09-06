import os

os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import warnings

warnings.filterwarnings("ignore", message=".*overflow encountered in matmul.*", category=RuntimeWarning)
warnings.filterwarnings("ignore", message=".*invalid value encountered in matmul.*", category=RuntimeWarning)
warnings.filterwarnings("ignore", message=".*divide by zero encountered in matmul.*", category=RuntimeWarning)

import json
import logging

import pandas as pd
from sklearn.model_selection import train_test_split

from src import config
from src.modeling.explain import (
    build_tree_explainer,
    compute_shap_values,
    logistic_odds_ratios,
    model_importance_table,
    shap_importance_table,
)
from src.modeling.plots import save_importance_bar_plot, save_partial_dependence_plot, save_shap_summary_plot
from src.modeling.registry import load_logistic_model, load_metadata, load_model, load_preprocessor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# SHAP on the full 2000-row test set doesn't tell a meaningfully different
# story than a 500-row sample — the summary plot and importance ranking both
# converge well before that, and TreeExplainer's cost scales with sample
# size, so this keeps report generation under a few seconds instead of
# a minute-plus for no real gain.
SHAP_SAMPLE_SIZE = 500

# Chosen for what they show, not just "top of the importance list" — a mix
# of the strongest signal (product count), a demographic (age), an
# engagement flag, and one of the engineered ratios, so the report
# demonstrates PDP working across different feature types.
PDP_FEATURES = ["is_overexposed", "Age", "IsActiveMember", "balance_salary_ratio"]


def main():
    if not config.PROCESSED_DATA_PATH.exists():
        raise FileNotFoundError(
            f"{config.PROCESSED_DATA_PATH} not found — run run_pipeline.py first"
        )

    df = pd.read_parquet(config.PROCESSED_DATA_PATH)
    X = df.drop(columns=[config.TARGET_COL])
    y = df[config.TARGET_COL]

    # Same random_state and test_size as train.py — this reproduces the
    # identical test split without needing to persist it separately.
    _, X_test, _, _ = train_test_split(
        X, y, test_size=config.TEST_SIZE, stratify=y, random_state=config.RANDOM_SEED,
    )

    model = load_model()
    logistic_model = load_logistic_model()
    preprocessor = load_preprocessor()
    feature_names = load_metadata()["feature_names"]

    X_test_enc = preprocessor.transform(X_test)
    sample_idx = pd.Series(range(len(X_test_enc))).sample(
        n=min(SHAP_SAMPLE_SIZE, len(X_test_enc)), random_state=config.RANDOM_SEED,
    )
    shap_sample = X_test_enc[sample_idx]

    logger.info("computing SHAP values for %d test-set rows", len(shap_sample))
    explainer = build_tree_explainer(model)
    shap_values = compute_shap_values(explainer, shap_sample)

    rf_importance = model_importance_table(model, feature_names)
    shap_importance = shap_importance_table(shap_values, feature_names)
    lr_odds = logistic_odds_ratios(logistic_model, feature_names)

    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    save_shap_summary_plot(shap_values, shap_sample, feature_names, config.FIGURES_DIR / "shap_summary.png")
    save_importance_bar_plot(
        rf_importance, "importance", config.FIGURES_DIR / "rf_feature_importance.png",
        "Random Forest Feature Importance",
    )
    save_importance_bar_plot(
        shap_importance, "mean_abs_shap", config.FIGURES_DIR / "shap_feature_importance.png",
        "Mean |SHAP Value| by Feature",
    )

    for feature in PDP_FEATURES:
        save_partial_dependence_plot(
            model, X_test_enc, feature_names, feature, config.FIGURES_DIR / f"pdp_{feature}.png",
        )

    report = {
        "shap_sample_size": len(shap_sample),
        "random_forest_importance": rf_importance.to_dict(orient="records"),
        "shap_importance": shap_importance.to_dict(orient="records"),
        "logistic_odds_ratios": lr_odds.to_dict(orient="records"),
    }
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = config.REPORTS_DIR / "explainability_summary.json"
    report_path.write_text(json.dumps(report, indent=2))

    logger.info("wrote explainability report to %s", report_path)
    logger.info("wrote %d figures to %s", 3 + len(PDP_FEATURES), config.FIGURES_DIR)
    logger.info("top SHAP driver: %s", shap_importance.iloc[0]["feature"])


if __name__ == "__main__":
    main()
