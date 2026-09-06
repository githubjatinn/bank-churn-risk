import sys
from pathlib import Path

# Every page in this app runs as an independent script under Streamlit, so
# none of them can rely on a package-relative import reaching src/. Walking
# upward from this file's own location (not the caller's) to find the
# project root works identically regardless of which page imports this
# module or how deep it's nested — safer than trusting Streamlit's own
# sys.path behavior to stay consistent across versions.
_current = Path(__file__).resolve()
for _parent in _current.parents:
    if (_parent / "src").is_dir():
        ROOT_DIR = _parent
        break
else:
    raise RuntimeError("could not locate project root (looked for a 'src' directory upward from this file)")

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
import shap
import streamlit as st

from src import config
from src.data.clean import clean
from src.data.loader import load_raw
from src.features.engineer import add_derived_features
from src.modeling.explain import build_tree_explainer, explain_single_prediction, get_base_value
from src.modeling.registry import artifacts_exist, load_logistic_model, load_metadata, load_model, load_preprocessor

RISK_TIERS = [
    (0.0, 0.3, "Low"),
    (0.3, 0.6, "Medium"),
    (0.6, 1.01, "High"),
]


@st.cache_resource(show_spinner="Loading trained model...")
def load_artifacts() -> dict:
    if not artifacts_exist():
        raise FileNotFoundError(
            "Model artifacts not found in models/ — run `python run_pipeline.py` first."
        )
    return {
        "model": load_model(),
        "logistic_model": load_logistic_model(),
        "preprocessor": load_preprocessor(),
        "metadata": load_metadata(),
    }


@st.cache_resource(show_spinner=False)
def get_shap_explainer(_model) -> shap.TreeExplainer:
    # Leading underscore tells Streamlit's cache not to try hashing the
    # model object — it can't, and doesn't need to, since exactly one
    # trained model is loaded per session anyway.
    return build_tree_explainer(_model)


@st.cache_data(show_spinner="Scoring customer base...")
def load_scored_customers() -> pd.DataFrame:
    """The full customer base, cleaned/engineered/scored — backs the
    probability distribution and feature importance pages, and the
    customer picker in the what-if simulator."""
    artifacts = load_artifacts()
    raw = load_raw(config.RAW_DATA_PATH)

    # clean()'s first step is exactly this same dedup, and it never removes
    # rows afterward — so this stays aligned with the cleaned/engineered
    # frame by position. CustomerId itself never touches the model; it's
    # here purely so the app can let someone pick a specific customer.
    customer_ids = raw.drop_duplicates(subset="CustomerId").reset_index(drop=True)["CustomerId"]

    featured = add_derived_features(clean(raw))
    X = featured.drop(columns=[config.TARGET_COL])
    X_encoded = artifacts["preprocessor"].transform(X)

    result = featured.copy()
    result.insert(0, "CustomerId", customer_ids)
    result["churn_probability"] = artifacts["model"].predict_proba(X_encoded)[:, 1]
    return result


def risk_tier(probability: float) -> str:
    for low, high, label in RISK_TIERS:
        if low <= probability < high:
            return label
    return "High"


def score_customer(inputs: dict) -> dict:
    """Runs a single manually-entered customer through the exact same
    clean -> engineer -> preprocess -> predict path as training data, so
    the app can never silently drift from what the model actually learned."""
    artifacts = load_artifacts()
    row = pd.DataFrame([inputs])
    featured = add_derived_features(clean(row))

    X_encoded = artifacts["preprocessor"].transform(featured)
    probability = float(artifacts["model"].predict_proba(X_encoded)[:, 1][0])

    explainer = get_shap_explainer(artifacts["model"])
    contributions = explain_single_prediction(
        explainer, X_encoded, artifacts["metadata"]["feature_names"], top_n=6,
    )

    return {
        "probability": probability,
        "tier": risk_tier(probability),
        "contributions": contributions,
        "base_value": get_base_value(explainer),
    }
