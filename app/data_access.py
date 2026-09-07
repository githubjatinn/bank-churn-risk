import sys
from pathlib import Path

_current = Path(__file__).resolve()
for _parent in _current.parents:
    if (_parent / "src").is_dir():
        ROOT_DIR = _parent
        break
else:
    raise RuntimeError("could not locate project root (looked for a 'src' directory upward from this file)")

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import logging

import pandas as pd
import shap
import streamlit as st

from src import config
from src.data.clean import clean
from src.data.loader import load_raw
from src.features.engineer import add_derived_features
from src.modeling.explain import build_tree_explainer, explain_single_prediction, get_base_value
from src.modeling.registry import artifacts_exist, load_logistic_model, load_metadata, load_model, load_preprocessor

logger = logging.getLogger(__name__)

RISK_TIERS = [
    (0.0, 0.3, "Low"),
    (0.3, 0.6, "Medium"),
    (0.6, 1.01, "High"),
]


@st.cache_resource(show_spinner="Loading trained model...")
def load_artifacts() -> dict:
    try:
        if not artifacts_exist():
            raise FileNotFoundError("no artifacts on disk")
        return {
            "model": load_model(),
            "logistic_model": load_logistic_model(),
            "preprocessor": load_preprocessor(),
            "metadata": load_metadata(),
        }
    except Exception as exc:
        # Committed pickle files were serialized by whatever scikit-learn
        # version trained them locally. A hosting platform that controls
        # its own Python/library versions (Streamlit Cloud ignores
        # runtime.txt and forces its own Python release) can easily end up
        # running a different scikit-learn version, and pickled sklearn
        # objects aren't guaranteed compatible across versions. Rather than
        # chase version-pinning across two environments we don't fully
        # control, retrain from scratch using whatever's actually
        # installed here — slower on first load, but can't drift.
        logger.warning("could not load saved artifacts (%s) — training fresh instead", exc)
        return _train_fresh()


def _train_fresh() -> dict:
    """Trains and saves the model using whatever scikit-learn/numpy/joblib
    versions are actually installed in this process — the one guarantee
    that matters when the deployment platform doesn't let us control its
    own Python version."""
    from src.modeling.train import train as train_model

    raw = load_raw(config.RAW_DATA_PATH)
    featured = add_derived_features(clean(raw))
    train_model(featured)

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
