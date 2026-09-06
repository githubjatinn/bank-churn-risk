import json
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
if str(ROOT_DIR / "app") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "app"))

import pandas as pd
import streamlit as st

from data_access import load_artifacts
from src import config

st.set_page_config(page_title="Feature Importance", page_icon="🔍", layout="wide")
st.title("Feature Importance Dashboard")

try:
    load_artifacts()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

report_path = config.REPORTS_DIR / "explainability_summary.json"
if not report_path.exists():
    st.warning(
        "No explainability report found. Run `python generate_report.py` first to "
        "generate SHAP values, importance rankings, and partial dependence plots."
    )
    st.stop()

report = json.loads(report_path.read_text())

tab_shap, tab_rf, tab_lr, tab_pdp = st.tabs(
    ["SHAP Importance", "Random Forest Importance", "Logistic Regression", "Partial Dependence"]
)

with tab_shap:
    st.caption(
        "Mean absolute SHAP value per feature — averaged over "
        f"{report['shap_sample_size']} held-out customers. This is the primary "
        "ranking; unlike raw impurity importance, it isn't biased toward "
        "continuous or high-cardinality features."
    )
    shap_df = pd.DataFrame(report["shap_importance"])
    st.bar_chart(shap_df.set_index("feature")["mean_abs_shap"])

    summary_path = config.FIGURES_DIR / "shap_summary.png"
    if summary_path.exists():
        st.image(str(summary_path), caption="SHAP summary — direction and magnitude per feature")

with tab_rf:
    st.caption(
        "Random Forest's built-in impurity-based importance. Included for "
        "comparison — this metric tends to inflate continuous features "
        "(more possible split points) relative to SHAP's ranking above."
    )
    rf_df = pd.DataFrame(report["random_forest_importance"])
    st.bar_chart(rf_df.set_index("feature")["importance"])

with tab_lr:
    st.caption(
        "Logistic Regression coefficients, shown as odds ratios. One row needs "
        "a caveat: `is_overexposed` shows an extreme odds ratio because it "
        "near-perfectly separates the two classes (86% of flagged customers "
        "churn vs 18% otherwise) — that's quasi-separation inflating the "
        "coefficient, not a reliable effect size. Treat the SHAP ranking as "
        "the trustworthy story and this table as a directional cross-check."
    )
    lr_df = pd.DataFrame(report["logistic_odds_ratios"])
    st.dataframe(
        lr_df,
        column_config={
            "feature": "Feature",
            "coefficient": st.column_config.NumberColumn("Coefficient", format="%.3f"),
            "odds_ratio": st.column_config.NumberColumn("Odds Ratio", format="%.2f"),
        },
        hide_index=True,
        width="stretch",
    )

with tab_pdp:
    st.caption(
        "Partial dependence — average predicted churn probability as one "
        "feature varies, holding everything else at its observed value."
    )
    pdp_files = sorted(config.FIGURES_DIR.glob("pdp_*.png"))
    if not pdp_files:
        st.info("No partial dependence plots found — run `python generate_report.py`.")
    else:
        cols = st.columns(2)
        for i, path in enumerate(pdp_files):
            with cols[i % 2]:
                st.image(str(path))
