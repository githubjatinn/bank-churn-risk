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

import streamlit as st

from data_access import load_artifacts, load_scored_customers

st.set_page_config(page_title="Bank Churn Risk Intelligence", page_icon="🏦", layout="wide")

st.title("Bank Churn Risk Intelligence")
st.caption("Predictive churn scoring and risk explainability for retail banking customers")

try:
    artifacts = load_artifacts()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

df = load_scored_customers()
rf_metrics = artifacts["metadata"]["test_metrics"]["random_forest"]

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Customers", f"{len(df):,}")
col2.metric("Overall Churn Rate", f"{df['Exited'].mean():.1%}")
col3.metric("Model ROC-AUC", f"{rf_metrics['roc_auc']:.3f}")
col4.metric("High-Risk Customers", f"{(df['churn_probability'] >= 0.6).sum():,}")

st.divider()

st.markdown(
    """
    Use the pages in the sidebar to explore the model:

    - **Risk Calculator** — score an individual customer and see what's driving their prediction
    - **Probability Distribution** — see how predicted risk is spread across the customer base
    - **Feature Importance** — compare what the Random Forest, SHAP, and Logistic Regression each say matters most
    - **What-If Simulator** — pick a real customer and adjust their attributes to see risk change live
    """
)

st.caption(
    f"Model trained on {artifacts['metadata']['n_train']:,} customers, "
    f"validated on {artifacts['metadata']['n_test']:,} held-out customers. "
    f"Production model: Random Forest (Logistic Regression kept alongside for interpretability)."
)
