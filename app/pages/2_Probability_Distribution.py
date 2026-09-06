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

import matplotlib.pyplot as plt
import streamlit as st

from data_access import load_artifacts, load_scored_customers

st.set_page_config(page_title="Probability Distribution", page_icon="📊", layout="wide")
st.title("Churn Probability Distribution")

try:
    load_artifacts()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

df = load_scored_customers()

threshold = st.slider(
    "Risk threshold — customers above this are flagged for retention outreach",
    0.0, 1.0, 0.5, 0.05,
)
flagged = (df["churn_probability"] >= threshold).sum()

col1, col2, col3 = st.columns(3)
col1.metric("Customers Above Threshold", f"{flagged:,}")
col2.metric("Share of Base", f"{flagged / len(df):.1%}")
col3.metric("Mean Predicted Probability", f"{df['churn_probability'].mean():.1%}")

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Overall Distribution")
    fig, ax = plt.subplots()
    ax.hist(df["churn_probability"], bins=30, color="#2563eb", edgecolor="white")
    ax.axvline(threshold, color="#dc2626", linestyle="--", label=f"Threshold ({threshold:.2f})")
    ax.set_xlabel("Predicted Churn Probability")
    ax.set_ylabel("Number of Customers")
    ax.legend()
    st.pyplot(fig)

with col2:
    st.subheader("Separation by Actual Outcome")
    fig, ax = plt.subplots()
    stayed = df.loc[df["Exited"] == 0, "churn_probability"]
    churned = df.loc[df["Exited"] == 1, "churn_probability"]
    ax.hist(stayed, bins=30, alpha=0.6, label="Actually stayed", color="#2563eb", density=True)
    ax.hist(churned, bins=30, alpha=0.6, label="Actually churned", color="#dc2626", density=True)
    ax.set_xlabel("Predicted Churn Probability")
    ax.set_ylabel("Density")
    ax.legend()
    st.pyplot(fig)
    st.caption(
        "A model with real separating power pushes these two distributions apart — "
        "customers who actually churned should cluster toward higher predicted "
        "probability than customers who stayed."
    )
