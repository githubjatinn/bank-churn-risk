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

import streamlit as st

from data_access import load_artifacts, score_customer

st.set_page_config(page_title="Risk Calculator", page_icon="🧮", layout="wide")
st.title("Customer Risk Calculator")
st.caption("Enter a customer's profile to get their predicted churn probability")

try:
    load_artifacts()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

with st.form("risk_calculator_form"):
    col1, col2 = st.columns(2)

    with col1:
        credit_score = st.slider("Credit Score", 300, 850, 650)
        geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.slider("Age", 18, 92, 40)
        tenure = st.slider("Tenure (years with bank)", 0, 10, 5)

    with col2:
        balance = st.number_input("Account Balance ($)", min_value=0.0, value=75000.0, step=1000.0)
        num_products = st.selectbox("Number of Products", [1, 2, 3, 4])
        estimated_salary = st.number_input("Estimated Salary ($)", min_value=0.0, value=100000.0, step=1000.0)
        has_cr_card = st.checkbox("Has Credit Card", value=True)
        is_active_member = st.checkbox("Is Active Member", value=True)

    submitted = st.form_submit_button("Calculate Risk", type="primary")

if submitted:
    result = score_customer({
        "CreditScore": credit_score,
        "Geography": geography,
        "Gender": gender,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": num_products,
        "HasCrCard": int(has_cr_card),
        "IsActiveMember": int(is_active_member),
        "EstimatedSalary": estimated_salary,
    })

    st.divider()

    tier_colors = {"Low": "green", "Medium": "orange", "High": "red"}
    color = tier_colors[result["tier"]]

    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric("Churn Probability", f"{result['probability']:.1%}")
        st.markdown(f"Risk Tier: **:{color}[{result['tier']}]**")
        st.progress(result["probability"])

    with col2:
        st.subheader("What's driving this prediction")
        st.caption(
            f"Average predicted risk across all customers is "
            f"{result['base_value']:.1%} — the SHAP values below show how this "
            f"customer's specific attributes moved their prediction away from that baseline."
        )
        contributions = result["contributions"].copy()
        contributions["contribution"] = contributions["contribution"].round(4)
        st.dataframe(
            contributions,
            column_config={
                "feature": "Feature",
                "contribution": "SHAP Value",
                "direction": "Effect",
            },
            hide_index=True,
            width="stretch",
        )
        st.caption(
            "SHAP values show how much each factor pushed this specific prediction "
            "up or down relative to the average customer — not a general ranking, "
            "just what mattered for this one profile."
        )
