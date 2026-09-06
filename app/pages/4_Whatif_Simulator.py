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

from data_access import load_artifacts, load_scored_customers, score_customer

st.set_page_config(page_title="What-If Simulator", page_icon="🎛️", layout="wide")
st.title("What-If Scenario Simulator")
st.caption("Pick a real customer, then adjust their engagement and product mix to see how risk responds")

try:
    load_artifacts()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

df = load_scored_customers()

customer_id = st.selectbox("Select a customer", df["CustomerId"].tolist())
customer = df.loc[df["CustomerId"] == customer_id].iloc[0]

baseline_inputs = {
    "CreditScore": int(customer["CreditScore"]),
    "Geography": customer["Geography"],
    "Gender": customer["Gender"],
    "Age": int(customer["Age"]),
    "Tenure": int(customer["Tenure"]),
    "Balance": float(customer["Balance"]),
    "NumOfProducts": int(customer["NumOfProducts"]),
    "HasCrCard": int(customer["HasCrCard"]),
    "IsActiveMember": int(customer["IsActiveMember"]),
    "EstimatedSalary": float(customer["EstimatedSalary"]),
}
baseline_result = score_customer(baseline_inputs)

st.divider()

st.subheader("Adjust this customer's scenario")
col1, col2 = st.columns(2)

with col1:
    num_products = st.selectbox(
        "Number of Products", [1, 2, 3, 4],
        index=[1, 2, 3, 4].index(baseline_inputs["NumOfProducts"]),
    )
    is_active_member = st.checkbox("Is Active Member", value=bool(baseline_inputs["IsActiveMember"]))

with col2:
    balance = st.number_input("Account Balance ($)", min_value=0.0, value=baseline_inputs["Balance"], step=1000.0)
    tenure = st.slider("Tenure (years with bank)", 0, 10, baseline_inputs["Tenure"])

scenario_inputs = {
    **baseline_inputs,
    "NumOfProducts": num_products,
    "IsActiveMember": int(is_active_member),
    "Balance": balance,
    "Tenure": tenure,
}
scenario_result = score_customer(scenario_inputs)

st.divider()

col1, col2, col3 = st.columns(3)
col1.metric("Baseline Risk", f"{baseline_result['probability']:.1%}")
col2.metric(
    "Scenario Risk",
    f"{scenario_result['probability']:.1%}",
    delta=f"{scenario_result['probability'] - baseline_result['probability']:+.1%}",
    delta_color="inverse",
)
col3.metric("Scenario Risk Tier", scenario_result["tier"])

st.caption(
    "Delta is colored inverse — a drop in churn probability (good outcome) shows green, "
    "an increase shows red."
)

if num_products == 4:
    st.info(
        "Heads up: every customer with 4 products in the training data churned, "
        "regardless of anything else about them — there's no real variation for the "
        "model to have learned from at this exact value, so small probability "
        "differences here reflect noise rather than a genuine relationship."
    )

st.subheader("What's driving the scenario prediction")
contributions = scenario_result["contributions"].copy()
contributions["contribution"] = contributions["contribution"].round(4)
st.dataframe(
    contributions,
    column_config={"feature": "Feature", "contribution": "SHAP Value", "direction": "Effect"},
    hide_index=True,
    width="stretch",
)
