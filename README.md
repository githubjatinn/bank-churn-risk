# Bank Customer Churn — Risk Intelligence Platform

Predictive churn scoring for retail banking customers, built for the Unified Mentor internship. Given a customer's profile, the model estimates their probability of churning and explains *why* — which factors are pushing that specific prediction up or down — rather than just returning a number.

## Results

| Model | Test ROC-AUC | Recall | Precision |
|---|---|---|---|
| Logistic Regression (baseline) | 0.836 | — | — |
| **Random Forest (production)** | **0.865** | **0.67** | **0.57** |

Recall is prioritized over precision by design (`class_weight="balanced"`) — in a retention-campaign setting, missing an actual churner costs more than one unnecessary outreach to a customer who was going to stay anyway.

**Top churn drivers** (SHAP): `Age`, `NumOfProducts`, engagement/product interaction, `Geography` (Germany), account activity. Full breakdown in `reports/explainability_summary.json` and `reports/figures/`.

## Project structure

```
bank-churn-risk/
├── data/
│   └── raw/European_Bank.csv       # source data
├── src/
│   ├── config.py                    # paths, schema, hyperparameters — single source of truth
│   ├── data/                        # load + validate + clean
│   ├── features/                    # derived features (stateless, safe for single-row inference)
│   └── modeling/                    # preprocessing, training, evaluation, SHAP explainability
├── app/                              # Streamlit dashboard (4 pages)
│   ├── data_access.py               # shared cached loading/scoring layer
│   └── pages/
├── tests/                            # pytest suite, 41 tests
├── reports/                           # SHAP plots, importance rankings (generated)
├── run_pipeline.py                   # train the model end to end
└── generate_report.py                # generate explainability report + figures
```

## Setup

```bash
pip install -r requirements.txt
python run_pipeline.py        # trains and saves the model to models/
python generate_report.py     # generates SHAP report + figures to reports/
python -m pytest -v           # 41 tests, ~8s
streamlit run app/Home.py     # launches the dashboard
```

## Architecture notes

A few decisions worth knowing about if you're reading the code:

- **Feature engineering vs. preprocessing are deliberately separate.** `src/features/engineer.py` only does stateless, row-wise math (ratios, interaction terms) — safe to run identically on a training batch or a single customer at inference. Anything that needs *fitting* (one-hot encoding, scaling, outlier clipping) lives in `src/modeling/preprocess.py`'s `ColumnTransformer`, fit on training data only, to avoid leakage.

- **`balance_salary_ratio` is log-transformed and winsorized.** A handful of customers with near-zero estimated salary produced extreme ratio values that destabilized the Logistic Regression solver during training (and, on some platforms, triggered numerical overflow warnings from the underlying BLAS library). The fix is in `src/features/engineer.py` (log transform) and `src/modeling/transformers.py` (percentile clipping) — both documented inline with the reasoning.

- **SHAP is the primary explainability signal, not Random Forest's built-in feature importance.** RF's impurity-based importance is known to inflate continuous/high-cardinality features regardless of real predictive value. SHAP doesn't have that bias, so it drives the actual "why" story; RF importance is kept only as a cross-check.

- **One Logistic Regression coefficient needs a caveat, not a citation.** `is_overexposed` (3+ products) shows an extreme odds ratio in the LR table — that's quasi-separation (the feature almost perfectly separates the two classes: 86% churn vs. 18%), not a real effect size. Flagged directly in the Feature Importance page and the report.

- **The Streamlit app never retrains.** It loads the artifacts `run_pipeline.py` already produced and reuses the exact same `clean → engineer → preprocess → predict` path for both the training set and a single manually-entered customer — same code, so the app can't silently drift from what the model actually learned.

## Dataset

10,000 retail banking customers — credit score, geography, age, tenure, balance, product count, activity status, estimated salary, and churn outcome (`Exited`). ~20.4% churn rate.

## Tech stack

pandas, scikit-learn, SHAP, Streamlit, matplotlib, pytest.
