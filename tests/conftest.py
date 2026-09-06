import numpy as np
import pandas as pd
import pytest

from src import config


@pytest.fixture
def sample_raw_df() -> pd.DataFrame:
    """A synthetic but schema-accurate raw dataset — 200 rows, both classes
    well represented, includes a duplicate CustomerId and a null value so
    the cleaning logic actually has something to do. Large enough that
    train() can run its 5-fold CV without a fold coming up empty on the
    minority class.
    """
    rng = np.random.default_rng(seed=7)
    n = 200

    geography = rng.choice(["France", "Germany", "Spain"], size=n, p=[0.5, 0.25, 0.25])
    num_products = rng.choice([1, 2, 3, 4], size=n, p=[0.5, 0.35, 0.1, 0.05])
    is_active = rng.integers(0, 2, size=n)

    # Churn correlates with product count and activity, same direction as
    # the real dataset, so tests exercise realistic model behavior rather
    # than pure noise.
    churn_prob = 0.1 + 0.2 * (num_products >= 3) + 0.15 * (is_active == 0)
    exited = (rng.random(n) < churn_prob).astype(int)

    df = pd.DataFrame({
        "Year": 2025,
        "CustomerId": np.arange(15600000, 15600000 + n),
        "Surname": [f"Customer{i}" for i in range(n)],
        "CreditScore": rng.integers(400, 850, size=n),
        "Geography": geography,
        "Gender": rng.choice(["Male", "Female"], size=n),
        "Age": rng.integers(18, 80, size=n),
        "Tenure": rng.integers(0, 10, size=n),
        "Balance": rng.uniform(0, 200000, size=n).round(2),
        "NumOfProducts": num_products,
        "HasCrCard": rng.integers(0, 2, size=n),
        "IsActiveMember": is_active,
        "EstimatedSalary": rng.uniform(20000, 180000, size=n).round(2),
        "Exited": exited,
    })

    # A real duplicate, and a real null — clean() needs to earn its keep.
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    df.loc[5, "CreditScore"] = np.nan

    return df


@pytest.fixture
def redirect_artifact_paths(tmp_path, monkeypatch):
    """Points every artifact path at a scratch directory so tests never
    touch the real models/ folder — each concrete path is monkeypatched
    individually since they're computed once at import time, not derived
    dynamically from MODEL_DIR."""
    model_dir = tmp_path / "models"
    monkeypatch.setattr(config, "MODEL_DIR", model_dir)
    monkeypatch.setattr(config, "MODEL_PATH", model_dir / "churn_model.pkl")
    monkeypatch.setattr(config, "LOGISTIC_MODEL_PATH", model_dir / "logistic_model.pkl")
    monkeypatch.setattr(config, "PREPROCESSOR_PATH", model_dir / "preprocessor.pkl")
    monkeypatch.setattr(config, "METADATA_PATH", model_dir / "metadata.json")
    return model_dir
