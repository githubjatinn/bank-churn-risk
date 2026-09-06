from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import CATEGORICAL_COLS, ENGINEERED_NUMERIC_COLS, NUMERIC_COLS, PASSTHROUGH_COLS
from src.modeling.transformers import PercentileClipper


def build_preprocessor() -> ColumnTransformer:
    # drop='first' on the categoricals — Geography/Gender each collapse to
    # one reference category, which keeps the logistic regression
    # coefficients interpretable (each dummy reads as "vs. the baseline")
    # instead of carrying a redundant column the model has to null out.
    # Costs nothing for the Random Forest, which doesn't care about
    # collinearity, so one preprocessor comfortably serves both models.
    #
    # The engineered numeric columns get an extra clipping step ahead of
    # scaling — see transformers.PercentileClipper for why. Raw numeric
    # columns (CreditScore, Age, Balance, ...) skip it; their skew is real
    # demographic signal, not the kind of outlier that destabilizes a solver.
    engineered_pipeline = Pipeline([
        ("clip", PercentileClipper()),
        ("scale", StandardScaler()),
    ])

    return ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(drop="first", handle_unknown="ignore"), CATEGORICAL_COLS),
            ("numeric", StandardScaler(), NUMERIC_COLS),
            ("engineered_numeric", engineered_pipeline, ENGINEERED_NUMERIC_COLS),
            ("passthrough", "passthrough", PASSTHROUGH_COLS),
        ],
        remainder="drop",
    )


def get_output_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    # get_feature_names_out prefixes each block with its transformer name
    # (e.g. "categorical__Geography_Germany") — stripping that back to the
    # underlying column name is what makes the SHAP plots in phase 3 and the
    # feature-importance table in the app readable instead of full of
    # sklearn internals.
    raw_names = preprocessor.get_feature_names_out()
    return [name.split("__", maxsplit=1)[-1] for name in raw_names]
