import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class PercentileClipper(BaseEstimator, TransformerMixin):
    """Winsorizes each column to a range fitted on training data.

    A few of the engineered features (ratios, interaction terms) still carry
    double-digit z-scores even after a log transform — not data errors, just
    genuinely unusual customers (zero tenure, very high balance-to-salary).
    That's fine for a Random Forest, which splits on rank order and doesn't
    care how far out a value sits. It's not fine for a gradient-based solver:
    on some BLAS backends (seen this on Apple's Accelerate framework, not
    just in theory) a handful of extreme rows are enough to overflow the
    weight updates in the first few iterations before regularization pulls
    them back down. Clipping the top/bottom 1% costs us almost nothing and
    removes the instability outright, independent of which machine trains it.
    """

    def __init__(self, lower_percentile: float = 1.0, upper_percentile: float = 99.0):
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.lower_bounds_ = np.percentile(X, self.lower_percentile, axis=0)
        self.upper_bounds_ = np.percentile(X, self.upper_percentile, axis=0)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return np.clip(X, self.lower_bounds_, self.upper_bounds_)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)
