import numpy as np

from src.modeling.transformers import PercentileClipper


def test_clipper_leaves_normal_values_untouched():
    X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
    clipper = PercentileClipper(lower_percentile=0.0, upper_percentile=100.0)
    result = clipper.fit_transform(X)
    np.testing.assert_array_almost_equal(result.ravel(), X.ravel())


def test_clipper_caps_extreme_outlier():
    X = np.array([[1.0], [2.0], [3.0], [4.0], [1000.0]])
    clipper = PercentileClipper(lower_percentile=1.0, upper_percentile=90.0)
    clipper.fit(X)
    result = clipper.transform(X)
    assert result.max() < 1000.0
    assert result.max() == clipper.upper_bounds_[0]


def test_clipper_fits_bounds_from_train_and_applies_to_new_data():
    X_train = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
    clipper = PercentileClipper(lower_percentile=0.0, upper_percentile=100.0)
    clipper.fit(X_train)

    X_new = np.array([[999.0], [-999.0]])
    result = clipper.transform(X_new)
    # bounds came from training data (max 5, min 1), not from X_new
    assert result[0, 0] == 5.0
    assert result[1, 0] == 1.0


def test_clipper_get_feature_names_out_passes_through():
    clipper = PercentileClipper()
    names = clipper.get_feature_names_out(["a", "b", "c"])
    assert list(names) == ["a", "b", "c"]
