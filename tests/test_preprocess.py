from src.config import TARGET_COL
from src.data.clean import clean
from src.features.engineer import add_derived_features
from src.modeling.preprocess import build_preprocessor, get_output_feature_names


def _featured(sample_raw_df):
    return add_derived_features(clean(sample_raw_df))


def test_preprocessor_output_shape_matches_feature_names(sample_raw_df):
    df = _featured(sample_raw_df)
    X = df.drop(columns=[TARGET_COL])
    preprocessor = build_preprocessor()
    X_encoded = preprocessor.fit_transform(X)
    feature_names = get_output_feature_names(preprocessor)
    assert X_encoded.shape[1] == len(feature_names)


def test_preprocessor_feature_names_have_no_transformer_prefix(sample_raw_df):
    df = _featured(sample_raw_df)
    X = df.drop(columns=[TARGET_COL])
    preprocessor = build_preprocessor()
    preprocessor.fit(X)
    feature_names = get_output_feature_names(preprocessor)
    assert all("__" not in name for name in feature_names)


def test_preprocessor_transform_is_consistent_on_unseen_row(sample_raw_df):
    df = _featured(sample_raw_df)
    X = df.drop(columns=[TARGET_COL])
    preprocessor = build_preprocessor()
    preprocessor.fit(X.iloc[:150])

    # transform on held-out rows shouldn't raise, even with categories/values
    # not seen during fit (drop='first' + handle_unknown='ignore' covers this)
    transformed = preprocessor.transform(X.iloc[150:])
    assert transformed.shape[0] == len(X) - 150


def test_preprocessor_handles_unknown_category_gracefully(sample_raw_df):
    df = _featured(sample_raw_df)
    X = df.drop(columns=[TARGET_COL])
    preprocessor = build_preprocessor()
    preprocessor.fit(X)

    novel_row = X.iloc[[0]].copy()
    novel_row["Geography"] = "Atlantis"  # not in training data
    transformed = preprocessor.transform(novel_row)  # should not raise
    assert transformed.shape[0] == 1
