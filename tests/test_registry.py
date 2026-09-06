from sklearn.linear_model import LogisticRegression

from src.modeling import registry
from src.modeling.preprocess import build_preprocessor


def test_save_and_load_round_trip(redirect_artifact_paths, sample_raw_df):
    from src.data.clean import clean
    from src.features.engineer import add_derived_features
    from src.config import TARGET_COL

    df = add_derived_features(clean(sample_raw_df))
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    preprocessor = build_preprocessor()
    X_encoded = preprocessor.fit_transform(X)

    model = LogisticRegression(max_iter=200).fit(X_encoded, y)
    metadata = {"feature_names": list(range(X_encoded.shape[1])), "note": "test artifact"}

    registry.save_artifacts(
        model=model, logistic_model=model, preprocessor=preprocessor, metadata=metadata,
    )

    assert registry.artifacts_exist()
    loaded_model = registry.load_model()
    loaded_preprocessor = registry.load_preprocessor()
    loaded_metadata = registry.load_metadata()

    assert loaded_metadata["note"] == "test artifact"
    assert "saved_at" in loaded_metadata  # timestamp added automatically
    # loaded model produces identical predictions to the original
    original_preds = model.predict(X_encoded)
    loaded_preds = loaded_model.predict(loaded_preprocessor.transform(X))
    assert (original_preds == loaded_preds).all()


def test_artifacts_exist_false_before_saving(redirect_artifact_paths):
    assert registry.artifacts_exist() is False
