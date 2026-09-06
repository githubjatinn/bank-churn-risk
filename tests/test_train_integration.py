from src.config import TARGET_COL
from src.data.clean import clean
from src.features.engineer import add_derived_features
from src.modeling import registry
from src.modeling.train import train


def test_train_end_to_end_produces_valid_artifacts(redirect_artifact_paths, sample_raw_df):
    featured = add_derived_features(clean(sample_raw_df))
    metadata = train(featured)

    # metadata has the shape everything downstream (run_pipeline.py's
    # logging, the app's Home page) expects
    assert "feature_names" in metadata
    assert "test_metrics" in metadata
    assert set(metadata["test_metrics"].keys()) == {"logistic_regression", "random_forest"}
    for model_key in ("logistic_regression", "random_forest"):
        m = metadata["test_metrics"][model_key]
        assert 0.0 <= m["roc_auc"] <= 1.0
        assert 0.0 <= m["precision"] <= 1.0
        assert 0.0 <= m["recall"] <= 1.0

    # artifacts were actually written and are loadable
    assert registry.artifacts_exist()
    model = registry.load_model()
    preprocessor = registry.load_preprocessor()

    # and the saved model actually produces valid probabilities on new data
    X = featured.drop(columns=[TARGET_COL])
    X_encoded = preprocessor.transform(X)
    probabilities = model.predict_proba(X_encoded)[:, 1]
    assert ((probabilities >= 0.0) & (probabilities <= 1.0)).all()


def test_train_test_split_is_reproducible(redirect_artifact_paths, sample_raw_df):
    featured = add_derived_features(clean(sample_raw_df))
    metadata_1 = train(featured)
    metadata_2 = train(featured)
    # same random seed, same data in -> identical split sizes and metrics
    assert metadata_1["n_train"] == metadata_2["n_train"]
    assert metadata_1["test_metrics"]["random_forest"]["roc_auc"] == \
        metadata_2["test_metrics"]["random_forest"]["roc_auc"]
