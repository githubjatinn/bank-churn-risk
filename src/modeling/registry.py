import json
from datetime import datetime, timezone

import joblib
from sklearn.base import ClassifierMixin
from sklearn.compose import ColumnTransformer

from src import config


def save_artifacts(
    model: ClassifierMixin,
    logistic_model: ClassifierMixin,
    preprocessor: ColumnTransformer,
    metadata: dict,
) -> None:
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, config.MODEL_PATH)
    joblib.dump(logistic_model, config.LOGISTIC_MODEL_PATH)
    joblib.dump(preprocessor, config.PREPROCESSOR_PATH)

    metadata = {**metadata, "saved_at": datetime.now(timezone.utc).isoformat()}
    config.METADATA_PATH.write_text(json.dumps(metadata, indent=2))


def load_model() -> ClassifierMixin:
    return joblib.load(config.MODEL_PATH)


def load_logistic_model() -> ClassifierMixin:
    return joblib.load(config.LOGISTIC_MODEL_PATH)


def load_preprocessor() -> ColumnTransformer:
    return joblib.load(config.PREPROCESSOR_PATH)


def load_metadata() -> dict:
    return json.loads(config.METADATA_PATH.read_text())


def artifacts_exist() -> bool:
    return all(
        p.exists()
        for p in (config.MODEL_PATH, config.LOGISTIC_MODEL_PATH, config.PREPROCESSOR_PATH, config.METADATA_PATH)
    )
