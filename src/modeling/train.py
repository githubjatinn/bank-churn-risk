import logging

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from src.config import RANDOM_SEED, TARGET_COL, TEST_SIZE
from src.modeling.evaluate import compute_metrics, cross_validated_auc
from src.modeling.preprocess import build_preprocessor, get_output_feature_names
from src.modeling.registry import save_artifacts

logger = logging.getLogger(__name__)


def train(df) -> dict:
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_SEED,
    )

    preprocessor = build_preprocessor()
    X_train_enc = preprocessor.fit_transform(X_train)
    X_test_enc = preprocessor.transform(X_test)
    feature_names = get_output_feature_names(preprocessor)

    # class_weight="balanced" on both models — churn is the minority class
    # at ~20%, and a model optimizing raw accuracy will happily predict
    # "stays" for everyone and still look decent. The business cost here is
    # asymmetric (missing a churner costs more than one unnecessary retention
    # offer), so recall matters more than the default threshold would give us.
    # solver="liblinear" rather than the default lbfgs — lbfgs computes
    # gradients via a full-batch matrix multiply, and that code path is where
    # the overflow warnings were coming from on Apple Silicon (NumPy links
    # against Apple's Accelerate BLAS there instead of OpenBLAS, and
    # Accelerate has known numerical stability issues with exactly this kind
    # of batched matmul — unrelated to how well-conditioned the actual data
    # is). liblinear uses coordinate descent instead, updating one feature at
    # a time, which sidesteps that path entirely. It's also just a
    # perfectly reasonable solver choice at this dataset size regardless.
    logistic_model = LogisticRegression(
        class_weight="balanced", max_iter=1000, random_state=RANDOM_SEED, solver="liblinear",
    )
    rf_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )

    logger.info("running %d-fold CV for logistic regression baseline", 5)
    lr_cv = cross_validated_auc(logistic_model, X_train_enc, y_train)
    logger.info("logistic CV ROC-AUC: %.4f (+/- %.4f)", lr_cv["mean"], lr_cv["std"])

    logger.info("running %d-fold CV for random forest", 5)
    rf_cv = cross_validated_auc(rf_model, X_train_enc, y_train)
    logger.info("random forest CV ROC-AUC: %.4f (+/- %.4f)", rf_cv["mean"], rf_cv["std"])

    logistic_model.fit(X_train_enc, y_train)
    rf_model.fit(X_train_enc, y_train)

    lr_test_metrics = compute_metrics(
        y_test.values,
        logistic_model.predict(X_test_enc),
        logistic_model.predict_proba(X_test_enc)[:, 1],
    )
    rf_test_metrics = compute_metrics(
        y_test.values,
        rf_model.predict(X_test_enc),
        rf_model.predict_proba(X_test_enc)[:, 1],
    )

    logger.info(
        "test set — logistic ROC-AUC: %.4f, random forest ROC-AUC: %.4f",
        lr_test_metrics["roc_auc"], rf_test_metrics["roc_auc"],
    )

    metadata = {
        "feature_names": feature_names,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "churn_rate_train": float(y_train.mean()),
        "production_model": "random_forest",
        "cross_validation": {"logistic_regression": lr_cv, "random_forest": rf_cv},
        "test_metrics": {"logistic_regression": lr_test_metrics, "random_forest": rf_test_metrics},
    }

    save_artifacts(
        model=rf_model,
        logistic_model=logistic_model,
        preprocessor=preprocessor,
        metadata=metadata,
    )

    return metadata
