import numpy as np

from src.modeling.evaluate import compute_metrics


def test_compute_metrics_perfect_predictions():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    y_proba = np.array([0.1, 0.2, 0.9, 0.8])
    metrics = compute_metrics(y_true, y_pred, y_proba)
    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["roc_auc"] == 1.0


def test_compute_metrics_confusion_matrix_breakdown():
    # 1 true negative, 1 false positive, 1 false negative, 1 true positive
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 1])
    y_proba = np.array([0.2, 0.6, 0.4, 0.7])
    metrics = compute_metrics(y_true, y_pred, y_proba)
    cm = metrics["confusion_matrix"]
    assert cm == {
        "true_negative": 1, "false_positive": 1,
        "false_negative": 1, "true_positive": 1,
    }
    assert metrics["accuracy"] == 0.5
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5


def test_compute_metrics_handles_no_positive_predictions():
    # model predicts everyone stays — precision would divide by zero
    # without zero_division=0
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 0, 0])
    y_proba = np.array([0.1, 0.2, 0.3, 0.4])
    metrics = compute_metrics(y_true, y_pred, y_proba)
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
