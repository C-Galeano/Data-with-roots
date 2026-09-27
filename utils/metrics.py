"""Evaluation metrics shared by the binary classification models."""

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


def classification_metrics(y_true, y_pred):
    """Confusion matrix plus accuracy, precision, recall and F1 score."""
    return {
        "cm": confusion_matrix(y_true, y_pred),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
