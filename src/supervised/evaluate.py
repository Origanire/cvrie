import time
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)


def evaluate_model(model, X_train, y_train, X_val, y_val):
    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start

    y_pred = model.predict(X_val)

    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_val)[:, 1]
    else:
        y_score = model.decision_function(X_val)

    metrics = {
        "accuracy": accuracy_score(y_val, y_pred),
        "precision": precision_score(y_val, y_pred, zero_division=0),
        "recall": recall_score(y_val, y_pred, zero_division=0),
        "f1": f1_score(y_val, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_val, y_score),
        "train_time_sec": train_time,
        "confusion_matrix": confusion_matrix(y_val, y_pred),
        "classification_report": classification_report(y_val, y_pred, zero_division=0),
        "roc_curve": roc_curve(y_val, y_score),
    }

    return metrics


def results_to_dataframe(results_dict):
    rows = []
    for model_name, metrics in results_dict.items():
        rows.append({
            "model": model_name,
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "roc_auc": metrics["roc_auc"],
            "train_time_sec": metrics["train_time_sec"],
        })
    return pd.DataFrame(rows).sort_values("roc_auc", ascending=False)