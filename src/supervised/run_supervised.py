"""Train on Train, select on Validation, evaluate selected model once on Test."""
import argparse
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)

from src.common.paths import DATA, MODEL, REPORT
from src.supervised.dataset import sha256, validate_manifest
from src.supervised.features import build_feature_matrix
from src.supervised.preprocess import PREPROCESS_VERSION
from src.supervised.train import build_models


def positive_scores(model, features):
    return model.predict_proba(features)[:, list(model.classes_).index(1)]


def measure(labels, scores, threshold=0.5):
    prediction = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, prediction, labels=[0, 1]).ravel()
    return {
        "n": len(labels), "accuracy": float(accuracy_score(labels, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, prediction)),
        "sensitivity": float(recall_score(labels, prediction, zero_division=0)),
        "specificity": float(tn / (tn + fp)),
        "precision": float(precision_score(labels, prediction, zero_division=0)),
        "f1": float(f1_score(labels, prediction, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def bootstrap_auc(labels, scores):
    rng = np.random.default_rng(42)
    values = []
    for _ in range(1000):
        idx = rng.integers(0, len(labels), len(labels))
        if len(np.unique(labels[idx])) == 2:
            values.append(roc_auc_score(labels[idx], scores[idx]))
    return np.percentile(values, [2.5, 97.5]).tolist()


def train(data_dir=DATA, model_path=MODEL, report_path=REPORT):
    manifest_path = data_dir / "manifest.csv"
    if not manifest_path.exists():
        raise ValueError("Données absentes. Lancez : python -m src.supervised.dataset")
    frame = pd.read_csv(manifest_path)
    validate_manifest(frame, data_dir)
    sets = {}
    for name in ("Train", "Validation", "Test"):
        records = frame[frame.split == name].to_dict("records")
        for record in records:
            record["path"] = str(data_dir / record["path"])
        print(f"Extraction des caractéristiques : {name} ({len(records)} images)", flush=True)
        sets[name] = build_feature_matrix(records)[:2]
    candidates = build_models()
    validation = {}
    for name, model in candidates.items():
        print(f"Entraînement : {name}", flush=True)
        start = time.perf_counter()
        model.fit(*sets["Train"])
        x, y = sets["Validation"]
        validation[name] = measure(y, positive_scores(model, x))
        validation[name]["train_seconds"] = round(time.perf_counter() - start, 3)
        print(f"  AUC validation : {validation[name]['roc_auc']:.3f}", flush=True)
    best = max(validation, key=lambda name: validation[name]["roc_auc"])
    model = candidates[best]  # No refit on Validation: the measured model is the saved one.
    x_test, y_test = sets["Test"]
    test_scores = positive_scores(model, x_test)
    test = measure(y_test, test_scores)
    test["roc_auc_ci95_bootstrap"] = bootstrap_auc(y_test, test_scores)
    report = {
        "schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": "miniJSRT PNG + Kaggle JSRT metadata", "dataset_size": len(frame),
        "source_size": 247, "excluded_duplicates": ["JPCNN014"],
        "target": "malignant_nodule_vs_benign_or_non_nodule", "clinical_use": False,
        "classes": {"0": "Bénin ou sans nodule dans JSRT", "1": "Nodule malin dans JSRT"},
        "split_counts": frame.split.value_counts().to_dict(),
        "original_states": frame.state.value_counts().to_dict(),
        "seed": 42, "threshold": 0.5, "threshold_policy": "fixed, research only; not clinically validated",
        "preprocess_version": PREPROCESS_VERSION,
        "manifest_sha256": sha256(manifest_path), "selected_model": best,
        "selection_rule": "Maximum validation ROC AUC; test never used for selection",
        "validation": validation, "test": test,
        "test_majority_baseline_accuracy": float(np.mean(y_test == 0)),
        "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__, "numpy": np.__version__},
        "limitations": [
            "246 radiographies historiques uniques uniquement ; aucune validation clinique ou externe.",
            "Le score du modèle n'est pas une probabilité individuelle de cancer.",
            "La classe négative réunit les nodules bénins et les images sans nodule ; elle ne prouve pas l'absence de cancer.",
            "Le modèle ne vérifie pas l'anatomie, la projection ni la provenance des images.",
            "Les images miniJSRT 256 × 256 en 8 bits réduisent la résolution des radiographies originales 12 bits.",
            "Les contrôles de doublons portent sur les identifiants et les fichiers exacts, pas les quasi-doublons.",
        ],
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {"schema_version": 1, "model": model, "report": report,
              "preprocess_version": PREPROCESS_VERSION,
              "dataset_hashes": dict(zip(frame.sha256, frame.split))}
    temporary = model_path.with_suffix(".tmp")
    joblib.dump(bundle, temporary, compress=3)
    temporary.replace(model_path)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    frame.to_csv(report_path.parent / "split_manifest.csv", index=False)
    predictions = frame[frame.split == "Test"][["patient_id", "state", "label"]].copy()
    predictions["score"] = test_scores
    predictions["prediction"] = (test_scores >= 0.5).astype(int)
    predictions.to_csv(report_path.parent / "test_predictions.csv", index=False)
    print(f"Modèle sauvegardé : {best}\nTest : {json.dumps(test, indent=2)}", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description="Entraîner et évaluer le prototype JSRT.")
    parser.add_argument("--data-dir", type=Path, default=DATA)
    args = parser.parse_args()
    try:
        train(args.data_dir)
    except (ValueError, OSError) as exc:
        parser.exit(1, f"Erreur : {exc}\n")


if __name__ == "__main__":
    main()
