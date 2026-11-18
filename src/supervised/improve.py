"""Select a transfer model by CV on Train, tune its threshold on Validation only."""
import hashlib
import json
import platform
import importlib.metadata
import time
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from src.common.paths import DATA, MODEL, REPORT
from src.supervised.dataset import sha256, validate_manifest
from src.supervised.deep_features import VERSION, ENCODER_NAME, CACHE, build_deep_matrix, global_view
from src.supervised.high_resolution import prepare_high_resolution
from src.supervised.run_supervised import measure, positive_scores, bootstrap_auc


def candidates():
    result = {}
    # Small, fixed search space declared before evaluating the test set.
    for view in ["global", "global_lungs"]:
        for strength in [0.001, 0.01, 0.1]:
            steps = [FunctionTransformer(global_view)] if view == "global" else []
            result[f"xrv_{view}_C{strength}"] = make_pipeline(*steps, StandardScaler(),
                LogisticRegression(C=strength, class_weight="balanced", max_iter=3000, random_state=42))
    return result


def choose_threshold(labels, scores):
    # Midpoints avoid tie ambiguity. Maximize balanced accuracy on Validation.
    unique = np.unique(np.r_[0.0, scores, 1.0])
    thresholds = np.r_[0.5, (unique[:-1] + unique[1:]) / 2]
    ranked = [(measure(labels, scores, float(t))["balanced_accuracy"], -abs(t - .5), float(t)) for t in thresholds]
    return max(ranked)[2]


def cached_features(frame):
    from src.supervised.deep_features import encoder
    encoder()  # Ensure pretrained weights exist before hashing the cache key.
    weights = list(CACHE.glob("*.pt"))
    if len(weights) != 1:
        raise ValueError("Le cache doit contenir exactement le poids DenseNet attendu.")
    weights_hash = sha256(weights[0])
    key = hashlib.sha256((VERSION + weights_hash + frame.to_csv(index=False)).encode()).hexdigest()
    cache = DATA / f"features_{key}.npz"
    if cache.exists():
        with np.load(cache, allow_pickle=False) as saved:
            return saved["features"], weights_hash
    records = frame.to_dict("records")
    for row in records:
        row["path"] = str(DATA / row["path"])
    features = build_deep_matrix(records)
    np.savez_compressed(cache, features=features)
    return features, weights_hash


def main():
    frame = pd.read_csv(DATA / "manifest_2048.csv") if (DATA / "manifest_2048.csv").exists() else prepare_high_resolution()
    validate_manifest(frame, DATA)
    features, weights_hash = cached_features(frame)
    y = frame.label.to_numpy()
    train_mask = (frame.split == "Train").to_numpy()
    val_mask = (frame.split == "Validation").to_numpy()
    test_mask = (frame.split == "Test").to_numpy()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    models = candidates()
    cv_scores = {}
    started = time.perf_counter()
    for name, model in models.items():
        scores = cross_val_score(model, features[train_mask], y[train_mask], cv=cv, scoring="roc_auc", n_jobs=1)
        cv_scores[name] = {"auc_mean": float(scores.mean()), "auc_std": float(scores.std()), "folds": scores.tolist()}
        print(f"{name} : AUC CV {scores.mean():.3f} ± {scores.std():.3f}", flush=True)
    best = max(cv_scores, key=lambda name: cv_scores[name]["auc_mean"])
    model = models[best].fit(features[train_mask], y[train_mask])
    val_scores = positive_scores(model, features[val_mask])
    threshold = choose_threshold(y[val_mask], val_scores)
    validation = measure(y[val_mask], val_scores, threshold)
    validation["train_seconds"] = round(time.perf_counter() - started, 3)
    baseline = json.loads((REPORT.parent / "baseline" / "metrics.json").read_text())
    baseline_val = baseline["validation"][baseline["selected_model"]]
    approved = validation["roc_auc"] > baseline_val["roc_auc"] and validation["balanced_accuracy"] > baseline_val["balanced_accuracy"]
    # Promotion decision is frozen before test labels/scores are used.
    selection = {"candidate": best, "threshold": threshold, "cv": cv_scores,
                 "validation": validation, "promote_by_validation": approved,
                 "selection_rule": "Highest mean 5-fold Train ROC AUC; threshold maximizes Validation balanced accuracy"}
    (REPORT.parent / "selection_v2.json").write_text(json.dumps(selection, indent=2))
    test_scores = positive_scores(model, features[test_mask])
    test = measure(y[test_mask], test_scores, threshold)
    test["roc_auc_ci95_bootstrap"] = bootstrap_auc(y[test_mask], test_scores)
    report = {
        **baseline, "schema_version": 2, "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": "JSRT 2048 × 2048 PNG / Kaggle raddar", "dataset_size": len(frame),
        "feature_backend": VERSION, "encoder": ENCODER_NAME, "encoder_sha256": weights_hash,
        "preprocess_version": VERSION, "selected_model": best,
        "threshold": threshold, "threshold_policy": "Maximum balanced accuracy on Validation only; research use",
        "selection_rule": selection["selection_rule"], "cross_validation": cv_scores,
        "validation": {best: validation}, "test": test,
        "test_fixed_threshold_05": measure(y[test_mask], test_scores, .5),
        "manifest_sha256": sha256(DATA / "manifest_2048.csv"),
        "data_quality": json.loads((DATA / "quality_2048.json").read_text()),
        "baseline": {"selected_model": baseline["selected_model"], "test": baseline["test"], "validation": baseline_val},
        "test_status": "Exploratory reused cohort: not an untouched external validation",
        "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__, "numpy": np.__version__, **{name: importlib.metadata.version(name) for name in ["torch", "torchvision", "torchxrayvision"]}},
        "limitations": [
            "246 patients JSRT uniquement ; aucun patient supplémentaire ni validation clinique externe.",
            "Le test a déjà été observé pendant le développement : comparaison exploratoire, pas validation indépendante.",
            "Le score n'est pas une probabilité individuelle de cancer ; seuil choisi sur seulement 49 images de validation.",
            "DenseNet est préentraîné sur des pathologies thoraciques ; ses sorties nodule/masse ne sont pas des diagnostics de cancer.",
            "Les sources de préentraînement déclarées par TorchXRayVision n'incluent pas JSRT ; leur absence de recoupement n'a pas été auditée patient par patient.",
            "Images sources 2048 × 2048 en 8 bits ; l'encodeur utilise trois vues de 224 × 224, pas la pleine résolution native.",
            "La classe négative regroupe bénin et sans nodule et ne prouve pas l'absence de cancer.",
            "Les contrôles d'entrée ne reconnaissent pas l'anatomie ; aucune robustesse clinique ou par sous-groupe démontrée.",
        ],
    }
    bundle = {"schema_version": 2, "model": model, "report": report, "preprocess_version": VERSION,
              "feature_backend": VERSION, "encoder_sha256": weights_hash,
              "dataset_hashes": dict(zip(frame.sha256, frame.split))}
    joblib.dump(bundle, MODEL.with_name("candidate_v2.joblib"), compress=3)
    (REPORT.parent / "candidate_v2.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    if approved:
        temporary = MODEL.with_suffix(".tmp")
        joblib.dump(bundle, temporary, compress=3)
        temporary.replace(MODEL)
        REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        frame.to_csv(REPORT.parent / "split_manifest.csv", index=False)
        predictions = frame[test_mask][["patient_id", "state", "label"]].copy()
        predictions["score"] = test_scores
        predictions["prediction"] = (test_scores >= threshold).astype(int)
        predictions.to_csv(REPORT.parent / "test_predictions.csv", index=False)
    print(json.dumps({"promoted": approved, "model": best, "threshold": threshold, "validation": validation, "test": test}, indent=2), flush=True)
    return report


if __name__ == "__main__":
    main()
