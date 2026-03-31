from pathlib import Path
import joblib

from supervised.data_cleaning import scan_dataset_root
from supervised.features import build_feature_matrix
from supervised.train import build_models
from supervised.evaluate import evaluate_model, results_to_dataframe


DATASET_ROOT = Path("data/supervised/raw")
MODEL_OUTPUT_DIR = Path("models/supervised")
MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def split_records(records):
    train = [r for r in records if r["split"] == "Train"]
    val = [r for r in records if r["split"] == "Validation"]
    test = [r for r in records if r["split"] == "Test"]
    return train, val, test


def main():
    print("Scanning dataset...")
    records = scan_dataset_root(DATASET_ROOT)
    train_rec, val_rec, test_rec = split_records(records)

    print("Building features...")
    X_train, y_train, _ = build_feature_matrix(train_rec)
    X_val, y_val, _ = build_feature_matrix(val_rec)
    X_test, y_test, _ = build_feature_matrix(test_rec)

    models = build_models()

    results = {}

    for name, model in models.items():
        print(f"Training {name}...")
        metrics = evaluate_model(model, X_train, y_train, X_val, y_val)
        results[name] = metrics

        model_path = MODEL_OUTPUT_DIR / f"{name}.joblib"
        joblib.dump(model, model_path)

    df = results_to_dataframe(results)
    print(df)

    best_model_name = df.iloc[0]["model"]
    print("Best model:", best_model_name)

    best_model = models[best_model_name]
    best_model.fit(X_train, y_train)

    joblib.dump(best_model, MODEL_OUTPUT_DIR / "best_model.joblib")
    print("Best model saved.")


if __name__ == "__main__":
    main()