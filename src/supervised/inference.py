"""Inference uses only the trusted, locally generated model artifact."""
import hashlib
import time
from pathlib import Path

import joblib
import numpy as np
import sklearn

from src.common.paths import MODEL
from src.supervised.features import extract_features
from src.supervised.preprocess import PREPROCESS_VERSION, read_image


def load_model(path=MODEL):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError("Modèle absent. Lancez : python -m src.supervised.run_supervised")
    # Never accept uploaded pickle/joblib files: deserialization executes code.
    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or bundle.get("schema_version") not in {1, 2, 3}:
        raise ValueError("Format de modèle incompatible. Réentraînez le modèle.")
    if bundle["schema_version"] == 3:
        from src.supervised.deep_features_512 import VERSION, CACHE
        from src.supervised.deep_features import CACHE as DENSE_CACHE
        from src.supervised.dataset import sha256
        if bundle.get("feature_backend") != VERSION or bundle.get("preprocess_version") != VERSION:
            raise ValueError("Encodeur 512 incompatible.")
        for cache, key in [(CACHE, "encoder_sha256"), (DENSE_CACHE, "dense_encoder_sha256")]:
            weights = list(cache.glob("*.pt"))
            if len(weights) != 1 or sha256(weights[0]) != bundle.get(key):
                raise ValueError("Poids des encodeurs absents ou modifiés.")
    elif bundle["schema_version"] == 2:
        from src.supervised.deep_features import VERSION, CACHE
        from src.supervised.dataset import sha256
        if bundle.get("feature_backend") != VERSION or bundle.get("preprocess_version") != VERSION:
            raise ValueError("Encodeur incompatible. Réentraînez le modèle.")
        weights = list(CACHE.glob("*.pt"))
        if len(weights) != 1 or sha256(weights[0]) != bundle.get("encoder_sha256"):
            raise ValueError("Poids de l'encodeur absents ou modifiés. Relancez python -m src.supervised.improve.")
    elif bundle.get("preprocess_version") != PREPROCESS_VERSION:
        raise ValueError("Prétraitement différent de celui du modèle. Réentraînez le modèle.")
    if bundle["report"]["versions"]["sklearn"] != sklearn.__version__:
        raise ValueError("Version scikit-learn différente de l'entraînement. Réentraînez le modèle.")
    if set(bundle["model"].classes_) != {0, 1}:
        raise ValueError("Classes du modèle incompatibles.")
    return bundle


def predict(content, bundle):
    started = time.perf_counter()
    image = read_image(content)
    if bundle.get("schema_version") == 3:
        from src.supervised.deep_features_512 import extract_features as extract_512
        features = extract_512(image).reshape(1, -1)
    elif bundle.get("schema_version") == 2:
        from src.supervised.deep_features import extract_deep_features
        features = extract_deep_features(image).reshape(1, -1)
    else:
        features = extract_features(image).reshape(1, -1)
    model = bundle["model"]
    score = float(model.predict_proba(features)[0, list(model.classes_).index(1)])
    if not np.isfinite(score) or not 0 <= score <= 1:
        raise ValueError("Le modèle a produit un score invalide.")
    threshold = bundle["report"]["threshold"]
    positive = score >= threshold
    return {
        "research_only": True,
        "model": bundle["report"]["selected_model"],
        "score": score,
        "threshold": threshold,
        "predicted_class": int(positive),
        "label": "Classe « nodule malin »" if positive else "Classe « bénin ou sans nodule »",
        "interpretation": "Classification expérimentale, sans conclusion diagnostique. "
                          "Ce résultat ne confirme ni n'exclut un cancer. Faites interpréter l'examen par un médecin.",
        "score_meaning": "Score de classification du modèle, pas probabilité personnelle de cancer.",
        "known_dataset_split": bundle.get("dataset_hashes", {}).get(hashlib.sha256(content).hexdigest()),
        "elapsed_ms": round((time.perf_counter() - started) * 1000),
        "image_size": list(image.size),
    }
