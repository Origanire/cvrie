"""Reproducible miniJSRT download, label join and patient-level split."""
import argparse
import hashlib
import json
import zipfile
from pathlib import Path

import pandas as pd
import requests
from sklearn.model_selection import train_test_split

from src.common.paths import DATA

SOURCES = {
    "Gender01.zip": (
        "http://imgcom.jsrt.or.jp/imgcom/wp-content/uploads/2018/11/Gender01.zip",
        "97a32539848c3531586297146c10d0266a00c0821e5d7893b5f8fdb5bc3ae1c1",
    ),
    "jsrt_metadata.csv": (
        "https://www.kaggle.com/api/v1/datasets/download/raddar/nodules-in-chest-xrays-jsrt/jsrt_metadata.csv",
        "f35076411f4b41218b752e7311c0ae64f9bef031ee3de1d54d3ba898e853874e",
    ),
}
STATES = {"malignant": 1, "benign": 0, "non-nodule": 0}


def sha256(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def download(url, destination, expected):
    if destination.exists() and sha256(destination) == expected:
        return
    temporary = destination.with_suffix(destination.suffix + ".part")
    try:
        with requests.get(url, stream=True, timeout=(15, 120)) as response:
            response.raise_for_status()
            with temporary.open("wb") as output:
                for chunk in response.iter_content(1024 * 1024):
                    output.write(chunk)
        if sha256(temporary) != expected:
            raise ValueError(f"Source modifiée ou téléchargement incomplet : {destination.name}")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def split_manifest(frame):
    # Exactly one radiograph per JSRT study/patient. Stratify all 3 original states.
    train_val, test = train_test_split(frame, test_size=0.20, stratify=frame.state, random_state=42)
    train, val = train_test_split(train_val, test_size=0.25, stratify=train_val.state, random_state=42)
    result = pd.concat([train.assign(split="Train"), val.assign(split="Validation"), test.assign(split="Test")])
    return result.sort_values("patient_id").reset_index(drop=True)


def validate_manifest(frame, root):
    required = {"path", "patient_id", "state", "label", "split", "sha256"}
    if not required.issubset(frame.columns) or frame.empty:
        raise ValueError("Manifeste incomplet ou vide.")
    if frame.patient_id.duplicated().any() or frame.sha256.duplicated().any():
        raise ValueError("Patients ou images dupliqués : risque de fuite entre les jeux.")
    if not frame.state.isin(STATES).all() or not (frame.label == frame.state.map(STATES)).all():
        raise ValueError("Étiquettes incohérentes : un nodule bénin ne doit pas être étiqueté malin.")
    if set(frame.split) != {"Train", "Validation", "Test"}:
        raise ValueError("Les trois jeux Train/Validation/Test sont requis.")
    for split, subset in frame.groupby("split"):
        if set(subset.label) != {0, 1}:
            raise ValueError(f"Les deux classes sont requises dans {split}.")
    for row in frame.itertuples():
        path = (root / row.path).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file():
            raise ValueError("Chemin d'image absent ou en dehors du jeu de données.")
        if sha256(path) != row.sha256:
            raise ValueError(f"Image modifiée depuis la préparation : {row.patient_id}")


def prepare(root=DATA):
    root = Path(root)
    downloads = root / "downloads"
    images = root / "images"
    downloads.mkdir(parents=True, exist_ok=True)
    images.mkdir(parents=True, exist_ok=True)
    for name, (url, digest) in SOURCES.items():
        print(f"Vérification / téléchargement : {name}", flush=True)
        download(url, downloads / name, digest)
    metadata = pd.read_csv(downloads / "jsrt_metadata.csv")
    if metadata.state.value_counts().to_dict() != {"malignant": 100, "non-nodule": 93, "benign": 54}:
        raise ValueError("Les annotations ne correspondent pas aux 247 cas JSRT attendus.")
    records = []
    with zipfile.ZipFile(downloads / "Gender01.zip") as archive:
        members = {}
        for name in archive.namelist():
            if name.endswith(".png"):
                patient_id = Path(name).stem
                if patient_id in members:
                    raise ValueError("Image JSRT dupliquée dans l'archive.")
                members[patient_id] = name
        if len(members) != 247:
            raise ValueError("L'archive doit contenir 247 images uniques.")
        for row in metadata.itertuples():
            patient = Path(row.study_id).stem
            if not patient.startswith(("JPCLN", "JPCNN")) or not patient[-3:].isdigit():
                raise ValueError("Identifiant JSRT invalide.")
            destination = images / f"{patient}.png"
            # Read only expected members, never extract arbitrary archive paths.
            destination.write_bytes(archive.read(members[patient]))
            records.append({"path": f"images/{patient}.png", "patient_id": patient,
                            "state": row.state, "label": STATES[row.state], "sha256": sha256(destination)})
    frame = pd.DataFrame(records).sort_values("patient_id")
    # Official miniJSRT archive contains the same pixels for JPCNN013/014.
    # Remove exact duplicates before any split; fail on conflicting annotations.
    if (frame.groupby("sha256").state.nunique() > 1).any():
        raise ValueError("Images identiques avec annotations contradictoires.")
    excluded = frame[frame.duplicated("sha256", keep="first")]
    excluded.to_csv(root / "excluded_duplicates.csv", index=False)
    if not excluded.empty:
        print("Doublons exclus avant séparation : " + ", ".join(excluded.patient_id), flush=True)
    manifest = split_manifest(frame.drop_duplicates("sha256", keep="first"))
    validate_manifest(manifest, root)
    manifest.to_csv(root / "manifest.csv", index=False)
    (root / "sources.json").write_text(json.dumps(SOURCES, indent=2))
    print(manifest.groupby(["split", "state"]).size().to_string())
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Télécharger et préparer les 247 radiographies JSRT.")
    parser.add_argument("--data-dir", type=Path, default=DATA)
    prepare(parser.parse_args().data_dir)


if __name__ == "__main__":
    main()
