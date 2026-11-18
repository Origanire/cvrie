"""Upgrade JSRT images while retaining exactly the previous patient splits."""
import hashlib
import io
import json
import zipfile

import pandas as pd
from PIL import Image

from src.common.paths import DATA
from src.supervised.dataset import SOURCES, download, prepare, sha256, validate_manifest

URL = "https://www.kaggle.com/api/v1/datasets/download/raddar/nodules-in-chest-xrays-jsrt"
DIGEST = "0de426a7abe4009de56ac37df4b851d3b8e138f3eb98a95e9dd82780e2b54678"


def prepare_high_resolution(root=DATA):
    if not (root / "manifest.csv").exists():
        prepare(root)
    base = pd.read_csv(root / "manifest.csv")
    archive_path = root / "downloads" / "jsrt_full.zip"
    download(URL, archive_path, DIGEST)
    images = root / "images_2048"
    images.mkdir(exist_ok=True)
    frame = base.copy()
    pixel_hashes = []
    with zipfile.ZipFile(archive_path) as archive:
        metadata = archive.read("jsrt_metadata.csv")
        if hashlib.sha256(metadata).hexdigest() != SOURCES["jsrt_metadata.csv"][1]:
            raise ValueError("Les annotations haute résolution diffèrent de la source vérifiée.")
        for idx, row in frame.iterrows():
            with Image.open(io.BytesIO(archive.read(f"images/images/{row.patient_id}.png"))) as im:
                if im.size != (2048, 2048):
                    raise ValueError("Dimensions source inattendues.")
                # Verify palette is grayscale rather than silently discarding colour.
                rgb = im.convert("RGB")
                r, g, b = rgb.split()
                if r.tobytes() != g.tobytes() or r.tobytes() != b.tobytes():
                    raise ValueError("La source contient des couleurs inattendues.")
                image = im.convert("L")
                pixel_hashes.append(hashlib.sha256(image.tobytes()).hexdigest())
                path = images / f"{row.patient_id}.png"
                image.save(path)
            frame.loc[idx, "path"] = f"images_2048/{row.patient_id}.png"
            frame.loc[idx, "sha256"] = sha256(path)
    if len(set(pixel_hashes)) != len(frame):
        raise ValueError("Doublons de pixels détectés dans la nouvelle source.")
    validate_manifest(frame, root)
    frame.to_csv(root / "manifest_2048.csv", index=False)
    audit = {"source_url": URL, "archive_sha256": DIGEST, "source_images": 247,
             "used_images": len(frame), "image_size": [2048, 2048], "bit_depth": 8,
             "excluded": {"JPCNN014": "Excluded in baseline due to duplicate; kept excluded for paired comparison"},
             "patient_splits_preserved": True, "unique_decoded_pixel_hashes": len(set(pixel_hashes)),
             "new_patients": 0, "note": "Higher resolution of the same patients, not an independent dataset."}
    (root / "quality_2048.json").write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2), flush=True)
    return frame


if __name__ == "__main__":
    prepare_high_resolution()
