"""Frozen radiograph encoder. No JSRT labels or lesion coordinates enter it."""
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import torchxrayvision as xrv
from PIL import Image, ImageOps

from src.common.paths import ROOT
from src.supervised.preprocess import read_image

ENCODER_NAME = "densenet121-res224-all"
VERSION = "xrv-densenet121-all-global-lungs224-v1"
CACHE = ROOT / "models" / "supervised" / "encoder"


@lru_cache(maxsize=1)
def encoder():
    torch.set_num_threads(4)
    model = xrv.models.DenseNet(weights=ENCODER_NAME, cache_dir=str(CACHE))
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def image_views(source):
    image = source if isinstance(source, Image.Image) else read_image(Path(source).read_bytes())
    image = image.convert("L")
    w, h = image.size
    # Fixed, overlapping anatomical regions, independent of all annotations.
    views = [image, image.crop((int(.03*w), int(.05*h), int(.55*w), int(.95*h))),
             image.crop((int(.45*w), int(.05*h), int(.97*w), int(.95*h)))]
    arrays = []
    for view in views:
        view = ImageOps.pad(view, (224, 224), method=Image.Resampling.LANCZOS, color=0)
        arrays.append((np.asarray(view, dtype=np.float32) / 255.0 * 2 - 1) * 1024)
    return torch.from_numpy(np.stack(arrays)[:, None])


def extract_deep_features(source):
    with torch.inference_mode():
        features = encoder().features2(image_views(source))
    return features.numpy().reshape(-1).copy()


def build_deep_matrix(records):
    features = []
    for index, record in enumerate(records):
        features.append(extract_deep_features(record["path"]))
        if (index + 1) % 25 == 0:
            print(f"Encodage thoracique : {index + 1}/{len(records)}", flush=True)
    return np.stack(features)


def global_view(features):
    """Importable transformer function, safe to resolve when unpickling."""
    return features[:, :1024]
