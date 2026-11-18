"""Frozen 512px chest ResNet encoder plus original DenseNet global features."""
import hashlib
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import torchxrayvision as xrv
from PIL import Image, ImageOps

from src.common.paths import DATA, ROOT
from src.supervised.dataset import sha256
from src.supervised.deep_features import extract_deep_features
from src.supervised.preprocess import read_image

VERSION = 'xrv-resnet512-densenet224-fusion-v1'
ENCODER_NAME = 'resnet50-res512-all'
CACHE = ROOT / 'models/supervised/encoder512'


@lru_cache(maxsize=1)
def encoder():
    torch.set_num_threads(4)
    model = xrv.models.ResNet(weights=ENCODER_NAME, cache_dir=str(CACHE))
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def extract_features(source):
    image = source if isinstance(source, Image.Image) else read_image(Path(source).read_bytes())
    view = ImageOps.pad(image.convert('L'), (512, 512), method=Image.Resampling.LANCZOS, color=0)
    tensor = torch.from_numpy((np.asarray(view, dtype=np.float32) / 255 * 2 - 1) * 1024)[None, None]
    with torch.inference_mode():
        features = encoder().features(tensor).numpy().reshape(-1)
    return np.concatenate([features, extract_deep_features(image)[:1024]])


def resnet_view(features):
    return features[:, :2048]


def cached_features(frame):
    from src.supervised.deep_features import CACHE as dense_cache, encoder as dense_encoder
    encoder(); dense_encoder()
    weights = list(CACHE.glob('*.pt')); dense_weights = list(dense_cache.glob('*.pt'))
    if len(weights) != 1 or len(dense_weights) != 1:
        raise ValueError('Exactly one verified weight file per encoder is required')
    digest = sha256(weights[0]); dense_digest = sha256(dense_weights[0])
    key = hashlib.sha256((VERSION + digest + dense_digest + frame.to_csv(index=False)).encode()).hexdigest()
    cache = DATA / f'features_{key}.npz'
    if cache.exists():
        with np.load(cache, allow_pickle=False) as saved:
            return saved['features'], digest
    rows = []
    for i, path in enumerate(frame.path):
        rows.append(extract_features(DATA / path))
        if (i + 1) % 25 == 0:
            print(f'Encodage 512px : {i + 1}/{len(frame)}', flush=True)
    features = np.stack(rows)
    np.savez_compressed(cache, features=features)
    return features, digest
