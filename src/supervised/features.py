import numpy as np
from PIL import Image
from skimage.feature import hog

from src.supervised.preprocess import preprocess_image


def extract_features(source):
    array = preprocess_image(source)
    gradients = hog(array, orientations=9, pixels_per_cell=(16, 16),
                    cells_per_block=(2, 2), block_norm="L2-Hys")
    low_res = np.asarray(Image.fromarray(array).resize((32, 32)), dtype=np.float32).ravel()
    return np.concatenate([gradients, low_res]).astype(np.float32)


def build_feature_matrix(records):
    if not records:
        raise ValueError("Aucune image à traiter.")
    features = np.stack([extract_features(record["path"]) for record in records])
    return features, np.array([r["label"] for r in records]), [r["path"] for r in records]
