from PIL import Image, ImageOps
import numpy as np


def center_crop_to_square(img: Image.Image) -> Image.Image:
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    return img.crop((left, top, left + side, top + side))


def preprocess_image(path: str, size=(128, 128)) -> np.ndarray:
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img)
        img = img.convert("L")
        img = center_crop_to_square(img)
        img = img.resize(size)
        arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr