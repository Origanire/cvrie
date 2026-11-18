"""Identical image decoding and preprocessing during training and inference."""
import io
import warnings

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

MAX_BYTES = 20 * 1024 * 1024
MAX_PIXELS = 25_000_000
PREPROCESS_VERSION = "grayscale-pad256-hog16-pixels32-v1"


def read_image(content: bytes) -> Image.Image:
    if not content or len(content) > MAX_BYTES:
        raise ValueError("Image vide ou trop volumineuse (maximum 20 Mo).")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(content)) as image:
                if image.format not in {"PNG", "JPEG"}:
                    raise ValueError("Formats acceptés : PNG et JPEG. Exportez les DICOM en image 8 bits.")
                if getattr(image, "n_frames", 1) != 1:
                    raise ValueError("Les images animées ne sont pas acceptées.")
                w, h = image.size
                if min(w, h) < 128 or w * h > MAX_PIXELS:
                    raise ValueError("Résolution requise : au moins 128 × 128, au maximum 25 mégapixels.")
                if not 0.65 <= w / h <= 1.5:
                    raise ValueError("Cadrage incompatible : utilisez une radiographie thoracique de face complète.")
                if image.mode not in {"L", "RGB", "RGBA", "P"}:
                    raise ValueError("Exportez l'image en niveaux de gris 8 bits ou RGB.")
                if "transparency" in image.info:
                    raise ValueError("Les images transparentes ne sont pas acceptées.")
                if image.mode == "RGBA" and image.getextrema()[-1] != (255, 255):
                    raise ValueError("Les images transparentes ne sont pas acceptées.")
                rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
                if np.mean(np.max(rgb, axis=2) - np.min(rgb, axis=2)) > 8:
                    raise ValueError("Image en couleur détectée : importez une radiographie en niveaux de gris.")
                image = ImageOps.exif_transpose(image).convert("L")
                if np.std(np.asarray(image, dtype=np.float32)) < 5:
                    raise ValueError("Image uniforme ou contraste insuffisant pour l'analyse.")
                return image.copy()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise ValueError("Fichier illisible, endommagé ou dimensions excessives.") from exc


def preprocess_image(source, size=(256, 256)) -> np.ndarray:
    if isinstance(source, Image.Image):
        image = source.convert("L")
    else:
        from pathlib import Path
        image = read_image(source if isinstance(source, bytes) else Path(source).read_bytes())
    # Preserve the entire radiograph; never crop peripheral lung tissue away.
    image = ImageOps.pad(image, size, method=Image.Resampling.LANCZOS, color=0)
    return np.asarray(image, dtype=np.float32) / 255.0
