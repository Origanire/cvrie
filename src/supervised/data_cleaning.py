from pathlib import Path
from typing import List, Dict

VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def infer_label_from_filename(filename: str) -> int:
    name = filename.lower().strip()
    if name.startswith("not cancer"):
        return 0
    if name.startswith("cancer"):
        return 1
    raise ValueError(f"Cannot infer label from filename: {filename}")


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in VALID_EXT


def scan_dataset_split(split_dir: Path) -> List[Dict]:
    rows = []
    for path in sorted(split_dir.iterdir()):
        if not path.is_file():
            continue
        if not is_image_file(path):
            continue

        try:
            label = infer_label_from_filename(path.name)
        except Exception:
            continue

        rows.append({
            "path": str(path),
            "filename": path.name,
            "label": label,
            "split": split_dir.name,
        })
    return rows


def scan_dataset_root(root_dir: Path):
    records = []
    for split in ["Train", "Validation", "Test"]:
        split_dir = root_dir / split
        records.extend(scan_dataset_split(split_dir))
    return records