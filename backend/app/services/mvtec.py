"""Reads the MVTec AD dataset from disk.

Expected layout (as extracted from the official archive):

    <root>/<category>/train/good/000.png
    <root>/<category>/test/good/000.png
    <root>/<category>/test/<defect_type>/000.png
    <root>/<category>/ground_truth/<defect_type>/000_mask.png
"""

import random
from dataclasses import dataclass
from pathlib import Path

from app.core.config import get_settings

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


@dataclass
class DatasetSample:
    path: Path
    category: str
    label: str  # "good" or the defect type
    mask_path: Path | None


def dataset_root() -> Path:
    return get_settings().mvtec_root


def _images(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def list_categories() -> list[str]:
    root = dataset_root()
    if not root.is_dir():
        return []
    return sorted(
        d.name for d in root.iterdir() if d.is_dir() and (d / "train").is_dir() and (d / "test").is_dir()
    )


def summarize_category(category: str) -> dict:
    cat_dir = dataset_root() / category
    defect_types = {
        d.name: len(_images(d))
        for d in sorted((cat_dir / "test").iterdir())
        if d.is_dir() and d.name != "good"
    }
    return {
        "name": category,
        "train_good": len(_images(cat_dir / "train" / "good")),
        "test_good": len(_images(cat_dir / "test" / "good")),
        "test_defective": sum(defect_types.values()),
        "defect_types": defect_types,
    }


def list_samples(category: str, split: str = "test") -> list[DatasetSample]:
    if category not in list_categories():
        raise ValueError(f"Unknown category '{category}'")
    cat_dir = dataset_root() / category
    samples: list[DatasetSample] = []
    for label_dir in sorted(p for p in (cat_dir / split).iterdir() if p.is_dir()):
        for img in _images(label_dir):
            mask = None
            if label_dir.name != "good":
                candidate = cat_dir / "ground_truth" / label_dir.name / f"{img.stem}_mask.png"
                mask = candidate if candidate.exists() else None
            samples.append(DatasetSample(img, category, label_dir.name, mask))
    return samples


def pick_camera_frames(category: str, count: int, defect_ratio: float, rng: random.Random | None = None) -> list[DatasetSample]:
    """Simulate a production camera: mostly good parts with some defective ones mixed in."""
    rng = rng or random.Random()
    samples = list_samples(category, "test")
    good = [s for s in samples if s.label == "good"]
    bad = [s for s in samples if s.label != "good"]
    frames = []
    for _ in range(count):
        pool = bad if (bad and (not good or rng.random() < defect_ratio)) else good
        frames.append(rng.choice(pool))
    return frames
