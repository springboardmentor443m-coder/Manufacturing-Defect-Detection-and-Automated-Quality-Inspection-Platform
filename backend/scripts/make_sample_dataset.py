"""Create a tiny synthetic dataset with the same folder layout as MVTec AD.

The real MVTec AD (~5 GB) must be downloaded manually after accepting its licence
(see docs/dataset.md). This sample lets you run the app and tests before that.

    python scripts/make_sample_dataset.py [output_root]
"""

import sys
from pathlib import Path

import cv2
import numpy as np

DEFECTS = ("scratch", "crack", "hole")


def _base_part(rng: np.random.Generator, size: int) -> np.ndarray:
    """A brushed-metal disc on a dark background."""
    img = np.full((size, size, 3), 40, np.uint8)
    texture = rng.normal(150, 12, (size, size)).astype(np.float32)
    texture = cv2.GaussianBlur(texture, (0, 0), sigmaX=6, sigmaY=0.8)  # brushed look
    disc = np.zeros((size, size), np.uint8)
    cv2.circle(disc, (size // 2, size // 2), int(size * 0.42), 255, -1)
    for c in range(3):
        img[:, :, c] = np.where(disc > 0, np.clip(texture + c * 4, 0, 255), img[:, :, c])
    return img


def _add_defect(img: np.ndarray, kind: str, rng: np.random.Generator) -> np.ndarray:
    size = img.shape[0]
    mask = np.zeros(img.shape[:2], np.uint8)
    c = size // 2
    if kind == "scratch":
        p1 = rng.integers(c - 60, c + 60, 2)
        p2 = p1 + rng.integers(-70, 70, 2)
        cv2.line(img, tuple(int(v) for v in p1), tuple(int(v) for v in p2), (230, 230, 230), 2)
        cv2.line(mask, tuple(int(v) for v in p1), tuple(int(v) for v in p2), 255, 4)
    elif kind == "crack":
        pts = [rng.integers(c - 40, c + 40, 2)]
        for _ in range(6):
            pts.append(pts[-1] + rng.integers(-18, 18, 2))
        pts_arr = np.array(pts, np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [pts_arr], False, (20, 20, 20), 2)
        cv2.polylines(mask, [pts_arr], False, 255, 5)
    else:  # hole
        center = tuple(int(v) for v in rng.integers(c - 50, c + 50, 2))
        r = int(rng.integers(6, 14))
        cv2.circle(img, center, r, (15, 15, 15), -1)
        cv2.circle(mask, center, r + 2, 255, -1)
    return mask


def make_sample(root: Path, category: str = "metal_disc", size: int = 256,
                n_train: int = 12, n_test_good: int = 6, n_per_defect: int = 4, seed: int = 7) -> Path:
    rng = np.random.default_rng(seed)
    cat = root / category

    def save(path: Path, img: np.ndarray) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), img)

    for i in range(n_train):
        save(cat / "train" / "good" / f"{i:03d}.png", _base_part(rng, size))
    for i in range(n_test_good):
        save(cat / "test" / "good" / f"{i:03d}.png", _base_part(rng, size))
    for kind in DEFECTS:
        for i in range(n_per_defect):
            img = _base_part(rng, size)
            mask = _add_defect(img, kind, rng)
            save(cat / "test" / kind / f"{i:03d}.png", img)
            save(cat / "ground_truth" / kind / f"{i:03d}_mask.png", mask)
    return cat


if __name__ == "__main__":
    default = Path(__file__).resolve().parents[2] / "data" / "mvtec_anomaly_detection"
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else default
    print(f"Sample dataset written to {make_sample(out)}")
