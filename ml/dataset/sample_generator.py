"""
VisionInspect AI - Synthetic MVTec AD Bottle Dataset Generator

Generates a standard-compliant MVTec AD bottle dataset structure for local development,
testing, and continuous integration:
bottle/
  train/
    good/              (000.png, 001.png, ...)
  test/
    good/              (000.png, ...)
    broken_large/      (000.png, ...)
    broken_small/      (000.png, ...)
    contamination/     (000.png, ...)
  ground_truth/
    broken_large/      (000_mask.png, ...)
    broken_small/      (000_mask.png, ...)
    contamination/     (000_mask.png, ...)
"""

import os
from pathlib import Path
import random
import tempfile
import cv2
import numpy as np


def generate_synthetic_bottle_dataset(
    output_root: Path,
    num_train_good: int = 24,
    num_test_good: int = 8,
    num_broken_large: int = 8,
    num_broken_small: int = 8,
    num_contamination: int = 8,
    seed: int = 42,
) -> Path:
    """
    Generates synthetic MVTec AD bottle images with pixel-accurate defect ground-truth masks.
    RESERVED FOR UNIT TESTS AND CI ONLY. Never writes to data/mvtec_ad.
    """
    output_root = Path(output_root).resolve()
    # Guard: never overwrite production dataset folder
    if "data" in output_root.parts and "mvtec_ad" in output_root.parts:
        raise ValueError(
            f"Safety Violation: sample_generator is restricted to unit tests and CI. "
            f"It cannot write to production data directory: {output_root}"
        )

    random.seed(seed)
    np.random.seed(seed)

    bottle_root = output_root / "bottle"
    train_good_dir = bottle_root / "train" / "good"
    test_good_dir = bottle_root / "test" / "good"
    
    defect_types = {
        "broken_large": num_broken_large,
        "broken_small": num_broken_small,
        "contamination": num_contamination,
    }

    train_good_dir.mkdir(parents=True, exist_ok=True)
    test_good_dir.mkdir(parents=True, exist_ok=True)

    img_w, img_h = 512, 512

    def draw_base_bottle() -> np.ndarray:
        # Background: dark industrial conveyor
        img = np.full((img_h, img_w, 3), 32, dtype=np.uint8)
        # Bottle body: light amber/translucent glass
        # Neck
        cv2.rectangle(img, (226, 60), (286, 150), (140, 160, 180), -1)
        # Cap
        cv2.rectangle(img, (220, 40), (292, 60), (200, 70, 50), -1)
        # Shoulder taper
        pts = np.array([[226, 150], [286, 150], [350, 210], [162, 210]], np.int32)
        cv2.fillPoly(img, [pts], (130, 150, 175))
        # Main body
        cv2.rectangle(img, (162, 210), (350, 460), (125, 145, 170), -1)
        # Subtle glass specular highlight line
        cv2.line(img, (185, 215), (185, 455), (195, 215, 235), 4)
        return img

    # 1. Train Good
    for i in range(num_train_good):
        img = draw_base_bottle()
        # Add slight natural variation
        noise = np.random.normal(0, 2, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        cv2.imwrite(str(train_good_dir / f"{i:03d}.png"), img)

    # 2. Test Good
    for i in range(num_test_good):
        img = draw_base_bottle()
        noise = np.random.normal(0, 2, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        cv2.imwrite(str(test_good_dir / f"{i:03d}.png"), img)

    # 3. Test Defective Categories + Ground Truth Masks
    for defect_name, count in defect_types.items():
        test_defect_dir = bottle_root / "test" / defect_name
        gt_defect_dir = bottle_root / "ground_truth" / defect_name
        test_defect_dir.mkdir(parents=True, exist_ok=True)
        gt_defect_dir.mkdir(parents=True, exist_ok=True)

        for i in range(count):
            img = draw_base_bottle()
            mask = np.zeros((img_h, img_w), dtype=np.uint8)

            if defect_name == "broken_large":
                # Large jagged crack on bottle shoulder/body
                cx = random.randint(200, 310)
                cy = random.randint(220, 360)
                r = random.randint(28, 45)
                pts = []
                for a in range(0, 360, 45):
                    rad = np.radians(a)
                    dist = r + random.randint(-10, 15)
                    pts.append([int(cx + dist * np.cos(rad)), int(cy + dist * np.sin(rad))])
                poly = np.array(pts, np.int32)
                cv2.fillPoly(mask, [poly], 255)
                # Render defect onto image: dark jagged crack
                cv2.fillPoly(img, [poly], (20, 20, 25))
                cv2.polylines(img, [poly], isClosed=True, color=(230, 240, 255), thickness=2)

            elif defect_name == "broken_small":
                # Small chip or puncture
                cx = random.randint(190, 320)
                cy = random.randint(180, 400)
                r = random.randint(8, 16)
                cv2.circle(mask, (cx, cy), r, 255, -1)
                cv2.circle(img, (cx, cy), r, (15, 15, 20), -1)
                cv2.circle(img, (cx, cy), r, (220, 230, 240), 1)

            elif defect_name == "contamination":
                # Dark blotch / liquid smudge
                cx = random.randint(200, 300)
                cy = random.randint(230, 380)
                axes = (random.randint(15, 30), random.randint(20, 45))
                angle = random.randint(0, 180)
                cv2.ellipse(mask, (cx, cy), axes, angle, 0, 360, 255, -1)
                cv2.ellipse(img, (cx, cy), axes, angle, 0, 360, (20, 40, 50), -1)

            # Save test image and matching ground-truth mask (<id>_mask.png)
            cv2.imwrite(str(test_defect_dir / f"{i:03d}.png"), img)
            cv2.imwrite(str(gt_defect_dir / f"{i:03d}_mask.png"), mask)

    print(f"✨ Standard MVTec AD bottle dataset generated at: {bottle_root.resolve()}")
    return bottle_root


if __name__ == "__main__":
    temp_dir = Path(tempfile.mkdtemp(prefix="mvtec_synthetic_test_"))
    print(f"Running CI/test synthetic generator into temporary directory: {temp_dir}")
    generate_synthetic_bottle_dataset(temp_dir)
