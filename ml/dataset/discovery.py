"""
VisionInspect AI - MVTec AD Dataset Discovery Module

Discovers and validates MVTec AD dataset directory structures across local development
and cloud/Kaggle environments without hardcoding defect names.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def find_dataset_root(
    custom_path: Optional[str] = None,
    category: str = "bottle",
) -> Path:
    """
    Resolves the MVTec AD root or category directory across environment variables,
    arguments, and standard local/Kaggle paths.
    """
    candidates: List[Path] = []

    # 1. Custom argument or CLI override
    if custom_path:
        candidates.append(Path(custom_path).resolve())

    # 2. Environment variables
    for env_var in ("MVTEC_AD_DIR", "MVTEC_DIR", "DATA_DIR", "DATASET_PATH"):
        val = os.environ.get(env_var)
        if val:
            candidates.append(Path(val).resolve())

    # 3. Known local project and Kaggle candidate paths
    workspace_root = Path(__file__).resolve().parent.parent.parent
    candidates.extend([
        workspace_root / "data" / "mvtec_ad",
        workspace_root / "data" / "mvtec_ad" / category,
        workspace_root / "data" / category,
        workspace_root / "data",
        Path("/kaggle/input/mvtec-ad"),
        Path("/kaggle/input/mvtec-ad/mvtec_anomaly_detection"),
        Path("/kaggle/input/mvtec-ad") / category,
    ])

    for c in candidates:
        if not c.exists():
            continue

        # Case A: c is already the category folder (contains train/good)
        if (c / "train" / "good").exists():
            return c

        # Case B: c is the parent dataset root containing category/
        cat_dir = c / category
        if cat_dir.exists() and (cat_dir / "train" / "good").exists():
            return cat_dir

    searched = "\n  - ".join(str(p) for p in candidates)
    raise FileNotFoundError(
        f"Could not locate MVTec AD '{category}' dataset. Looked in:\n  - {searched}\n"
        f"Set MVTEC_AD_DIR environment variable or pass --data-dir."
    )


def discover_category_dataset(
    category_dir: Path,
) -> Dict[str, Any]:
    """
    Validates that category/train/good, category/test/<defect_type>, and
    category/ground_truth/<defect_type> exist.
    Dynamically discovers defect types from disk without hardcoding names.
    """
    category_dir = category_dir.resolve()
    category_name = category_dir.name

    train_good_dir = category_dir / "train" / "good"
    if not train_good_dir.exists() or not train_good_dir.is_dir():
        raise FileNotFoundError(f"Missing required train/good directory at: {train_good_dir}")

    test_dir = category_dir / "test"
    if not test_dir.exists() or not test_dir.is_dir():
        raise FileNotFoundError(f"Missing test directory at: {test_dir}")

    gt_dir = category_dir / "ground_truth"

    valid_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}

    # 1. Discover train good images
    train_good_files = sorted([
        f for f in train_good_dir.iterdir()
        if f.is_file() and f.suffix.lower() in valid_extensions
    ])

    # 2. Discover test subfolders dynamically
    test_subdirs = [d for d in test_dir.iterdir() if d.is_dir()]
    if not test_subdirs:
        raise ValueError(f"No test subdirectories found in: {test_dir}")

    test_good_files: List[Path] = []
    defects: Dict[str, Dict[str, Any]] = {}

    for sdir in sorted(test_subdirs, key=lambda d: d.name):
        dname = sdir.name
        img_files = sorted([
            f for f in sdir.iterdir()
            if f.is_file() and f.suffix.lower() in valid_extensions
        ])

        if dname == "good":
            test_good_files = img_files
        else:
            # Defect folder: must have corresponding ground_truth/<defect_type>
            gt_defect_dir = gt_dir / dname
            gt_exists = gt_defect_dir.exists() and gt_defect_dir.is_dir()
            gt_masks: List[Path] = []
            if gt_exists:
                gt_masks = sorted([
                    f for f in gt_defect_dir.iterdir()
                    if f.is_file() and f.suffix.lower() in valid_extensions
                ])

            defects[dname] = {
                "test_dir": sdir,
                "gt_dir": gt_defect_dir if gt_exists else None,
                "has_ground_truth": gt_exists,
                "image_count": len(img_files),
                "mask_count": len(gt_masks),
                "image_files": img_files,
                "mask_files": gt_masks,
            }

    # Summary counts
    total_train = len(train_good_files)
    total_test_good = len(test_good_files)
    total_test_defective = sum(d["image_count"] for d in defects.values())
    total_images = total_train + total_test_good + total_test_defective

    discovery_result = {
        "category": category_name,
        "category_dir": category_dir,
        "train_good_dir": train_good_dir,
        "train_good_count": total_train,
        "test_good_count": total_test_good,
        "defects_discovered": list(defects.keys()),
        "defects_details": defects,
        "total_images": total_images,
        "total_train": total_train,
        "total_test": total_test_good + total_test_defective,
    }

    return discovery_result


# Standard expected image counts for MVTec AD categories
EXPECTED_CATEGORY_COUNTS: Dict[str, Dict[str, int]] = {
    "bottle": {
        "train_good": 209,
        "test_good": 20,
        "defects": 63,
        "total": 292,
    },
    "metal_nut": {
        "train_good": 220,
        "test_good": 22,
        "defects": 93,
        "total": 335,
    },
    "tile": {
        "train_good": 230,
        "test_good": 33,
        "defects": 84,
        "total": 347,
    },
}


def print_discovery_summary(info: Dict[str, Any]) -> None:
    print("=" * 65)
    print(f"📦 MVTec AD Dataset Discovery Summary: [{info['category'].upper()}]")
    print(f"📁 Path: {info['category_dir']}")
    print("=" * 65)
    print(f"  • Train (good normal): {info['train_good_count']} frames")
    print(f"  • Test (good normal) : {info['test_good_count']} frames")
    print(f"  • Defect Categories  : {len(info['defects_discovered'])} types detected")

    for defect_name, details in info["defects_details"].items():
        gt_status = "✅ GT Masks Present" if details["has_ground_truth"] else "❌ MISSING GT"
        print(f"      - {defect_name.ljust(18)} : {details['image_count']} images, {details['mask_count']} masks ({gt_status})")

    print("-" * 65)
    print(f"  Total Images Discovered: {info['total_images']}")

    # Startup validation against expected full dataset counts
    cat_key = info["category"].lower()
    if cat_key in EXPECTED_CATEGORY_COUNTS:
        expected = EXPECTED_CATEGORY_COUNTS[cat_key]
        exp_total = expected["total"]
        if info["total_images"] < exp_total:
            print(
                f"  ⚠️  WARNING: Discovered image count ({info['total_images']}) is below standard "
                f"full MVTec AD {cat_key} count ({exp_total}). Incomplete or sample dataset detected!"
            )
        else:
            print(f"  ✅ Verified: Full MVTec AD {cat_key} dataset ({info['total_images']}/{exp_total} images)")
    print("=" * 65)


def main():
    parser = argparse.ArgumentParser(description="MVTec AD Dataset Discovery")
    parser.add_argument("--data-dir", default=None, help="Root path to MVTec AD or category folder")
    parser.add_argument("--category", default="bottle", help="Target category (default: bottle)")
    args = parser.parse_args()

    try:
        cat_path = find_dataset_root(custom_path=args.data_dir, category=args.category)
        info = discover_category_dataset(cat_path)
        print_discovery_summary(info)
    except Exception as e:
        print(f"❌ Discovery error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
