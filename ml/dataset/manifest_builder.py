"""
VisionInspect AI - Protocol B Dataset Manifest Builder & Integrity Verifier

Builds standardized CSV manifests for supervised and unsupervised defect models
following Protocol B:
- original train/good: ~85% train, 15% validation
- each original test/<defect_type>: ~60% train, 20% validation, 20% final test (min 1 in val and test)
- original test/good: 100% final held-out test
- Fixed random seed for 100% reproducible splitting
- Never moves or alters source image files

Performs strict integrity audits:
- Disjoint splits (no image in more than one split)
- mask_path is empty if and only if is_anomaly == 0
- Image and mask spatial dimensions match 1:1
- Manifest CSVs written with complete split breakdown
"""

import argparse
import math
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import cv2
import pandas as pd

from ml.dataset.discovery import discover_category_dataset, find_dataset_root


def split_items_protocol_b(
    items: List[Any],
    train_ratio: float = 0.60,
    val_ratio: float = 0.20,
    test_ratio: float = 0.20,
    min_val: int = 1,
    min_test: int = 1,
    seed: int = 42,
) -> Tuple[List[Any], List[Any], List[Any]]:
    """Splits a list of items into (train, val, test) following Protocol B guarantees."""
    n = len(items)
    if n == 0:
        return [], [], []

    rng = random.Random(seed)
    shuffled = list(items)
    rng.shuffle(shuffled)

    if n == 1:
        # Single sample: put in validation or test
        return [], shuffled, []
    if n == 2:
        return [shuffled[0]], [shuffled[1]], []

    n_val = max(min_val, int(round(n * val_ratio)))
    n_test = max(min_test, int(round(n * test_ratio)))

    # Ensure train gets remaining
    if n_val + n_test >= n:
        n_val = 1
        n_test = 1

    n_train = n - n_val - n_test
    if n_train <= 0 and n >= 3:
        n_train = 1
        if n_val > 1:
            n_val -= 1
        elif n_test > 1:
            n_test -= 1

    train_slice = shuffled[:n_train]
    val_slice = shuffled[n_train:n_train + n_val]
    test_slice = shuffled[n_train + n_val:]

    return train_slice, val_slice, test_slice


def build_protocol_b_manifest(
    category_dir: Path,
    output_dir: Optional[Path] = None,
    seed: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Builds Protocol B dataset manifest dataframe and exports CSV files.
    """
    if output_dir is None:
        output_dir = Path(__file__).resolve().parent.parent / "manifests"
    output_dir.mkdir(parents=True, exist_ok=True)

    info = discover_category_dataset(category_dir)
    category = info["category"]

    rows: List[Dict[str, Any]] = []

    # ==============================================================
    # 1. Original train/good: ~85% train, ~15% validation
    # ==============================================================
    train_good_files = sorted(info["train_good_dir"].glob("*.png"))
    rng = random.Random(seed)
    shuffled_train_good = list(train_good_files)
    rng.shuffle(shuffled_train_good)

    n_tg = len(shuffled_train_good)
    n_tg_val = max(1, int(round(n_tg * 0.15))) if n_tg > 1 else 0
    n_tg_train = n_tg - n_tg_val

    for img_p in shuffled_train_good[:n_tg_train]:
        rows.append({
            "image_path": str(img_p.resolve()),
            "mask_path": "",
            "category": category,
            "defect_type": "good",
            "is_anomaly": 0,
            "project_split": "train",
        })

    for img_p in shuffled_train_good[n_tg_train:]:
        rows.append({
            "image_path": str(img_p.resolve()),
            "mask_path": "",
            "category": category,
            "defect_type": "good",
            "is_anomaly": 0,
            "project_split": "val",
        })

    # ==============================================================
    # 2. Each original test/<defect_type>: ~60% train, 20% val, 20% test
    # ==============================================================
    for idx, (defect_name, details) in enumerate(sorted(info["defects_details"].items())):
        pairs = []
        for img_p in details["image_files"]:
            stem = img_p.stem
            expected_mask = details["gt_dir"] / f"{stem}_mask{img_p.suffix}"
            if not expected_mask.exists():
                expected_mask_png = details["gt_dir"] / f"{stem}_mask.png"
                if expected_mask_png.exists():
                    expected_mask = expected_mask_png
                else:
                    raise FileNotFoundError(f"Missing mask for defect image: {img_p}")

            pairs.append((img_p, expected_mask))

        # Split defect samples
        tr_pairs, val_pairs, te_pairs = split_items_protocol_b(
            pairs,
            train_ratio=0.60,
            val_ratio=0.20,
            test_ratio=0.20,
            min_val=1,
            min_test=1,
            seed=seed + idx + 1,
        )

        for img_p, msk_p in tr_pairs:
            rows.append({
                "image_path": str(img_p.resolve()),
                "mask_path": str(msk_p.resolve()),
                "category": category,
                "defect_type": defect_name,
                "is_anomaly": 1,
                "project_split": "train",
            })

        for img_p, msk_p in val_pairs:
            rows.append({
                "image_path": str(img_p.resolve()),
                "mask_path": str(msk_p.resolve()),
                "category": category,
                "defect_type": defect_name,
                "is_anomaly": 1,
                "project_split": "val",
            })

        for img_p, msk_p in te_pairs:
            rows.append({
                "image_path": str(img_p.resolve()),
                "mask_path": str(msk_p.resolve()),
                "category": category,
                "defect_type": defect_name,
                "is_anomaly": 1,
                "project_split": "test",
            })

    # ==============================================================
    # 3. Original test/good: 100% final held-out test
    # ==============================================================
    test_good_dir = category_dir / "test" / "good"
    if test_good_dir.exists():
        test_good_files = sorted(test_good_dir.glob("*.png"))
        for img_p in test_good_files:
            rows.append({
                "image_path": str(img_p.resolve()),
                "mask_path": "",
                "category": category,
                "defect_type": "good",
                "is_anomaly": 0,
                "project_split": "test",
            })

    df = pd.DataFrame(rows)

    # ==============================================================
    # 4. Strict Integrity Verification
    # ==============================================================
    integrity_report = verify_manifest_integrity(df)

    # 5. Export CSV files
    all_manifest_path = output_dir / f"manifest_{category}_protocol_b.csv"
    train_manifest_path = output_dir / f"manifest_{category}_train.csv"
    val_manifest_path = output_dir / f"manifest_{category}_val.csv"
    test_manifest_path = output_dir / f"manifest_{category}_test.csv"

    df.to_csv(all_manifest_path, index=False)
    df[df["project_split"] == "train"].to_csv(train_manifest_path, index=False)
    df[df["project_split"] == "val"].to_csv(val_manifest_path, index=False)
    df[df["project_split"] == "test"].to_csv(test_manifest_path, index=False)

    integrity_report["manifest_files"] = {
        "all": str(all_manifest_path),
        "train": str(train_manifest_path),
        "val": str(val_manifest_path),
        "test": str(test_manifest_path),
    }

    print_manifest_summary(df, integrity_report)
    return df, integrity_report


def verify_manifest_integrity(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Audits the generated manifest:
    - No duplicate image paths across splits (disjoint)
    - mask_path is empty if and only if is_anomaly == 0
    - Mask dimensions match image dimensions exactly
    """
    errors: List[str] = []

    # Check 1: Disjoint splits & no duplicate images
    all_imgs = df["image_path"].tolist()
    if len(all_imgs) != len(set(all_imgs)):
        duplicates = [p for p in set(all_imgs) if all_imgs.count(p) > 1]
        errors.append(f"DUPLICATE IMAGES: Found {len(duplicates)} images appearing in multiple rows!")

    splits = {s: set(df[df["project_split"] == s]["image_path"]) for s in ("train", "val", "test")}
    for s1 in ("train", "val", "test"):
        for s2 in ("train", "val", "test"):
            if s1 < s2:
                overlap = splits[s1].intersection(splits[s2])
                if overlap:
                    errors.append(f"SPLIT OVERLAP: {len(overlap)} images overlap between '{s1}' and '{s2}'!")

    # Check 2: mask_path is empty only when is_anomaly == 0
    anom_missing_mask = df[(df["is_anomaly"] == 1) & (df["mask_path"].isna() | (df["mask_path"] == ""))]
    if len(anom_missing_mask) > 0:
        errors.append(f"ANOMALY WITHOUT MASK: {len(anom_missing_mask)} defective rows have empty mask_path!")

    normal_with_mask = df[(df["is_anomaly"] == 0) & (df["mask_path"].notna()) & (df["mask_path"] != "")]
    if len(normal_with_mask) > 0:
        errors.append(f"NORMAL WITH MASK: {len(normal_with_mask)} normal rows have non-empty mask_path!")

    # Check 3: Dimension matching for all masks
    mismatches = 0
    for _, row in df[df["is_anomaly"] == 1].iterrows():
        img_p = row["image_path"]
        msk_p = row["mask_path"]
        if not os.path.exists(msk_p):
            errors.append(f"MASK MISSING ON DISK: {msk_p}")
            continue

        im = cv2.imread(img_p)
        mk = cv2.imread(msk_p, cv2.IMREAD_GRAYSCALE)
        if im is None or mk is None:
            errors.append(f"UNABLE TO DECODE: {img_p} or {msk_p}")
            continue

        if im.shape[:2] != mk.shape[:2]:
            errors.append(f"DIMENSION MISMATCH: Image {img_p} {im.shape[:2]} != Mask {msk_p} {mk.shape[:2]}")
            mismatches += 1

    return {
        "passed": len(errors) == 0,
        "error_count": len(errors),
        "errors": errors,
        "total_rows": len(df),
        "split_counts": df["project_split"].value_counts().to_dict(),
        "anomaly_distribution": df.groupby(["project_split", "is_anomaly"]).size().unstack(fill_value=0).to_dict(),
    }


def print_manifest_summary(df: pd.DataFrame, integrity: Dict[str, Any]) -> None:
    print("=" * 65)
    print("📋 PROTOCOL B MANIFEST CREATION & INTEGRITY REPORT")
    print("=" * 65)
    print(f"  • Integrity Check : {'PASSED ✅' if integrity['passed'] else 'FAILED ❌'}")
    if integrity["errors"]:
        for e in integrity["errors"]:
            print(f"    ⚠️ {e}")
    else:
        print("    ✅ Splits are 100% disjoint (no image appears in more than one split).")
        print("    ✅ Mask paths exist and are empty only when is_anomaly == 0.")
        print("    ✅ All image and mask spatial dimensions match perfectly.")

    print("\n📊 Counts Per Split:")
    for split in ("train", "val", "test"):
        count = len(df[df["project_split"] == split])
        normal = len(df[(df["project_split"] == split) & (df["is_anomaly"] == 0)])
        anom = len(df[(df["project_split"] == split) & (df["is_anomaly"] == 1)])
        print(f"  • {split.ljust(6)} : {str(count).rjust(3)} samples (Normal: {str(normal).rjust(2)}, Defective: {str(anom).rjust(2)})")

    print("\n📦 Defect Distribution Across Splits:")
    pivot = df.pivot_table(index="defect_type", columns="project_split", values="image_path", aggfunc="count", fill_value=0)
    print(pivot.to_string())

    print(f"\n💾 Manifest CSV exported: {integrity['manifest_files']['all']}")
    print("=" * 65)


def main():
    parser = argparse.ArgumentParser(description="MVTec AD Protocol B Manifest Builder")
    parser.add_argument("--data-dir", default=None, help="Root path to MVTec AD or category folder")
    parser.add_argument("--category", default="bottle", help="Target category (default: bottle)")
    parser.add_argument("--output-dir", default=None, help="Output folder for manifest CSVs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible split")
    args = parser.parse_args()

    cat_path = find_dataset_root(custom_path=args.data_dir, category=args.category)
    out_dir = Path(args.output_dir) if args.output_dir else None
    df, report = build_protocol_b_manifest(cat_path, out_dir, seed=args.seed)
    if not report["passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
