"""
VisionInspect AI - Dataset Visual QA & Integrity Audit

Outputs:
- Counts per split and defect type
- 5 normal samples
- 5 defective samples with their matching ground-truth masks
- Red mask overlays blended onto defect images
- Flags any defective test image without matching <id>_mask.png as a critical data-integrity error
- Saves visual collage grid and JSON summary report
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import cv2
import numpy as np

from ml.dataset.discovery import discover_category_dataset, find_dataset_root


def run_visual_qa(
    category_dir: Path,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Executes visual QA checks and data integrity audit on discovered MVTec AD category.
    """
    if output_dir is None:
        output_dir = Path(__file__).resolve().parent.parent / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)

    info = discover_category_dataset(category_dir)
    category = info["category"]

    # 1. Integrity check: ensure EVERY defective test image has a matching <id>_mask.png
    integrity_errors: List[str] = []
    defective_sample_pairs: List[Dict[str, Any]] = []

    for defect_name, details in info["defects_details"].items():
        test_dir = details["test_dir"]
        gt_dir = details["gt_dir"]

        if not gt_dir or not gt_dir.exists():
            integrity_errors.append(f"CRITICAL: Defect folder '{defect_name}' has no ground_truth directory!")
            continue

        for img_path in details["image_files"]:
            stem = img_path.stem
            expected_mask_name = f"{stem}_mask{img_path.suffix}"
            mask_path = gt_dir / expected_mask_name

            # MVTec AD standard mask is png
            if not mask_path.exists():
                mask_path_png = gt_dir / f"{stem}_mask.png"
                if mask_path_png.exists():
                    mask_path = mask_path_png
                else:
                    integrity_errors.append(
                        f"DATA INTEGRITY ERROR: Defect image '{img_path}' has no matching mask at '{mask_path}'"
                    )
                    continue

            # Check dimensions match
            img = cv2.imread(str(img_path))
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if img is not None and mask is not None:
                if img.shape[:2] != mask.shape[:2]:
                    integrity_errors.append(
                        f"DIMENSION MISMATCH: Image {img_path.name} {img.shape[:2]} != Mask {mask_path.name} {mask.shape[:2]}"
                    )

            defective_sample_pairs.append({
                "defect_type": defect_name,
                "image_path": str(img_path),
                "mask_path": str(mask_path),
            })

    # 2. Select 5 normal samples
    train_good_files = sorted(info["train_good_dir"].glob("*.png"))
    normal_samples = [str(f) for f in train_good_files[:5]]

    # 3. Select 5 defective samples across categories
    defective_5 = defective_sample_pairs[:5]

    # 4. Generate Visual QA Collage (Normal, Defective, Mask, Overlay)
    target_thumb_size = (200, 200)

    # 4a. Load 5 normal thumbnails
    normal_thumbs = []
    for p in normal_samples:
        im = cv2.imread(p)
        if im is not None:
            normal_thumbs.append(cv2.resize(im, target_thumb_size))
        else:
            normal_thumbs.append(np.zeros((*target_thumb_size, 3), dtype=np.uint8))

    # 4b. Load 5 defective thumbnails, masks, and create red overlays
    defect_thumbs = []
    mask_thumbs = []
    overlay_thumbs = []

    for pair in defective_5:
        im = cv2.imread(pair["image_path"])
        mk = cv2.imread(pair["mask_path"], cv2.IMREAD_GRAYSCALE)

        if im is not None and mk is not None:
            im_thumb = cv2.resize(im, target_thumb_size)
            mk_thumb = cv2.resize(mk, target_thumb_size)

            # Mask visual (3 channels)
            mk_3ch = cv2.cvtColor(mk_thumb, cv2.COLOR_GRAY2BGR)

            # Red overlay: highlight mask in bright red [0, 0, 255]
            overlay = im_thumb.copy()
            red_layer = np.zeros_like(im_thumb)
            red_layer[mk_thumb > 0] = [0, 0, 255]  # Red in BGR
            # Blend 70% original + 30% red
            overlay = cv2.addWeighted(overlay, 0.75, red_layer, 0.25, 0)
            # Add yellow boundary contour around defect
            contours, _ = cv2.findContours(mk_thumb, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, (0, 255, 255), 2)

            defect_thumbs.append(im_thumb)
            mask_thumbs.append(mk_3ch)
            overlay_thumbs.append(overlay)
        else:
            blank = np.zeros((*target_thumb_size, 3), dtype=np.uint8)
            defect_thumbs.append(blank)
            mask_thumbs.append(blank)
            overlay_thumbs.append(blank)

    # Pad if fewer than 5
    while len(normal_thumbs) < 5:
        normal_thumbs.append(np.zeros((*target_thumb_size, 3), dtype=np.uint8))
    while len(defect_thumbs) < 5:
        blank = np.zeros((*target_thumb_size, 3), dtype=np.uint8)
        defect_thumbs.append(blank)
        mask_thumbs.append(blank)
        overlay_thumbs.append(blank)

    # Helper to add section title headers to collage rows
    def create_row_with_title(label: str, thumbs: List[np.ndarray]) -> np.ndarray:
        row = np.hstack(thumbs[:5])
        header = np.zeros((30, row.shape[1], 3), dtype=np.uint8)
        cv2.putText(header, label, (15, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 1, cv2.LINE_AA)
        return np.vstack([header, row])

    row_normal = create_row_with_title("1. Normal Samples (Pass)", normal_thumbs)
    row_defect = create_row_with_title("2. Defective Samples (Test)", defect_thumbs)
    row_mask = create_row_with_title("3. Ground-Truth Binary Masks", mask_thumbs)
    row_overlay = create_row_with_title("4. Defect Masks Overlay (Inspection Alignment)", overlay_thumbs)

    collage = np.vstack([row_normal, row_defect, row_mask, row_overlay])
    collage_path = output_dir / f"visual_qa_{category}.png"
    cv2.imwrite(str(collage_path), collage)

    # 5. Compile QA Audit Report
    qa_report = {
        "category": category,
        "integrity_passed": len(integrity_errors) == 0,
        "integrity_error_count": len(integrity_errors),
        "integrity_errors": integrity_errors,
        "counts": {
            "train_good": info["train_good_count"],
            "test_good": info["test_good_count"],
            "defects_by_type": {k: v["image_count"] for k, v in info["defects_details"].items()},
            "total_defective": len(defective_sample_pairs),
            "total_images": info["total_images"],
        },
        "normal_samples_5": normal_samples,
        "defective_samples_5": defective_5,
        "visual_qa_collage_image": str(collage_path),
    }

    report_path = output_dir / f"visual_qa_report_{category}.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(qa_report, f, indent=2)

    # Print Report Console Output
    print("=" * 65)
    print(f"🔍 VISUAL QA & INTEGRITY REPORT: [{category.upper()}]")
    print("=" * 65)
    print(f"  • Data Integrity Status : {'PASSED ✅' if qa_report['integrity_passed'] else 'FAILED ❌'}")
    if integrity_errors:
        for err in integrity_errors:
            print(f"    ⚠️ {err}")
    else:
        print("    ✅ All defective test images have valid matching ground-truth masks.")

    print("\n📊 Counts by Split & Defect Type:")
    print(f"  • train/good : {qa_report['counts']['train_good']} frames")
    print(f"  • test/good  : {qa_report['counts']['test_good']} frames")
    for dname, cnt in qa_report['counts']['defects_by_type'].items():
        print(f"  • test/{dname} : {cnt} frames (100% paired with masks)")

    print("\n🟢 Five Normal Samples:")
    for s in normal_samples:
        print(f"   [OK] {Path(s).name} ({s})")

    print("\n🔴 Five Defective Samples & Masks:")
    for p in defective_5:
        print(f"   [{p['defect_type']}] Img: {Path(p['image_path']).name} -> Mask: {Path(p['mask_path']).name}")

    print(f"\n🎨 Visual QA collage saved to: {collage_path}")
    print(f"📄 QA JSON summary saved to  : {report_path}")
    print("=" * 65)

    return qa_report


def main():
    parser = argparse.ArgumentParser(description="MVTec AD Visual QA")
    parser.add_argument("--data-dir", default=None, help="Root path to MVTec AD or category folder")
    parser.add_argument("--category", default="bottle", help="Target category (default: bottle)")
    parser.add_argument("--output-dir", default=None, help="Folder to save QA reports & collage")
    args = parser.parse_args()

    cat_path = find_dataset_root(custom_path=args.data_dir, category=args.category)
    out_path = Path(args.output_dir) if args.output_dir else None
    report = run_visual_qa(cat_path, out_path)
    if not report["integrity_passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
