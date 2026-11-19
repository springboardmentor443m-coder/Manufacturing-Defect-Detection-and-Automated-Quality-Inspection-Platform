"""Dataset Validator for VisionInspect AI.

Performs comprehensive, non-destructive validation of MVTec AD directory structures,
image-mask correspondence, dimensions, pixel values, duplicates, and integrity.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
UNEXPECTED_FILE_NAMES = {".ds_store", "thumbs.db", "desktop.ini", ".gitkeep"}


@dataclass
class ValidationIssue:
    """Represents a specific structural or integrity issue discovered in the dataset."""

    severity: str  # "ERROR" or "WARNING"
    category: str
    issue_type: str
    relative_path: str
    message: str


@dataclass
class ValidationReport:
    """Aggregated validation findings and metrics across all scanned categories."""

    is_valid: bool = True
    scanned_categories: List[str] = field(default_factory=list)
    total_images_scanned: int = 0
    total_masks_scanned: int = 0
    error_count: int = 0
    warning_count: int = 0
    issues: List[ValidationIssue] = field(default_factory=list)
    category_summaries: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def add_issue(
        self,
        severity: str,
        category: str,
        issue_type: str,
        relative_path: str,
        message: str,
    ) -> None:
        """Adds an issue and updates aggregate status."""
        severity_upper = severity.upper()
        if severity_upper == "ERROR":
            self.is_valid = False
            self.error_count += 1
        elif severity_upper == "WARNING":
            self.warning_count += 1

        self.issues.append(
            ValidationIssue(
                severity=severity_upper,
                category=category,
                issue_type=issue_type,
                relative_path=str(relative_path).replace("\\", "/"),
                message=message,
            )
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serializes report into a nested dictionary."""
        return {
            "is_valid": self.is_valid,
            "scanned_categories": self.scanned_categories,
            "total_images_scanned": self.total_images_scanned,
            "total_masks_scanned": self.total_masks_scanned,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "category_summaries": self.category_summaries,
            "issues": [asdict(issue) for issue in self.issues],
        }

    def save_json(self, output_path: Path) -> None:
        """Exports report to formatted JSON file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info("Saved validation JSON report to %s", output_path)

    def save_csv(self, output_path: Path) -> None:
        """Exports all detected issues to a flat CSV file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["severity", "category", "issue_type", "relative_path", "message"])
            for issue in self.issues:
                writer.writerow(
                    [
                        issue.severity,
                        issue.category,
                        issue.issue_type,
                        issue.relative_path,
                        issue.message,
                    ]
                )
        logger.info("Saved validation CSV report to %s", output_path)


class DatasetValidator:
    """Inspects and validates MVTec AD dataset directory layouts and assets."""

    def __init__(
        self,
        dataset_root: Path,
        target_categories: Optional[List[str]] = None,
        check_hashes: bool = True,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.target_categories = target_categories
        self.check_hashes = check_hashes

    @staticmethod
    def calculate_file_hash(file_path: Path) -> str:
        """Computes SHA-256 hash of a file."""
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        return hasher.hexdigest()

    def validate(self) -> ValidationReport:
        """Executes full validation suite across target categories."""
        report = ValidationReport()

        if not self.dataset_root.exists():
            report.add_issue(
                severity="ERROR",
                category="root",
                issue_type="MISSING_DATASET_ROOT",
                relative_path="",
                message=f"Dataset root directory does not exist: {self.dataset_root}",
            )
            return report

        if not self.dataset_root.is_dir():
            report.add_issue(
                severity="ERROR",
                category="root",
                issue_type="INVALID_DATASET_ROOT",
                relative_path="",
                message=f"Dataset root is not a directory: {self.dataset_root}",
            )
            return report

        available_categories = [
            d.name for d in self.dataset_root.iterdir() if d.is_dir() and not d.name.startswith(".")
        ]

        if not available_categories:
            report.add_issue(
                severity="ERROR",
                category="root",
                issue_type="EMPTY_DATASET_ROOT",
                relative_path="",
                message=f"No category directories found under {self.dataset_root}",
            )
            return report

        categories_to_scan = self.target_categories if self.target_categories else sorted(available_categories)

        for category_name in categories_to_scan:
            category_dir = self.dataset_root / category_name
            if not category_dir.is_dir():
                report.add_issue(
                    severity="ERROR",
                    category=category_name,
                    issue_type="MISSING_CATEGORY_DIRECTORY",
                    relative_path=category_name,
                    message=f"Category directory does not exist: {category_dir}",
                )
                continue

            report.scanned_categories.append(category_name)
            self._validate_category(category_name, category_dir, report)

        return report

    def _validate_category(
        self,
        category: str,
        category_dir: Path,
        report: ValidationReport,
    ) -> None:
        """Validates internal directory layout, images, and masks for a single category."""
        summary: Dict[str, Any] = {
            "train_good_count": 0,
            "test_good_count": 0,
            "test_defect_count": 0,
            "mask_count": 0,
            "defect_subtypes": [],
        }

        train_dir = category_dir / "train"
        test_dir = category_dir / "test"
        gt_dir = category_dir / "ground_truth"

        # Check required top-level folders
        if not train_dir.is_dir():
            report.add_issue("ERROR", category, "MISSING_TRAIN_DIR", f"{category}/train", "Missing 'train' directory")
        if not test_dir.is_dir():
            report.add_issue("ERROR", category, "MISSING_TEST_DIR", f"{category}/test", "Missing 'test' directory")

        # Scan for unexpected files at category root
        for item in category_dir.iterdir():
            if item.is_file():
                report.add_issue(
                    "WARNING",
                    category,
                    "UNEXPECTED_FILE",
                    f"{category}/{item.name}",
                    f"Unexpected file at category root: {item.name}",
                )

        image_hashes: Dict[str, str] = {}  # hash -> relative_path

        # Validate train folder
        if train_dir.is_dir():
            train_good_dir = train_dir / "good"
            if not train_good_dir.is_dir():
                report.add_issue(
                    "ERROR",
                    category,
                    "MISSING_TRAIN_GOOD_DIR",
                    f"{category}/train/good",
                    "Missing 'train/good' directory",
                )
            else:
                train_good_images = self._scan_images(train_good_dir, category, report, is_mask=False)
                summary["train_good_count"] = len(train_good_images)
                report.total_images_scanned += len(train_good_images)
                if len(train_good_images) == 0:
                    report.add_issue(
                        "ERROR",
                        category,
                        "EMPTY_IMAGE_DIRECTORY",
                        f"{category}/train/good",
                        "Train good directory contains no images",
                    )
                self._check_duplicates(train_good_images, image_hashes, category, report)

            # Check for non-good dirs in train
            for sub in train_dir.iterdir():
                if sub.is_dir() and sub.name != "good":
                    report.add_issue(
                        "WARNING",
                        category,
                        "UNEXPECTED_TRAIN_SUBDIR",
                        f"{category}/train/{sub.name}",
                        f"Unexpected subdirectory in train: {sub.name}. MVTec AD train should contain only 'good'.",
                    )

        # Validate test and ground_truth folders
        if test_dir.is_dir():
            test_subdirs = [d for d in test_dir.iterdir() if d.is_dir()]
            defect_subtypes = [d.name for d in test_subdirs if d.name != "good"]
            summary["defect_subtypes"] = sorted(defect_subtypes)

            test_good_dir = test_dir / "good"
            if not test_good_dir.is_dir():
                report.add_issue(
                    "WARNING",
                    category,
                    "MISSING_TEST_GOOD_DIR",
                    f"{category}/test/good",
                    "Missing 'test/good' directory",
                )
            else:
                test_good_images = self._scan_images(test_good_dir, category, report, is_mask=False)
                summary["test_good_count"] = len(test_good_images)
                report.total_images_scanned += len(test_good_images)
                self._check_duplicates(test_good_images, image_hashes, category, report)

            # Ground truth directory check
            if defect_subtypes and not gt_dir.is_dir():
                report.add_issue(
                    "ERROR",
                    category,
                    "MISSING_GROUND_TRUTH_DIR",
                    f"{category}/ground_truth",
                    "Defective test images present, but 'ground_truth' directory is missing",
                )

            # Track all defect test images and corresponding masks
            for defect_name in defect_subtypes:
                defect_test_dir = test_dir / defect_name
                defect_images = self._scan_images(defect_test_dir, category, report, is_mask=False)
                summary["test_defect_count"] += len(defect_images)
                report.total_images_scanned += len(defect_images)
                self._check_duplicates(defect_images, image_hashes, category, report)

                defect_gt_dir = gt_dir / defect_name if gt_dir.is_dir() else None

                if defect_gt_dir is None or not defect_gt_dir.is_dir():
                    report.add_issue(
                        "ERROR",
                        category,
                        "MISSING_DEFECT_GT_DIR",
                        f"{category}/ground_truth/{defect_name}",
                        f"Missing ground_truth directory for defect type '{defect_name}'",
                    )
                    continue

                gt_masks = self._scan_images(defect_gt_dir, category, report, is_mask=True)
                summary["mask_count"] += len(gt_masks)
                report.total_masks_scanned += len(gt_masks)

                # Pair images with masks
                self._verify_image_mask_pairing(
                    category=category,
                    defect_name=defect_name,
                    defect_images=defect_images,
                    gt_masks=gt_masks,
                    report=report,
                )

            # Check for unmatched mask subdirectories in ground_truth
            if gt_dir.is_dir():
                for gt_sub in gt_dir.iterdir():
                    if gt_sub.is_dir() and gt_sub.name not in defect_subtypes:
                        report.add_issue(
                            "WARNING",
                            category,
                            "UNMATCHED_GT_SUBDIR",
                            f"{category}/ground_truth/{gt_sub.name}",
                            f"Ground truth subdirectory '{gt_sub.name}' has no matching defective test subdirectory",
                        )

        report.category_summaries[category] = summary

    def _scan_images(
        self,
        folder: Path,
        category: str,
        report: ValidationReport,
        is_mask: bool = False,
    ) -> Dict[str, Path]:
        """Scans a directory for images, verifying readability and headers."""
        image_map: Dict[str, Path] = {}

        for item in folder.iterdir():
            if item.name.lower() in UNEXPECTED_FILE_NAMES or item.name.startswith("."):
                report.add_issue(
                    "WARNING",
                    category,
                    "UNEXPECTED_SYSTEM_FILE",
                    f"{category}/{item.relative_to(self.dataset_root / category)}",
                    f"Unexpected metadata or hidden file found: {item.name}",
                )
                continue

            if item.is_dir():
                report.add_issue(
                    "WARNING",
                    category,
                    "UNEXPECTED_NESTED_DIRECTORY",
                    f"{category}/{item.relative_to(self.dataset_root / category)}",
                    f"Unexpected nested directory inside image folder: {item.name}",
                )
                continue

            ext = item.suffix.lower()
            if ext not in SUPPORTED_IMAGE_EXTENSIONS:
                report.add_issue(
                    "WARNING",
                    category,
                    "UNSUPPORTED_IMAGE_EXTENSION",
                    f"{category}/{item.relative_to(self.dataset_root / category)}",
                    f"Unsupported image file extension: {ext}",
                )
                continue

            # Verify integrity
            try:
                with Image.open(item) as img:
                    img.verify()
                # Reopen to verify loading pixel buffer
                with Image.open(item) as img:
                    _ = img.size
                    w, h = img.size
                    if w < 16 or h < 16:
                        report.add_issue(
                            "ERROR",
                            category,
                            "TINY_IMAGE_DIMENSIONS",
                            f"{category}/{item.relative_to(self.dataset_root / category)}",
                            f"Image dimensions ({w}x{h}) are abnormally small",
                        )
                image_map[item.stem] = item
            except Exception as e:
                report.add_issue(
                    "ERROR",
                    category,
                    "CORRUPTED_IMAGE",
                    f"{category}/{item.relative_to(self.dataset_root / category)}",
                    f"Corrupted or unreadable image file: {str(e)}",
                )

        return image_map

    def _verify_image_mask_pairing(
        self,
        category: str,
        defect_name: str,
        defect_images: Dict[str, Path],
        gt_masks: Dict[str, Path],
        report: ValidationReport,
    ) -> None:
        """Verifies exact 1:1 image-to-mask correspondence, dimensions, and pixel value validity."""
        # Check every defect image has a mask
        for img_stem, img_path in defect_images.items():
            expected_mask_stem_1 = f"{img_stem}_mask"
            expected_mask_stem_2 = img_stem

            mask_path = gt_masks.get(expected_mask_stem_1) or gt_masks.get(expected_mask_stem_2)

            if not mask_path:
                report.add_issue(
                    "ERROR",
                    category,
                    "MISSING_GROUND_TRUTH_MASK",
                    f"{category}/test/{defect_name}/{img_path.name}",
                    f"No matching ground truth mask found for test image {img_path.name}",
                )
                continue

            # Verify dimensions and binary pixel values
            try:
                with Image.open(img_path) as img, Image.open(mask_path) as mask:
                    if img.size != mask.size:
                        report.add_issue(
                            "ERROR",
                            category,
                            "DIMENSION_MISMATCH",
                            f"{category}/test/{defect_name}/{img_path.name}",
                            f"Dimension mismatch between image ({img.size}) and mask ({mask.size})",
                        )

                    # Validate mask pixel values
                    mask_arr = np.array(mask)
                    unique_vals = set(np.unique(mask_arr).tolist())
                    valid_sets = [{0}, {255}, {0, 255}, {1}, {0, 1}]
                    if not any(unique_vals.issubset(s) for s in valid_sets):
                        report.add_issue(
                            "ERROR",
                            category,
                            "INVALID_MASK_VALUES",
                            f"{category}/ground_truth/{defect_name}/{mask_path.name}",
                            f"Mask contains non-binary pixel values: {sorted(list(unique_vals))[:10]}",
                        )
            except Exception as e:
                report.add_issue(
                    "ERROR",
                    category,
                    "MASK_INSPECTION_FAILURE",
                    f"{category}/ground_truth/{defect_name}/{mask_path.name}",
                    f"Failed to inspect mask pixel values: {str(e)}",
                )

        # Check for orphan / unmatched masks
        for mask_stem, mask_path in gt_masks.items():
            # Strip trailing _mask if present
            base_stem = mask_stem[:-5] if mask_stem.endswith("_mask") else mask_stem
            if base_stem not in defect_images and mask_stem not in defect_images:
                report.add_issue(
                    "WARNING",
                    category,
                    "UNMATCHED_GROUND_TRUTH_MASK",
                    f"{category}/ground_truth/{defect_name}/{mask_path.name}",
                    f"Ground truth mask {mask_path.name} has no matching test image in test/{defect_name}",
                )

    def _check_duplicates(
        self,
        images: Dict[str, Path],
        hash_registry: Dict[str, str],
        category: str,
        report: ValidationReport,
    ) -> None:
        """Detects identical duplicate image files via SHA-256."""
        if not self.check_hashes:
            return

        for img_path in images.values():
            try:
                file_hash = self.calculate_file_hash(img_path)
                rel_path = str(img_path.relative_to(self.dataset_root)).replace("\\", "/")
                if file_hash in hash_registry:
                    report.add_issue(
                        "WARNING",
                        category,
                        "DUPLICATE_IMAGE_CONTENT",
                        rel_path,
                        f"Exact duplicate image content of: {hash_registry[file_hash]}",
                    )
                else:
                    hash_registry[file_hash] = rel_path
            except Exception as e:
                logger.warning("Could not compute hash for %s: %s", img_path, e)
