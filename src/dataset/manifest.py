"""Preliminary Dataset Inventory Manifest Builder for VisionInspect AI.

Generates a structured, reproducible inventory of all raw images, ground-truth masks,
metadata, and cryptographic checksums without assigning training splits.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image

logger = logging.getLogger(__name__)

SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


@dataclass
class ManifestRecord:
    """Represents a single verified image sample in the preliminary dataset inventory."""

    image_path: str                 # Relative POSIX path from dataset root
    mask_path: Optional[str]        # Relative POSIX path from dataset root, or None
    category: str                   # e.g., "bottle", "metal_nut", "tile"
    split: str                      # "train" or "test" (as per original directory structure)
    defect_type: str                # e.g., "good", "broken_large", "scratch"
    is_anomalous: bool              # True for defects, False for good/normal
    image_width: int
    image_height: int
    image_format: str               # e.g., "PNG", "JPEG"
    file_size_bytes: int
    image_sha256: str
    validation_status: str          # "VALID", "WARNING", "ERROR"


class DatasetManifestBuilder:
    """Builds and serializes the preliminary inventory manifest for the raw dataset."""

    def __init__(self, dataset_root: Path, target_categories: Optional[List[str]] = None) -> None:
        self.dataset_root = Path(dataset_root)
        self.target_categories = target_categories

    @staticmethod
    def calculate_file_hash(file_path: Path) -> str:
        """Computes SHA-256 hash of a file."""
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        return hasher.hexdigest()

    def build_inventory(self) -> List[ManifestRecord]:
        """Scans the dataset directory and constructs the preliminary inventory records."""
        records: List[ManifestRecord] = []

        if not self.dataset_root.is_dir():
            logger.error("Dataset root does not exist: %s", self.dataset_root)
            return records

        available_categories = [
            d.name for d in self.dataset_root.iterdir() if d.is_dir() and not d.name.startswith(".")
        ]
        categories_to_scan = self.target_categories if self.target_categories else sorted(available_categories)

        for category in categories_to_scan:
            category_dir = self.dataset_root / category
            if not category_dir.is_dir():
                continue

            # Process train images (all normal/good in standard MVTec AD)
            train_good_dir = category_dir / "train" / "good"
            if train_good_dir.is_dir():
                for img_path in sorted(train_good_dir.iterdir()):
                    if img_path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
                        rec = self._create_record(
                            img_path=img_path,
                            mask_path=None,
                            category=category,
                            split="train",
                            defect_type="good",
                            is_anomalous=False,
                        )
                        if rec:
                            records.append(rec)

            # Process test images (normal + defect subtypes)
            test_dir = category_dir / "test"
            gt_dir = category_dir / "ground_truth"

            if test_dir.is_dir():
                for defect_folder in sorted(test_dir.iterdir()):
                    if not defect_folder.is_dir() or defect_folder.name.startswith("."):
                        continue

                    defect_type = defect_folder.name
                    is_anomalous = defect_type != "good"

                    for img_path in sorted(defect_folder.iterdir()):
                        if img_path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
                            continue

                        # Resolve mask if anomalous
                        mask_path = None
                        if is_anomalous and gt_dir.is_dir():
                            # MVTec AD masks are typically named <name>_mask.png or <name>.png
                            cand_1 = gt_dir / defect_type / f"{img_path.stem}_mask.png"
                            cand_2 = gt_dir / defect_type / f"{img_path.stem}.png"
                            if cand_1.is_file():
                                mask_path = cand_1
                            elif cand_2.is_file():
                                mask_path = cand_2

                        rec = self._create_record(
                            img_path=img_path,
                            mask_path=mask_path,
                            category=category,
                            split="test",
                            defect_type=defect_type,
                            is_anomalous=is_anomalous,
                        )
                        if rec:
                            records.append(rec)

        return records

    def _create_record(
        self,
        img_path: Path,
        mask_path: Optional[Path],
        category: str,
        split: str,
        defect_type: str,
        is_anomalous: bool,
    ) -> Optional[ManifestRecord]:
        """Constructs a single ManifestRecord with file attributes and checksums."""
        try:
            with Image.open(img_path) as img:
                w, h = img.size
                img_format = img.format or img_path.suffix[1:].upper()

            file_size = img_path.stat().st_size
            sha256 = self.calculate_file_hash(img_path)

            rel_img_path = str(img_path.relative_to(self.dataset_root)).replace("\\", "/")
            rel_mask_path = (
                str(mask_path.relative_to(self.dataset_root)).replace("\\", "/") if mask_path else None
            )

            val_status = "VALID"
            if is_anomalous and mask_path is None:
                val_status = "ERROR"

            return ManifestRecord(
                image_path=rel_img_path,
                mask_path=rel_mask_path,
                category=category,
                split=split,
                defect_type=defect_type,
                is_anomalous=is_anomalous,
                image_width=w,
                image_height=h,
                image_format=img_format,
                file_size_bytes=file_size,
                image_sha256=sha256,
                validation_status=val_status,
            )
        except Exception as e:
            logger.error("Failed to process image %s: %s", img_path, e)
            return None

    def export_json(self, records: List[ManifestRecord], output_path: Path) -> None:
        """Serializes inventory manifest records to JSON."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "manifest_version": "1.0.0",
            "phase": "Phase 2 - Dataset Inventory",
            "total_records": len(records),
            "records": [asdict(r) for r in records],
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.info("Saved dataset inventory JSON to %s (%d records)", output_path, len(records))

    def export_csv(self, records: List[ManifestRecord], output_path: Path) -> None:
        """Serializes inventory manifest records to CSV."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            if not records:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "image_path",
                        "mask_path",
                        "category",
                        "split",
                        "defect_type",
                        "is_anomalous",
                        "image_width",
                        "image_height",
                        "image_format",
                        "file_size_bytes",
                        "image_sha256",
                        "validation_status",
                    ]
                )
                return

            fieldnames = list(asdict(records[0]).keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in records:
                writer.writerow(asdict(r))
        logger.info("Saved dataset inventory CSV to %s (%d records)", output_path, len(records))
