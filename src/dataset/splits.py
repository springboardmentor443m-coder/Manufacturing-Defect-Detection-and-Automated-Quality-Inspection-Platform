"""Dataset splitting architecture and protocols for VisionInspect AI.

Implements Protocol A (Official Normal-Only Anomaly Detection for PatchCore) and
Protocol B (Custom Supervised Segmentation Comparison for U-Net) with strict
data leakage prevention, rare defect subtype handling, and deterministic seeding.
"""

from __future__ import annotations

import csv
import json
import logging
import random
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from .manifest import ManifestRecord

logger = logging.getLogger(__name__)


class SplitProtocol(str, Enum):
    """Supported data evaluation protocols."""

    PROTOCOL_A_PATCHCORE = "protocol_a_patchcore"
    PROTOCOL_B_SUPERVISED_UNET = "protocol_b_supervised_unet"


class DataLeakageError(Exception):
    """Raised when an identical image appears in more than one dataset split."""
    pass


class InvalidSplitConfigError(Exception):
    """Raised when requested split ratios or sample allocations are mathematically invalid."""
    pass


@dataclass(frozen=True)
class SplitConfig:
    """Hyperparameters and configuration for dataset splitting."""

    protocol: SplitProtocol = SplitProtocol.PROTOCOL_A_PATCHCORE
    train_ratio: float = 0.80
    val_ratio: float = 0.20
    test_ratio: float = 0.00         # For Protocol A, test is already isolated in test/ folder
    seed: int = 42
    stratify_by_subtype: bool = True
    split_version: str = "1.0.0"

    def __post_init__(self) -> None:
        """Validates numerical integrity of split ratios upon initialization."""
        if self.protocol == SplitProtocol.PROTOCOL_A_PATCHCORE:
            total = self.train_ratio + self.val_ratio
            if abs(total - 1.0) > 1e-4:
                raise InvalidSplitConfigError(
                    f"Protocol A requires train_ratio ({self.train_ratio}) + val_ratio ({self.val_ratio}) == 1.0"
                )
        elif self.protocol == SplitProtocol.PROTOCOL_B_SUPERVISED_UNET:
            total = self.train_ratio + self.val_ratio + self.test_ratio
            if abs(total - 1.0) > 1e-4:
                raise InvalidSplitConfigError(
                    f"Protocol B requires train_ratio + val_ratio + test_ratio == 1.0 (got {total:.4f})"
                )

        if self.train_ratio <= 0.0:
            raise InvalidSplitConfigError("train_ratio must be strictly positive (> 0)")


@dataclass
class SplitResult:
    """Container for split manifest records, metadata, and audit summary."""

    train_records: List[ManifestRecord] = field(default_factory=list)
    val_records: List[ManifestRecord] = field(default_factory=list)
    test_records: List[ManifestRecord] = field(default_factory=list)
    protocol: str = ""
    seed: int = 42
    summary: Dict[str, Any] = field(default_factory=dict)

    def verify_no_leakage(self) -> None:
        """Verifies that train, val, and test partitions are strictly disjoint."""
        train_paths = {r.image_path for r in self.train_records}
        val_paths = {r.image_path for r in self.val_records}
        test_paths = {r.image_path for r in self.test_records}

        train_val = train_paths & val_paths
        if train_val:
            raise DataLeakageError(f"Data leakage detected between TRAIN and VAL ({len(train_val)} overlapping images)!")

        train_test = train_paths & test_paths
        if train_test:
            raise DataLeakageError(f"Data leakage detected between TRAIN and TEST ({len(train_test)} overlapping images)!")

        val_test = val_paths & test_paths
        if val_test:
            raise DataLeakageError(f"Data leakage detected between VAL and TEST ({len(val_test)} overlapping images)!")

    def export_manifests(self, output_dir: Path) -> None:
        """Exports separated JSON and CSV manifests for each split."""
        output_dir.mkdir(parents=True, exist_ok=True)

        splits = [
            ("train", self.train_records),
            ("val", self.val_records),
            ("test", self.test_records),
        ]

        for split_name, records in splits:
            json_path = output_dir / f"{split_name}_manifest.json"
            csv_path = output_dir / f"{split_name}_manifest.csv"

            payload = {
                "split_name": split_name,
                "protocol": self.protocol,
                "seed": self.seed,
                "total_records": len(records),
                "records": [asdict(r) for r in records],
            }
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)

            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                if records:
                    writer = csv.DictWriter(f, fieldnames=list(asdict(records[0]).keys()))
                    writer.writeheader()
                    for r in records:
                        writer.writerow(asdict(r))
                else:
                    writer = csv.writer(f)
                    writer.writerow(["image_path", "category", "split", "defect_type"])

            logger.info("Exported %s manifest to %s (%d records)", split_name, json_path, len(records))


class DatasetSplitter:
    """Executes deterministic, leakage-safe dataset splitting according to formal protocols."""

    def __init__(self, config: Optional[SplitConfig] = None) -> None:
        self.config = config or SplitConfig()
        self._validate_config(self.config)

    @staticmethod
    def _validate_config(config: SplitConfig) -> None:
        """Validates numerical integrity of split ratios."""
        if config.protocol == SplitProtocol.PROTOCOL_A_PATCHCORE:
            # Protocol A splits normal training images into train and val (calibration)
            total = config.train_ratio + config.val_ratio
            if abs(total - 1.0) > 1e-4:
                raise InvalidSplitConfigError(
                    f"Protocol A requires train_ratio ({config.train_ratio}) + val_ratio ({config.val_ratio}) == 1.0"
                )
        elif config.protocol == SplitProtocol.PROTOCOL_B_SUPERVISED_UNET:
            total = config.train_ratio + config.val_ratio + config.test_ratio
            if abs(total - 1.0) > 1e-4:
                raise InvalidSplitConfigError(
                    f"Protocol B requires train_ratio + val_ratio + test_ratio == 1.0 (got {total:.4f})"
                )

        if config.train_ratio <= 0.0:
            raise InvalidSplitConfigError("train_ratio must be strictly positive (> 0)")

    def split(self, inventory_records: List[ManifestRecord]) -> SplitResult:
        """Executes splitting logic according to the configured protocol."""
        if not inventory_records:
            raise ValueError("Cannot split an empty list of inventory records!")

        if self.config.protocol == SplitProtocol.PROTOCOL_A_PATCHCORE:
            result = self._split_protocol_a(inventory_records)
        elif self.config.protocol == SplitProtocol.PROTOCOL_B_SUPERVISED_UNET:
            result = self._split_protocol_b(inventory_records)
        else:
            raise ValueError(f"Unknown split protocol: {self.config.protocol}")

        result.verify_no_leakage()
        return result

    def _split_protocol_a(self, records: List[ManifestRecord]) -> SplitResult:
        """Protocol A: Normal-Only Anomaly Detection for PatchCore.

        - Normal training images (split=='train') partitioned into:
            * PatchCore Train (e.g. 80%)
            * PatchCore Val Calibration (e.g. 20%)
        - Defective images NEVER enter train or val!
        - All test folder images (normal test + defective test) are held out for final test.
        """
        rng = random.Random(self.config.seed)
        result = SplitResult(protocol=self.config.protocol.value, seed=self.config.seed)

        # Segregate by original directory partition
        train_normal_by_cat: Dict[str, List[ManifestRecord]] = defaultdict(list)
        test_records: List[ManifestRecord] = []

        for r in records:
            if r.split == "train":
                if r.is_anomalous:
                    raise DataLeakageError(
                        f"Protocol A violation: Defective image found in train split: {r.image_path}"
                    )
                train_normal_by_cat[r.category].append(r)
            else:
                test_records.append(r)

        train_records: List[ManifestRecord] = []
        val_records: List[ManifestRecord] = []

        for cat, cat_train_records in sorted(train_normal_by_cat.items()):
            shuffled = list(cat_train_records)
            rng.shuffle(shuffled)

            n_total = len(shuffled)
            if n_total < 2 and self.config.val_ratio > 0:
                raise InvalidSplitConfigError(
                    f"Category '{cat}' has only {n_total} normal train samples; cannot split into train and val!"
                )

            n_train = max(1, int(round(n_total * self.config.train_ratio)))
            # Ensure val is not empty if val_ratio > 0 and samples allow
            if self.config.val_ratio > 0 and n_train == n_total and n_total > 1:
                n_train -= 1

            cat_train = shuffled[:n_train]
            cat_val = shuffled[n_train:]

            # Assign split tags
            for rec in cat_train:
                train_records.append(
                    ManifestRecord(
                        image_path=rec.image_path,
                        mask_path=rec.mask_path,
                        category=rec.category,
                        split="train",
                        defect_type=rec.defect_type,
                        is_anomalous=rec.is_anomalous,
                        image_width=rec.image_width,
                        image_height=rec.image_height,
                        image_format=rec.image_format,
                        file_size_bytes=rec.file_size_bytes,
                        image_sha256=rec.image_sha256,
                        validation_status=rec.validation_status,
                    )
                )

            for rec in cat_val:
                val_records.append(
                    ManifestRecord(
                        image_path=rec.image_path,
                        mask_path=rec.mask_path,
                        category=rec.category,
                        split="val",
                        defect_type=rec.defect_type,
                        is_anomalous=rec.is_anomalous,
                        image_width=rec.image_width,
                        image_height=rec.image_height,
                        image_format=rec.image_format,
                        file_size_bytes=rec.file_size_bytes,
                        image_sha256=rec.image_sha256,
                        validation_status=rec.validation_status,
                    )
                )

        result.train_records = train_records
        result.val_records = val_records
        result.test_records = test_records

        result.summary = {
            "protocol": "Protocol A (PatchCore Normal-Only)",
            "total_train_normal": len(train_records),
            "total_val_normal": len(val_records),
            "total_test_held_out": len(test_records),
            "anomalous_in_train": 0,
            "anomalous_in_val": 0,
            "anomalous_in_test": sum(1 for r in test_records if r.is_anomalous),
        }
        return result

    def _split_protocol_b(self, records: List[ManifestRecord]) -> SplitResult:
        """Protocol B: Custom Supervised Segmentation for U-Net.

        - Normal images partitioned into train/val/test according to ratios.
        - Defective images stratified by category and defect subtype into train/val/test.
        - Rare defect subtypes handled safely without leaving empty training splits.
        """
        rng = random.Random(self.config.seed)
        result = SplitResult(protocol=self.config.protocol.value, seed=self.config.seed)

        # Group records by category and (defect_type, is_anomalous)
        grouped: Dict[str, Dict[str, List[ManifestRecord]]] = defaultdict(lambda: defaultdict(list))
        for r in records:
            grouped[r.category][r.defect_type].append(r)

        train_records: List[ManifestRecord] = []
        val_records: List[ManifestRecord] = []
        test_records: List[ManifestRecord] = []

        for cat, subtypes in sorted(grouped.items()):
            for defect_type, items in sorted(subtypes.items()):
                shuffled = list(items)
                rng.shuffle(shuffled)
                n = len(shuffled)

                # Rare subtype handling
                if n < 3:
                    logger.warning(
                        "Rare subtype detected: '%s/%s' has only %d samples. Allocating safely.",
                        cat, defect_type, n
                    )
                    # For extremely small samples, allocate to train and test, ensuring no empty crash
                    if n == 1:
                        # Single sample placed into train so model can learn this defect
                        cat_train = shuffled
                        cat_val = []
                        cat_test = []
                    elif n == 2:
                        cat_train = [shuffled[0]]
                        cat_val = []
                        cat_test = [shuffled[1]]
                else:
                    n_train = max(1, int(round(n * self.config.train_ratio)))
                    n_val = int(round(n * self.config.val_ratio))
                    if self.config.val_ratio > 0 and n_val == 0 and n >= 3:
                        n_val = 1
                    # Remaining to test
                    n_test = max(0, n - n_train - n_val)
                    if self.config.test_ratio > 0 and n_test == 0 and n >= 3:
                        if n_train > 1:
                            n_train -= 1
                            n_test = 1

                    cat_train = shuffled[:n_train]
                    cat_val = shuffled[n_train:n_train + n_val]
                    cat_test = shuffled[n_train + n_val:]

                for rec in cat_train:
                    train_records.append(self._clone_with_split(rec, "train"))
                for rec in cat_val:
                    val_records.append(self._clone_with_split(rec, "val"))
                for rec in cat_test:
                    test_records.append(self._clone_with_split(rec, "test"))

        result.train_records = train_records
        result.val_records = val_records
        result.test_records = test_records

        result.summary = {
            "protocol": "Protocol B (Supervised U-Net Custom Split)",
            "total_train": len(train_records),
            "train_anomalous": sum(1 for r in train_records if r.is_anomalous),
            "train_normal": sum(1 for r in train_records if not r.is_anomalous),
            "total_val": len(val_records),
            "val_anomalous": sum(1 for r in val_records if r.is_anomalous),
            "val_normal": sum(1 for r in val_records if not r.is_anomalous),
            "total_test": len(test_records),
            "test_anomalous": sum(1 for r in test_records if r.is_anomalous),
            "test_normal": sum(1 for r in test_records if not r.is_anomalous),
        }
        return result

    @staticmethod
    def _clone_with_split(record: ManifestRecord, new_split: str) -> ManifestRecord:
        """Returns a copy of a ManifestRecord with an updated split attribute."""
        return ManifestRecord(
            image_path=record.image_path,
            mask_path=record.mask_path,
            category=record.category,
            split=new_split,
            defect_type=record.defect_type,
            is_anomalous=record.is_anomalous,
            image_width=record.image_width,
            image_height=record.image_height,
            image_format=record.image_format,
            file_size_bytes=record.file_size_bytes,
            image_sha256=record.image_sha256,
            validation_status=record.validation_status,
        )
