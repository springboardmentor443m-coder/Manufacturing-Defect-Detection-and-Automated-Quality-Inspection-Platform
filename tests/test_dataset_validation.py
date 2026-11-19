"""Automated test suite for VisionInspect AI dataset validation, manifest, and EDA utilities.

Uses synthetic in-memory image fixtures to test all edge cases deterministically
without requiring the full MVTec AD dataset.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple

import numpy as np
import pytest
from PIL import Image

from src.dataset.acquisition import LicensingGate
from src.dataset.eda import DatasetEDAAnalyzer
from src.dataset.manifest import DatasetManifestBuilder
from src.dataset.validator import DatasetValidator


def create_synthetic_image(path: Path, size: Tuple[int, int] = (64, 64), color: Tuple[int, int, int] = (128, 128, 128)) -> None:
    """Helper to create a valid synthetic RGB test image."""
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", size, color=color)
    img.save(path, format="PNG")


def create_synthetic_mask(path: Path, size: Tuple[int, int] = (64, 64), has_defect: bool = True, invalid_val: int = 0) -> None:
    """Helper to create a valid or invalid synthetic 1-channel ground-truth mask."""
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.zeros(size, dtype=np.uint8)
    if has_defect:
        arr[10:30, 10:30] = 255  # Square defect
    if invalid_val != 0:
        arr[40:50, 40:50] = invalid_val  # Invalid intermediate value (e.g. 128)
    mask = Image.fromarray(arr, mode="L")
    mask.save(path, format="PNG")


class TestDatasetValidation:
    """Tests the DatasetValidator against standard and anomalous directory configurations."""

    def test_empty_dataset_directory(self, tmp_path: Path) -> None:
        """Test detection of an empty dataset root directory."""
        validator = DatasetValidator(dataset_root=tmp_path)
        report = validator.validate()

        assert not report.is_valid
        assert report.error_count >= 1
        assert any(issue.issue_type == "EMPTY_DATASET_ROOT" for issue in report.issues)

    def test_missing_category_directory(self, tmp_path: Path) -> None:
        """Test reporting of a missing category requested explicitly."""
        (tmp_path / "bottle").mkdir()
        validator = DatasetValidator(dataset_root=tmp_path, target_categories=["tile"])
        report = validator.validate()

        assert not report.is_valid
        assert any(issue.issue_type == "MISSING_CATEGORY_DIRECTORY" for issue in report.issues)

    def test_valid_normal_image(self, tmp_path: Path) -> None:
        """Test validation of category with valid normal images in train and test."""
        cat_dir = tmp_path / "bottle"
        create_synthetic_image(cat_dir / "train" / "good" / "000.png")
        create_synthetic_image(cat_dir / "test" / "good" / "000.png")

        validator = DatasetValidator(dataset_root=tmp_path, target_categories=["bottle"])
        report = validator.validate()

        assert report.is_valid
        assert report.error_count == 0
        assert report.total_images_scanned == 2
        assert report.category_summaries["bottle"]["train_good_count"] == 1
        assert report.category_summaries["bottle"]["test_good_count"] == 1

    def test_valid_defective_image_with_matching_mask(self, tmp_path: Path) -> None:
        """Test validation of defective image paired with matching ground truth mask."""
        cat_dir = tmp_path / "bottle"
        create_synthetic_image(cat_dir / "train" / "good" / "000.png")
        create_synthetic_image(cat_dir / "test" / "broken_large" / "000.png")
        create_synthetic_mask(cat_dir / "ground_truth" / "broken_large" / "000_mask.png")

        validator = DatasetValidator(dataset_root=tmp_path, target_categories=["bottle"])
        report = validator.validate()

        assert report.is_valid
        assert report.error_count == 0
        assert report.total_images_scanned == 2
        assert report.total_masks_scanned == 1
        assert "broken_large" in report.category_summaries["bottle"]["defect_subtypes"]

    def test_missing_mask(self, tmp_path: Path) -> None:
        """Test error when defective test image has no matching ground-truth mask."""
        cat_dir = tmp_path / "bottle"
        create_synthetic_image(cat_dir / "train" / "good" / "000.png")
        create_synthetic_image(cat_dir / "test" / "broken_large" / "000.png")
        (cat_dir / "ground_truth" / "broken_large").mkdir(parents=True)
        # Note: 000_mask.png is NOT created

        validator = DatasetValidator(dataset_root=tmp_path, target_categories=["bottle"])
        report = validator.validate()

        assert not report.is_valid
        assert any(issue.issue_type == "MISSING_GROUND_TRUTH_MASK" for issue in report.issues)

    def test_unmatched_mask(self, tmp_path: Path) -> None:
        """Test warning when ground truth mask has no corresponding defective test image."""
        cat_dir = tmp_path / "bottle"
        create_synthetic_image(cat_dir / "train" / "good" / "000.png")
        create_synthetic_image(cat_dir / "test" / "broken_large" / "000.png")
        create_synthetic_mask(cat_dir / "ground_truth" / "broken_large" / "000_mask.png")
        # Add orphan mask
        create_synthetic_mask(cat_dir / "ground_truth" / "broken_large" / "orphan_999_mask.png")

        validator = DatasetValidator(dataset_root=tmp_path, target_categories=["bottle"])
        report = validator.validate()

        assert any(issue.issue_type == "UNMATCHED_GROUND_TRUTH_MASK" for issue in report.issues)
        assert any(issue.severity == "WARNING" for issue in report.issues)

    def test_corrupted_image(self, tmp_path: Path) -> None:
        """Test error when image binary is truncated or invalid."""
        cat_dir = tmp_path / "bottle"
        corrupt_file = cat_dir / "train" / "good" / "corrupted.png"
        corrupt_file.parent.mkdir(parents=True, exist_ok=True)
        corrupt_file.write_bytes(b"INVALID_HEADER_GARBAGE_BYTES_1234567890")

        validator = DatasetValidator(dataset_root=tmp_path, target_categories=["bottle"])
        report = validator.validate()

        assert not report.is_valid
        assert any(issue.issue_type == "CORRUPTED_IMAGE" for issue in report.issues)

    def test_invalid_mask_values(self, tmp_path: Path) -> None:
        """Test error when mask contains non-binary pixel values (e.g. grayscale 128)."""
        cat_dir = tmp_path / "bottle"
        create_synthetic_image(cat_dir / "train" / "good" / "000.png")
        create_synthetic_image(cat_dir / "test" / "contamination" / "000.png")
        create_synthetic_mask(
            cat_dir / "ground_truth" / "contamination" / "000_mask.png",
            invalid_val=128,
        )

        validator = DatasetValidator(dataset_root=tmp_path, target_categories=["bottle"])
        report = validator.validate()

        assert not report.is_valid
        assert any(issue.issue_type == "INVALID_MASK_VALUES" for issue in report.issues)

    def test_dimension_mismatch(self, tmp_path: Path) -> None:
        """Test error when source image and mask dimensions do not match."""
        cat_dir = tmp_path / "bottle"
        create_synthetic_image(cat_dir / "train" / "good" / "000.png")
        create_synthetic_image(cat_dir / "test" / "contamination" / "000.png", size=(64, 64))
        create_synthetic_mask(
            cat_dir / "ground_truth" / "contamination" / "000_mask.png",
            size=(32, 32),
        )

        validator = DatasetValidator(dataset_root=tmp_path, target_categories=["bottle"])
        report = validator.validate()

        assert not report.is_valid
        assert any(issue.issue_type == "DIMENSION_MISMATCH" for issue in report.issues)

    def test_unexpected_file(self, tmp_path: Path) -> None:
        """Test warning when unexpected files (.DS_Store, random .txt) are present."""
        cat_dir = tmp_path / "bottle"
        create_synthetic_image(cat_dir / "train" / "good" / "000.png")
        stray = cat_dir / "train" / "good" / ".DS_Store"
        stray.write_text("system junk")

        validator = DatasetValidator(dataset_root=tmp_path, target_categories=["bottle"])
        report = validator.validate()

        assert any(issue.issue_type == "UNEXPECTED_SYSTEM_FILE" for issue in report.issues)
        assert any(issue.severity == "WARNING" for issue in report.issues)


class TestLicensingGate:
    """Tests the LicensingGate authorization checks."""

    def test_licensing_gate_default_unapproved(self) -> None:
        """Test that gate rejects without environment variable or confirmation."""
        old_val = os.environ.pop(LicensingGate.ENV_APPROVAL_VAR, None)
        try:
            assert not LicensingGate.is_approved()
            assert not LicensingGate.is_approved(explicit_confirmation="NO")
        finally:
            if old_val:
                os.environ[LicensingGate.ENV_APPROVAL_VAR] = old_val

    def test_licensing_gate_explicit_confirmation(self) -> None:
        """Test that explicit confirmation approves the gate."""
        assert LicensingGate.is_approved(explicit_confirmation="CONFIRMED")
        assert LicensingGate.is_approved(explicit_confirmation="true")
        assert LicensingGate.is_approved(explicit_confirmation="YES")

    def test_licensing_gate_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that environment variable approves the gate."""
        monkeypatch.setenv(LicensingGate.ENV_APPROVAL_VAR, "TRUE")
        assert LicensingGate.is_approved()


class TestManifestAndEDA:
    """Tests preliminary manifest generation and EDA computation."""

    def test_manifest_builder(self, tmp_path: Path) -> None:
        """Test that manifest correctly indexes images with relative paths and metadata."""
        cat_dir = tmp_path / "metal_nut"
        create_synthetic_image(cat_dir / "train" / "good" / "000.png", size=(64, 64))
        create_synthetic_image(cat_dir / "test" / "scratch" / "001.png", size=(64, 64))
        create_synthetic_mask(cat_dir / "ground_truth" / "scratch" / "001_mask.png", size=(64, 64))

        builder = DatasetManifestBuilder(dataset_root=tmp_path, target_categories=["metal_nut"])
        records = builder.build_inventory()

        assert len(records) == 2

        # Check normal record
        train_rec = next(r for r in records if r.split == "train")
        assert train_rec.category == "metal_nut"
        assert train_rec.defect_type == "good"
        assert not train_rec.is_anomalous
        assert train_rec.mask_path is None
        assert train_rec.image_width == 64
        assert train_rec.image_height == 64
        assert train_rec.validation_status == "VALID"

        # Check defect record
        test_rec = next(r for r in records if r.split == "test")
        assert test_rec.category == "metal_nut"
        assert test_rec.defect_type == "scratch"
        assert test_rec.is_anomalous
        assert test_rec.mask_path == "metal_nut/ground_truth/scratch/001_mask.png"
        assert test_rec.validation_status == "VALID"

        # Export test
        json_out = tmp_path / "reports" / "manifest.json"
        csv_out = tmp_path / "reports" / "manifest.csv"
        builder.export_json(records, json_out)
        builder.export_csv(records, csv_out)

        assert json_out.is_file()
        assert csv_out.is_file()

    def test_eda_analyzer(self, tmp_path: Path) -> None:
        """Test EDA statistical computation and report generation."""
        cat_dir = tmp_path / "tile"
        create_synthetic_image(cat_dir / "train" / "good" / "000.png", size=(100, 100))
        create_synthetic_image(cat_dir / "test" / "crack" / "001.png", size=(100, 100))
        # Mask has 20x20 = 400 defect pixels out of 10000 -> 4.0%
        create_synthetic_mask(cat_dir / "ground_truth" / "crack" / "001_mask.png", size=(100, 100))

        analyzer = DatasetEDAAnalyzer(
            dataset_root=tmp_path,
            output_reports_dir=tmp_path / "reports",
            output_vis_dir=tmp_path / "vis",
        )
        profiles = analyzer.run_eda(categories=["tile"])

        assert "tile" in profiles
        profile = profiles["tile"]
        assert profile.total_images == 2
        assert profile.train_normal_count == 1
        assert profile.test_defective_count == 1
        assert profile.total_masks_count == 1
        assert profile.defect_area_stats.max_area_pixels == 400
        assert abs(profile.defect_area_stats.mean_area_pct - 4.0) < 1e-3

        # Visualizations test
        vis_paths = analyzer.generate_visualizations(category="tile")
        assert len(vis_paths) == 1
        assert vis_paths[0].is_file()
