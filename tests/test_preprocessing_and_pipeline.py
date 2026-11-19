"""Automated test suite for Phase 3 Data Preprocessing, Splitting, and Datasets.

Tests all 20 required pipeline requirements using isolated synthetic fixtures:
1. RGB image preprocessing.
2. Grayscale image conversion.
3. Invalid image rejection.
4. Image resizing.
5. Mask nearest-neighbor resizing.
6. Binary mask preservation.
7. Image-mask alignment.
8. Small defect disappearance detection.
9. Empty normal mask generation.
10. Invalid defective mask rejection.
11. Deterministic dataset splitting.
12. No cross-split image overlap.
13. Rare defect subtype handling.
14. Impossible split configuration.
15. PatchCore normal-only training enforcement.
16. U-Net image-mask pairing.
17. Augmentation synchronization.
18. Validation/test augmentation isolation.
19. Reproducibility checks.
20. Missing dataset handling.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pytest
import torch
from PIL import Image

from src.dataset.manifest import ManifestRecord
from src.dataset.splits import (
    DataLeakageError,
    DatasetSplitter,
    InvalidSplitConfigError,
    SplitConfig,
    SplitProtocol,
)
from src.dataset.torch_datasets import PatchCoreDataset, UNetDataset, create_dataloader
from src.preprocessing.augmentation import AugmentationConfig, SynchronizedAugmentor
from src.preprocessing.mask_processor import MaskPreprocessConfig, MaskProcessor
from src.preprocessing.preprocessor import ImagePreprocessConfig, ImagePreprocessor


# ---------------------------------------------------------------------------
# Synthetic Test Fixtures Helpers
# ---------------------------------------------------------------------------

def create_synthetic_image(
    path: Path,
    size: Tuple[int, int] = (64, 64),
    mode: str = "RGB",
    color: int | Tuple[int, int, int] = (100, 150, 200),
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new(mode, size, color=color)
    img.save(path, format="PNG")


def create_synthetic_mask(
    path: Path,
    size: Tuple[int, int] = (64, 64),
    defect_box: Optional[Tuple[int, int, int, int]] = (10, 10, 30, 30),
    invalid_val: Optional[int] = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.zeros(size, dtype=np.uint8)
    if defect_box:
        x0, y0, x1, y1 = defect_box
        arr[y0:y1, x0:x1] = 255
    if invalid_val is not None:
        arr[0:5, 0:5] = invalid_val
    mask = Image.fromarray(arr, mode="L")
    mask.save(path, format="PNG")


def create_sample_manifest(n_normal: int = 10, n_defect: int = 6) -> List[ManifestRecord]:
    records: List[ManifestRecord] = []
    # Normal train samples
    for i in range(n_normal):
        records.append(
            ManifestRecord(
                image_path=f"bottle/train/good/{i:03d}.png",
                mask_path=None,
                category="bottle",
                split="train",
                defect_type="good",
                is_anomalous=False,
                image_width=256,
                image_height=256,
                image_format="PNG",
                file_size_bytes=1000,
                image_sha256=f"hash_normal_{i}",
                validation_status="VALID",
            )
        )
    # Defect test samples across two subtypes
    for i in range(n_defect):
        dtype = "broken_large" if i % 2 == 0 else "contamination"
        records.append(
            ManifestRecord(
                image_path=f"bottle/test/{dtype}/{i:03d}.png",
                mask_path=f"bottle/ground_truth/{dtype}/{i:03d}_mask.png",
                category="bottle",
                split="test",
                defect_type=dtype,
                is_anomalous=True,
                image_width=256,
                image_height=256,
                image_format="PNG",
                file_size_bytes=1200,
                image_sha256=f"hash_defect_{i}",
                validation_status="VALID",
            )
        )
    return records


# ---------------------------------------------------------------------------
# Test Suite: 20 Required Pipeline Tests
# ---------------------------------------------------------------------------

class TestPhase3DataPipeline:

    # 1. RGB image preprocessing
    def test_01_rgb_image_preprocessing(self, tmp_path: Path) -> None:
        img_path = tmp_path / "test.png"
        create_synthetic_image(img_path, size=(100, 100))
        preprocessor = ImagePreprocessor()
        prep = preprocessor.preprocess(img_path)

        assert prep.tensor.shape == (3, 256, 256)
        assert prep.tensor.dtype == torch.float32
        assert prep.original_size == (100, 100)
        assert prep.original_mode == "RGB"

    # 2. Grayscale image conversion
    def test_02_grayscale_image_conversion(self, tmp_path: Path) -> None:
        img_path = tmp_path / "gray.png"
        create_synthetic_image(img_path, size=(80, 80), mode="L", color=120)
        preprocessor = ImagePreprocessor()
        prep = preprocessor.preprocess(img_path)

        assert prep.tensor.shape == (3, 256, 256)
        assert prep.original_mode == "L"

    # 3. Invalid image rejection
    def test_03_invalid_image_rejection(self, tmp_path: Path) -> None:
        corrupt_path = tmp_path / "corrupt.png"
        corrupt_path.write_bytes(b"NOT_A_VALID_IMAGE_BYTES")
        preprocessor = ImagePreprocessor()

        with pytest.raises(ValueError, match="Corrupted or unreadable"):
            preprocessor.preprocess(corrupt_path)

    # 4. Image resizing
    def test_04_image_resizing(self, tmp_path: Path) -> None:
        img_path = tmp_path / "resize.png"
        create_synthetic_image(img_path, size=(512, 384))
        preprocessor = ImagePreprocessor(ImagePreprocessConfig(target_size=(128, 128)))
        prep = preprocessor.preprocess(img_path)

        assert prep.tensor.shape == (3, 128, 128)
        assert prep.original_size == (512, 384)

    # 5. Mask nearest-neighbor resizing
    def test_05_mask_nearest_neighbor_resizing(self, tmp_path: Path) -> None:
        mask_path = tmp_path / "mask.png"
        create_synthetic_mask(mask_path, size=(64, 64))
        processor = MaskProcessor(MaskPreprocessConfig(target_size=(32, 32)))
        prep = processor.process_mask(mask_path)

        assert prep.tensor.shape == (1, 32, 32)
        unique_vals = torch.unique(prep.tensor).tolist()
        assert set(unique_vals).issubset({0.0, 1.0})

    # 6. Binary mask preservation
    def test_06_binary_mask_preservation(self, tmp_path: Path) -> None:
        mask_path = tmp_path / "binary_mask.png"
        create_synthetic_mask(mask_path, size=(64, 64), defect_box=(15, 15, 35, 35))
        processor = MaskProcessor()
        prep = processor.process_mask(mask_path)

        unique_vals = set(torch.unique(prep.tensor).numpy().tolist())
        assert unique_vals == {0.0, 1.0}
        assert prep.resized_foreground_pixels > 0

    # 7. Image-mask alignment
    def test_07_image_mask_alignment(self, tmp_path: Path) -> None:
        mask_path = tmp_path / "mismatch_mask.png"
        create_synthetic_mask(mask_path, size=(64, 64))
        processor = MaskProcessor()

        # Expected image size is (128, 128), but mask is (64, 64) -> Dimension mismatch!
        with pytest.raises(ValueError, match="Dimension mismatch"):
            processor.process_mask(mask_path, expected_image_size=(128, 128))

    # 8. Small defect disappearance detection
    def test_08_small_defect_disappearance_detection(self, tmp_path: Path) -> None:
        mask_path = tmp_path / "tiny_defect.png"
        # 1-pixel defect at (0, 0) in 1024x1024 image
        create_synthetic_mask(mask_path, size=(1024, 1024), defect_box=(0, 0, 1, 1))
        # Downsample drastically to (16, 16)
        processor = MaskProcessor(MaskPreprocessConfig(target_size=(16, 16)))
        prep = processor.process_mask(mask_path)

        assert prep.original_foreground_pixels == 1
        # When defect vanishes:
        if prep.resized_foreground_pixels == 0:
            assert prep.defect_disappeared is True

    # 9. Empty normal mask generation
    def test_09_empty_normal_mask_generation(self) -> None:
        processor = MaskProcessor(MaskPreprocessConfig(target_size=(256, 256)))
        prep = processor.process_mask(mask_input=None, expected_image_size=(900, 900))

        assert prep.tensor.shape == (1, 256, 256)
        assert torch.all(prep.tensor == 0.0)
        assert prep.is_empty_normal is True
        assert prep.original_size == (900, 900)

    # 10. Invalid defective mask rejection
    def test_10_invalid_defective_mask_rejection(self, tmp_path: Path) -> None:
        mask_path = tmp_path / "invalid_mask.png"
        # Mask contains pixel value 128 (invalid intermediate value)
        create_synthetic_mask(mask_path, size=(64, 64), invalid_val=128)
        processor = MaskProcessor()

        with pytest.raises(ValueError, match="Invalid non-binary mask pixel values"):
            processor.process_mask(mask_path)

    # 11. Deterministic dataset splitting
    def test_11_deterministic_dataset_splitting(self) -> None:
        records = create_sample_manifest(n_normal=20, n_defect=10)
        cfg = SplitConfig(protocol=SplitProtocol.PROTOCOL_A_PATCHCORE, train_ratio=0.8, val_ratio=0.2, seed=42)

        splitter_1 = DatasetSplitter(cfg)
        res_1 = splitter_1.split(records)

        splitter_2 = DatasetSplitter(cfg)
        res_2 = splitter_2.split(records)

        paths_1 = [r.image_path for r in res_1.train_records]
        paths_2 = [r.image_path for r in res_2.train_records]
        assert paths_1 == paths_2

    # 12. No cross-split image overlap
    def test_12_no_cross_split_image_overlap(self) -> None:
        records = create_sample_manifest(n_normal=30, n_defect=12)
        cfg = SplitConfig(
            protocol=SplitProtocol.PROTOCOL_B_SUPERVISED_UNET,
            train_ratio=0.6,
            val_ratio=0.2,
            test_ratio=0.2,
            seed=100,
        )
        splitter = DatasetSplitter(cfg)
        res = splitter.split(records)

        # Built-in verification
        res.verify_no_leakage()

        train_s = {r.image_path for r in res.train_records}
        val_s = {r.image_path for r in res.val_records}
        test_s = {r.image_path for r in res.test_records}

        assert len(train_s & val_s) == 0
        assert len(train_s & test_s) == 0
        assert len(val_s & test_s) == 0
        assert len(train_s | val_s | test_s) == len(records)

    # 13. Rare defect subtype handling
    def test_13_rare_defect_subtype_handling(self) -> None:
        records = create_sample_manifest(n_normal=10, n_defect=0)
        # Add a rare defect subtype with only 1 sample
        records.append(
            ManifestRecord(
                image_path="tile/test/rare_blemish/001.png",
                mask_path="tile/ground_truth/rare_blemish/001_mask.png",
                category="tile",
                split="test",
                defect_type="rare_blemish",
                is_anomalous=True,
                image_width=256,
                image_height=256,
                image_format="PNG",
                file_size_bytes=1000,
                image_sha256="rare_hash_1",
                validation_status="VALID",
            )
        )
        cfg = SplitConfig(
            protocol=SplitProtocol.PROTOCOL_B_SUPERVISED_UNET,
            train_ratio=0.5,
            val_ratio=0.25,
            test_ratio=0.25,
            seed=42,
        )
        splitter = DatasetSplitter(cfg)
        # Should not crash or fail when handling the 1 rare sample
        res = splitter.split(records)
        assert len(res.train_records) > 0

    # 14. Impossible split configuration
    def test_14_impossible_split_configuration(self) -> None:
        # Ratios sum to 0.70 instead of 1.0 -> Invalid!
        with pytest.raises(InvalidSplitConfigError, match="requires train_ratio"):
            SplitConfig(protocol=SplitProtocol.PROTOCOL_A_PATCHCORE, train_ratio=0.5, val_ratio=0.2)

        with pytest.raises(InvalidSplitConfigError, match="requires train_ratio"):
            SplitConfig(
                protocol=SplitProtocol.PROTOCOL_B_SUPERVISED_UNET,
                train_ratio=0.5,
                val_ratio=0.2,
                test_ratio=0.1,
            )

    # 15. PatchCore normal-only training enforcement
    def test_15_patchcore_normal_only_training_enforcement(self, tmp_path: Path) -> None:
        img_path = tmp_path / "defect.png"
        create_synthetic_image(img_path)

        defective_record = ManifestRecord(
            image_path="defect.png",
            mask_path=None,
            category="bottle",
            split="train",
            defect_type="broken",
            is_anomalous=True,  # ANOMALOUS SAMPLE IN TRAINING
            image_width=64,
            image_height=64,
            image_format="PNG",
            file_size_bytes=500,
            image_sha256="hash1",
            validation_status="VALID",
        )

        with pytest.raises(ValueError, match="Protocol Violation: PatchCore training dataset must contain exclusively normal"):
            PatchCoreDataset(records=[defective_record], dataset_root=tmp_path, is_train=True)

    # 16. U-Net image-mask pairing
    def test_16_unet_image_mask_pairing(self, tmp_path: Path) -> None:
        img_path = tmp_path / "bottle/test/broken/000.png"
        mask_path = tmp_path / "bottle/ground_truth/broken/000_mask.png"
        create_synthetic_image(img_path, size=(64, 64))
        create_synthetic_mask(mask_path, size=(64, 64))

        record = ManifestRecord(
            image_path="bottle/test/broken/000.png",
            mask_path="bottle/ground_truth/broken/000_mask.png",
            category="bottle",
            split="test",
            defect_type="broken",
            is_anomalous=True,
            image_width=64,
            image_height=64,
            image_format="PNG",
            file_size_bytes=600,
            image_sha256="h1",
            validation_status="VALID",
        )

        ds = UNetDataset(records=[record], dataset_root=tmp_path, is_train=False)
        item = ds[0]

        assert item["image"].shape == (3, 256, 256)
        assert item["mask"].shape == (1, 256, 256)
        assert item["category"] == "bottle"
        assert item["is_anomalous"] is True

    # 17. Augmentation synchronization
    def test_17_augmentation_synchronization(self) -> None:
        # Create asymmetric tensor
        img = torch.zeros((3, 64, 64), dtype=torch.float32)
        mask = torch.zeros((1, 64, 64), dtype=torch.float32)

        # Place defect in top-left quadrant [0:20, 0:20]
        img[:, 0:20, 0:20] = 1.0
        mask[:, 0:20, 0:20] = 1.0

        # Force horizontal flip with prob=1.0 and zero color jitter for exact geometric check
        cfg = AugmentationConfig(
            horizontal_flip_prob=1.0,
            vertical_flip_prob=0.0,
            rotation_90_prob=0.0,
            brightness_jitter=0.0,
            contrast_jitter=0.0,
        )
        aug = SynchronizedAugmentor(cfg)
        aug_img, aug_mask = aug.transform(img, mask)

        assert aug_mask is not None
        # Flipped horizontally: defect should now be in top-right quadrant [0:20, 44:64]
        assert aug_mask[0, 10, 50].item() == 1.0
        assert aug_img[0, 10, 50].item() == 1.0
        assert aug_mask[0, 10, 10].item() == 0.0

    # 18. Validation/test augmentation isolation
    def test_18_validation_test_augmentation_isolation(self, tmp_path: Path) -> None:
        img_path = tmp_path / "bottle/train/good/000.png"
        create_synthetic_image(img_path, size=(64, 64))

        record = ManifestRecord(
            image_path="bottle/train/good/000.png",
            mask_path=None,
            category="bottle",
            split="val",
            defect_type="good",
            is_anomalous=False,
            image_width=64,
            image_height=64,
            image_format="PNG",
            file_size_bytes=500,
            image_sha256="h0",
            validation_status="VALID",
        )

        cfg = AugmentationConfig(horizontal_flip_prob=1.0)
        aug = SynchronizedAugmentor(cfg)

        # When is_train=False, augmentor must be bypassed
        ds_val = UNetDataset(records=[record], dataset_root=tmp_path, augmentor=aug, is_train=False)
        item1 = ds_val[0]
        item2 = ds_val[0]

        assert torch.all(item1["image"] == item2["image"])

    # 19. Reproducibility checks
    def test_19_reproducibility_checks(self) -> None:
        records = create_sample_manifest(n_normal=25, n_defect=10)
        cfg1 = SplitConfig(seed=42)
        cfg2 = SplitConfig(seed=999)

        res1 = DatasetSplitter(cfg1).split(records)
        res2 = DatasetSplitter(cfg2).split(records)

        # Different seeds should yield different random permutations
        paths1 = [r.image_path for r in res1.train_records]
        paths2 = [r.image_path for r in res2.train_records]
        assert paths1 != paths2

    # 20. Missing dataset handling
    def test_20_missing_dataset_handling(self, tmp_path: Path) -> None:
        non_existent = tmp_path / "does_not_exist.png"
        preprocessor = ImagePreprocessor()

        with pytest.raises(FileNotFoundError, match="Image file does not exist"):
            preprocessor.preprocess(non_existent)
