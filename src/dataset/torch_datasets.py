"""PyTorch Dataset implementations for VisionInspect AI.

Provides strictly segregated dataset interfaces for U-Net (supervised segmentation)
and PatchCore (unsupervised one-class anomaly detection), with normal-only training enforcement.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from torch.utils.data import DataLoader, Dataset

from src.preprocessing.augmentation import SynchronizedAugmentor
from src.preprocessing.mask_processor import MaskProcessor
from src.preprocessing.preprocessor import ImagePreprocessor
from .manifest import ManifestRecord

logger = logging.getLogger(__name__)


class UNetDataset(Dataset):
    """PyTorch Dataset for supervised defect segmentation (U-Net).

    Pairs RGB images with ground-truth binary masks, automatically synthesizing
    empty masks for normal samples without modifying or creating physical files on disk.
    """

    def __init__(
        self,
        records: List[ManifestRecord],
        dataset_root: Path,
        preprocessor: Optional[ImagePreprocessor] = None,
        mask_processor: Optional[MaskProcessor] = None,
        augmentor: Optional[SynchronizedAugmentor] = None,
        is_train: bool = False,
    ) -> None:
        self.records = list(records)
        self.dataset_root = Path(dataset_root)
        self.preprocessor = preprocessor or ImagePreprocessor()
        self.mask_processor = mask_processor or MaskProcessor()
        self.augmentor = augmentor
        self.is_train = is_train

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        record = self.records[idx]
        image_abs_path = self.dataset_root / record.image_path

        # 1. Preprocess RGB Image
        prep_img = self.preprocessor.preprocess(image_abs_path)
        img_tensor = prep_img.tensor
        orig_size = prep_img.original_size

        # 2. Preprocess Binary Mask
        mask_abs_path = (self.dataset_root / record.mask_path) if record.mask_path else None
        prep_mask = self.mask_processor.process_mask(
            mask_input=mask_abs_path,
            expected_image_size=orig_size,
        )
        mask_tensor = prep_mask.tensor

        # 3. Synchronized Augmentation (strictly if is_train and augmentor provided)
        if self.is_train and self.augmentor is not None:
            img_tensor, mask_tensor = self.augmentor.transform(img_tensor, mask_tensor)
            if mask_tensor is None:
                mask_tensor = prep_mask.tensor

        return {
            "image": img_tensor,                                # Shape: (3, H, W), float32
            "mask": mask_tensor,                                # Shape: (1, H, W), float32 in {0.0, 1.0}
            "category": record.category,
            "defect_type": record.defect_type,
            "is_anomalous": record.is_anomalous,
            "original_size": orig_size,
            "image_path": record.image_path,
            "mask_path": record.mask_path,
        }


class PatchCoreDataset(Dataset):
    """PyTorch Dataset for unsupervised one-class anomaly detection (PatchCore).

    Enforces that only normal, defect-free images enter the training memory bank.
    """

    def __init__(
        self,
        records: List[ManifestRecord],
        dataset_root: Path,
        preprocessor: Optional[ImagePreprocessor] = None,
        is_train: bool = False,
    ) -> None:
        self.records = list(records)
        self.dataset_root = Path(dataset_root)
        self.preprocessor = preprocessor or ImagePreprocessor()
        self.is_train = is_train

        # STRICT ENFORCEMENT: Training set must contain exclusively normal images
        if self.is_train:
            anomalous = [r for r in self.records if r.is_anomalous]
            if anomalous:
                raise ValueError(
                    f"Protocol Violation: PatchCore training dataset must contain exclusively normal images! "
                    f"Found {len(anomalous)} anomalous samples in training split (e.g. {anomalous[0].image_path})."
                )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        record = self.records[idx]
        image_abs_path = self.dataset_root / record.image_path

        prep_img = self.preprocessor.preprocess(image_abs_path)

        return {
            "image": prep_img.tensor,                            # Shape: (3, H, W), float32
            "category": record.category,
            "defect_type": record.defect_type,
            "is_anomalous": record.is_anomalous,
            "original_size": prep_img.original_size,
            "image_path": record.image_path,
        }


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 16,
    shuffle: bool = False,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = False,
) -> DataLoader:
    """Configures and returns an optimized PyTorch DataLoader instance."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
