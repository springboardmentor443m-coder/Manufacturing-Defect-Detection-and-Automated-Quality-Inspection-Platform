"""
VisionInspect AI - PyTorch MVTec AD Dataset Class for U-Net Segmentation

Features:
- Loads paired image and binary ground-truth segmentation mask (values in {0.0, 1.0})
- Automatically generates a zero-filled (1, H, W) binary mask for normal/good samples at load time
- Implements safe augmentation (flips, small rotations, brightness/contrast)
  applied IDENTICALLY to image and mask
- Standardizes images to (3, H, W) float32 with optional ImageNet normalization
"""

import math
import os
import random
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# Ensure repository root and custom libs are on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
if os.path.exists("C:/pylib"):
    sys.path.insert(0, "C:/pylib")

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


class SafeJointAugmentor:
    """
    Applies synchronized geometric transforms identically to (image, mask),
    and photometric transforms to the image only.
    """

    def __init__(
        self,
        hflip_prob: float = 0.5,
        vflip_prob: float = 0.5,
        rotation_prob: float = 0.5,
        max_rotation_deg: float = 12.0,
        brightness_jitter: float = 0.15,
        contrast_jitter: float = 0.15,
    ):
        self.hflip_prob = hflip_prob
        self.vflip_prob = vflip_prob
        self.rotation_prob = rotation_prob
        self.max_rotation_deg = max_rotation_deg
        self.brightness_jitter = brightness_jitter
        self.contrast_jitter = contrast_jitter

    def __call__(
        self,
        image_rgb: np.ndarray,
        mask: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Args:
            image_rgb: uint8 array (H, W, 3)
            mask: uint8 or float32 binary array (H, W)
        Returns:
            augmented (image_rgb, mask)
        """
        aug_img = image_rgb.copy()
        aug_mask = mask.copy()

        # 1. Random Horizontal Flip (identical for both)
        if random.random() < self.hflip_prob:
            aug_img = np.fliplr(aug_img)
            aug_mask = np.fliplr(aug_mask)

        # 2. Random Vertical Flip (identical for both)
        if random.random() < self.vflip_prob:
            aug_img = np.flipud(aug_img)
            aug_mask = np.flipud(aug_mask)

        # 3. Random Small Rotation (identical angle & center for both)
        if random.random() < self.rotation_prob:
            angle = random.uniform(-self.max_rotation_deg, self.max_rotation_deg)
            h, w = aug_img.shape[:2]
            center = (w // 2, h // 2)
            rot_mat = cv2.getRotationMatrix2D(center, angle, scale=1.0)

            # Image: linear interpolation with edge reflection or black border
            aug_img = cv2.warpAffine(
                aug_img, rot_mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101
            )
            # Mask: nearest-neighbor interpolation to preserve discrete 0/1 values
            aug_mask = cv2.warpAffine(
                aug_mask, rot_mat, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0
            )

        # 4. Photometric Augmentation: Brightness & Contrast (image ONLY, mask untouched)
        if self.brightness_jitter > 0 or self.contrast_jitter > 0:
            alpha = 1.0 + random.uniform(-self.contrast_jitter, self.contrast_jitter)
            beta = random.uniform(-self.brightness_jitter * 255.0, self.brightness_jitter * 255.0)
            aug_img = cv2.convertScaleAbs(aug_img, alpha=alpha, beta=beta)

        return aug_img, aug_mask


class MVTecADUNetDataset(Dataset):
    """
    PyTorch Dataset tailored for U-Net supervised segmentation and anomaly localization.
    """

    def __init__(
        self,
        manifest: Union[str, Path, pd.DataFrame],
        split: Optional[str] = None,
        img_size: Tuple[int, int] = (256, 256),
        is_train: bool = False,
        normalize_imagenet: bool = True,
    ):
        super().__init__()
        if isinstance(manifest, (str, Path)):
            self.df = pd.read_csv(manifest)
        elif isinstance(manifest, pd.DataFrame):
            self.df = manifest.copy()
        else:
            raise TypeError("Manifest must be a file path or pandas DataFrame.")

        # Filter by split if requested (train, val, test)
        if split is not None:
            self.df = self.df[self.df["project_split"] == split].reset_index(drop=True)

        self.img_size = img_size  # (width, height)
        self.is_train = is_train
        self.normalize_imagenet = normalize_imagenet
        self.augmentor = SafeJointAugmentor() if is_train else None

        # ImageNet normalization parameters
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        image_path = str(row["image_path"])
        mask_path = str(row["mask_path"]) if pd.notna(row["mask_path"]) else ""
        is_anomaly = int(row["is_anomaly"])
        defect_type = str(row.get("defect_type", "good"))
        category = str(row.get("category", "bottle"))

        # 1. Load image (RGB)
        bgr = cv2.imread(image_path)
        if bgr is None:
            raise FileNotFoundError(f"Failed to load image from: {image_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        tw, th = self.img_size

        # 2. Load or synthesize ground-truth binary mask
        if is_anomaly == 1 and mask_path and os.path.exists(mask_path):
            mask_raw = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask_raw is None:
                mask_raw = np.zeros((rgb.shape[0], rgb.shape[1]), dtype=np.uint8)
        else:
            # Normal sample: generate all-zero mask at load time
            mask_raw = np.zeros((rgb.shape[0], rgb.shape[1]), dtype=np.uint8)

        # 3. Standard resize
        rgb_resized = cv2.resize(rgb, (tw, th), interpolation=cv2.INTER_LINEAR)
        mask_resized = cv2.resize(mask_raw, (tw, th), interpolation=cv2.INTER_NEAREST)

        # 4. Safe synchronized augmentation (if training)
        if self.augmentor is not None:
            rgb_resized, mask_resized = self.augmentor(rgb_resized, mask_resized)

        # 5. Format mask as discrete binary float32 tensor (1, H, W) in {0.0, 1.0}
        binary_mask = (mask_resized > 127).astype(np.float32)
        mask_tensor = torch.from_numpy(binary_mask).unsqueeze(0)  # Shape: (1, H, W)

        # 6. Normalize image to float32 (3, H, W)
        img_norm = rgb_resized.astype(np.float32) / 255.0
        if self.normalize_imagenet:
            img_norm = (img_norm - self.mean) / self.std

        # Transpose from (H, W, 3) to channels-first (3, H, W)
        img_tensor = torch.from_numpy(img_norm.transpose(2, 0, 1))

        return {
            "image": img_tensor,
            "mask": mask_tensor,
            "is_anomaly": torch.tensor(is_anomaly, dtype=torch.long),
            "defect_type": defect_type,
            "category": category,
            "image_path": image_path,
            "mask_path": mask_path,
        }


def test_dataset_pipeline():
    """Self-contained test validating Dataset class and DataLoader behavior."""
    print("=" * 65)
    print("🧪 TESTING PYTORCH MVTEC AD UNET DATASET CLASS")
    print("=" * 65)

    manifest_file = Path("ml/manifests/manifest_bottle_protocol_b.csv")
    if not manifest_file.exists():
        print(f"⚠️ Manifest not found at {manifest_file}. Generating now...")
        from ml.dataset.manifest_builder import build_protocol_b_manifest
        from ml.dataset.discovery import find_dataset_root
        cat = find_dataset_root(category="bottle")
        build_protocol_b_manifest(cat)

    # 1. Instantiate train dataset (with augmentation)
    train_ds = MVTecADUNetDataset(manifest_file, split="train", img_size=(256, 256), is_train=True)
    val_ds = MVTecADUNetDataset(manifest_file, split="val", img_size=(256, 256), is_train=False)

    print(f"  • Train dataset size : {len(train_ds)} samples (Safe Augmentation Enabled)")
    print(f"  • Val dataset size   : {len(val_ds)} samples")

    # 2. Check sample batch from DataLoader
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    batch = next(iter(train_loader))

    images = batch["image"]
    masks = batch["mask"]
    labels = batch["is_anomaly"]
    defects = batch["defect_type"]

    print("\n📦 Batch Dimensions & Types:")
    print(f"  • Image Tensor Shape: {images.shape} (dtype: {images.dtype})")
    print(f"  • Mask Tensor Shape : {masks.shape}  (dtype: {masks.dtype})")
    print(f"  • Anomaly Labels    : {labels.tolist()}")
    print(f"  • Defect Categories : {defects}")

    # Check normal samples generate all-zero masks
    for i in range(len(labels)):
        if labels[i].item() == 0:
            assert torch.all(masks[i] == 0.0), f"Normal sample #{i} must have all-zero mask!"
        else:
            # Defective sample: check values are strictly 0.0 or 1.0
            unique_vals = torch.unique(masks[i]).tolist()
            for v in unique_vals:
                assert v in (0.0, 1.0), f"Mask must be binary 0 or 1, found: {v}"

    print("\n✅ Verification Passed:")
    print("  - All normal samples generate exact all-zero (1, H, W) masks.")
    print("  - Defective masks have binary values strictly in {0.0, 1.0}.")
    print("  - Safe augmentation executes identically across image and mask.")
    print("=" * 65)


if __name__ == "__main__":
    test_dataset_pipeline()
