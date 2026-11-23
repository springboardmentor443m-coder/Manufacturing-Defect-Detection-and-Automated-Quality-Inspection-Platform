#!/usr/bin/env python
"""
ml/train/bottle_unet_v2.py
Phase 4A-v2 – U-Net v2 training for the 'bottle' category.

Improvements over v1:
  - albumentations augmentation with REFLECT_101 padding (no zero-border artifact)
  - Per-image hard-thresholded Dice as validation metric (fix v1 batch soft-Dice bug)
  - Border margin cropping (4px at 256px, 6px at 384px) before computing metrics
  - Connected-component area filter (min_area tuned on validation, start 150px at 256px)
  - Selection score = mean per-image Dice on defective - 0.5 * FP rate on normal
    (FP = normal image has a CC larger than min_area after border margin)
  - Joint threshold + min_area grid tuning on validation only
  - BCEWithLogits + SoftDice + Focal loss, configurable weights
  - AdamW lr 3e-4, 3-epoch linear warmup then CosineAnnealingLR
  - Weighted sampler (~50% defective per batch)
  - TTA: average over {orig, hflip, vflip, hflip+vflip}
  - Multi-seed loop, reports mean ± std across seeds
  - Checks whether (0,1) artifact is still present in normal images post-training
  - Recomputes v1 metrics fairly with same border+CC filter
  - Registry update: does NOT overwrite v1 entry

Usage:
    python ml/train/bottle_unet_v2.py --config ml/configs/bottle_unet_v2_256.json
"""

import argparse
import hashlib
import json
import math
import random
import sys
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

warnings.filterwarnings("ignore", category=UserWarning)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ──────────────────────────────────────────────────────────────────────────────
# 0. CONFIG LOADING
# ──────────────────────────────────────────────────────────────────────────────
DEFAULT_CFG = {
    "image_size": 256,
    "encoder": "resnet34",
    "batch_size": 8,
    "fallback_batch_size": 4,
    "epochs": 100,
    "min_epochs": 30,
    "patience": 20,
    "lr": 3e-4,
    "weight_decay": 1e-4,
    "warmup_epochs": 3,
    "seeds": [42, 43, 44],
    "loss_bce_weight": 0.3,
    "loss_dice_weight": 0.5,
    "loss_focal_weight": 0.2,
    "tversky_beta": 0.7,
    "border_margin_px": 4,
    "min_component_area_start": 150,
    "amp": True,
    "tta": True,
    "out_dir": "ml/outputs/bottle/unet_v2",
    "version": "2.0-256",
}


def load_config(config_path: str | None) -> dict:
    cfg = DEFAULT_CFG.copy()
    if config_path:
        with open(config_path, "r") as f:
            override = json.load(f)
        cfg.update(override)
    return cfg


# ──────────────────────────────────────────────────────────────────────────────
# 1. DEVICE SETUP
# ──────────────────────────────────────────────────────────────────────────────
def setup_device():
    cuda = torch.cuda.is_available()
    if cuda:
        device = torch.device("cuda")
        dev_name = torch.cuda.get_device_name(0)
        total_vram_mib = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
        print("=" * 65)
        print("DEVICE: GPU")
        print(f"  GPU        : {dev_name}")
        print(f"  Total VRAM : {total_vram_mib:.0f} MiB")
        print(f"  CUDA       : {getattr(torch.version, 'cuda', 'unknown')}")
        print("=" * 65)
    else:
        device = torch.device("cpu")
        dev_name = "CPU"
        total_vram_mib = 0.0
        print("WARNING: CUDA not available. Falling back to CPU.")
    return device, cuda, dev_name, total_vram_mib


# ──────────────────────────────────────────────────────────────────────────────
# 2. AUGMENTATION
# ──────────────────────────────────────────────────────────────────────────────
def build_augmentation(image_size: int, is_train: bool):
    """
    Returns an albumentations transform.
    All padding uses REFLECT_101 to eliminate zero-border artifacts.
    """
    import albumentations as A
    import cv2

    if is_train:
        transforms = A.Compose([
            A.SmallestMaxSize(max_size=image_size + 32, interpolation=cv2.INTER_LINEAR),
            A.RandomCrop(height=image_size, width=image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.05, scale_limit=0.1, rotate_limit=15,
                border_mode=cv2.BORDER_REFLECT_101, p=0.5
            ),
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.4),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ], additional_targets={"mask": "mask"})
    else:
        transforms = A.Compose([
            A.Resize(height=image_size, width=image_size, interpolation=cv2.INTER_LINEAR),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ], additional_targets={"mask": "mask"})
    return transforms


# ──────────────────────────────────────────────────────────────────────────────
# 3. DATASET
# ──────────────────────────────────────────────────────────────────────────────
class BottleDatasetV2(Dataset):
    """Protocol B Dataset with albumentations augmentation."""

    def __init__(self, manifest_path: Path, image_size: int, is_train: bool = False):
        self.df = pd.read_csv(manifest_path)
        self.image_size = image_size
        self.transform = build_augmentation(image_size, is_train)
        for col in ("image_path", "is_anomaly"):
            if col not in self.df.columns:
                raise ValueError(f"Manifest {manifest_path} missing column '{col}'")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        with Image.open(row["image_path"]) as pil_img:
            img_np = np.array(pil_img.convert("RGB"), dtype=np.uint8)

        mask_val = row.get("mask_path", "")
        if pd.isna(mask_val) or str(mask_val).strip() == "":
            mask_np = np.zeros((img_np.shape[0], img_np.shape[1]), dtype=np.uint8)
        else:
            with Image.open(str(mask_val).strip()) as pil_mask:
                mask_np = np.array(pil_mask.convert("L"), dtype=np.uint8)
                mask_np = (mask_np > 0).astype(np.uint8)

        augmented = self.transform(image=img_np, mask=mask_np)
        img_aug = augmented["image"]   # H x W x 3, float32 normalized
        mask_aug = augmented["mask"]   # H x W, uint8

        img_t = torch.from_numpy(img_aug.transpose(2, 0, 1)).float()
        mask_t = torch.from_numpy(mask_aug.astype(np.float32)).unsqueeze(0)
        label = int(row["is_anomaly"])
        # Also keep the filename for diagnostics
        fname = Path(row["image_path"]).name
        return img_t, mask_t, label, fname

    def get_defect_type(self, idx):
        row = self.df.iloc[idx]
        p = Path(row["image_path"])
        return p.parent.name  # e.g. broken_large, broken_small, contamination, good


def collate_fn(batch):
    imgs, masks, labels, fnames = zip(*batch)
    return (torch.stack(imgs), torch.stack(masks),
            torch.tensor(labels, dtype=torch.long), list(fnames))


# ──────────────────────────────────────────────────────────────────────────────
# 4. LOSS FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────
def soft_dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    p = probs.reshape(probs.size(0), -1)
    t = targets.reshape(targets.size(0), -1)
    inter = (p * t).sum(dim=1)
    union = p.sum(dim=1) + t.sum(dim=1)
    dice = 1.0 - (2.0 * inter + eps) / (union + eps)
    return dice.mean()


def focal_loss(logits: torch.Tensor, targets: torch.Tensor,
               gamma: float = 2.0, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    pt = torch.where(targets > 0.5, probs, 1.0 - probs)
    fl = ((1 - pt) ** gamma) * bce
    return fl.mean()


class CombinedLoss(nn.Module):
    def __init__(self, bce_w: float = 0.3, dice_w: float = 0.5, focal_w: float = 0.2):
        super().__init__()
        self.bce_w = bce_w
        self.dice_w = dice_w
        self.focal_w = focal_w
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return (self.bce_w * self.bce(logits, targets)
                + self.dice_w * soft_dice_loss(logits, targets)
                + self.focal_w * focal_loss(logits, targets))


# ──────────────────────────────────────────────────────────────────────────────
# 5. BORDER MASK & CONNECTED-COMPONENT FILTER
# ──────────────────────────────────────────────────────────────────────────────
def make_interior_mask(H: int, W: int, margin: int) -> np.ndarray:
    """Returns bool mask True inside the border margin."""
    m = np.zeros((H, W), dtype=bool)
    if margin > 0:
        m[margin:H - margin, margin:W - margin] = True
    else:
        m[:] = True
    return m


def apply_border_and_cc_filter(pred_bin: np.ndarray,
                                interior: np.ndarray,
                                min_area: int) -> np.ndarray:
    """
    1. Zero out border margin pixels.
    2. Remove connected components smaller than min_area.
    Returns filtered binary mask (same shape).
    """
    from scipy import ndimage as ndi
    result = pred_bin.copy()
    result[~interior] = 0  # zero border margin

    labeled, n_cc = ndi.label(result)
    if n_cc > 0:
        sizes = ndi.sum(result, labeled, range(1, n_cc + 1))
        for cc_idx, sz in enumerate(sizes, start=1):
            if sz < min_area:
                result[labeled == cc_idx] = 0
    return result


# ──────────────────────────────────────────────────────────────────────────────
# 6. PER-IMAGE HARD DICE & METRICS (THE KEY V2 FIX)
# ──────────────────────────────────────────────────────────────────────────────
def per_image_hard_dice_iou(probs_np: np.ndarray, mask_np: np.ndarray,
                             thr: float, interior: np.ndarray,
                             min_area: int) -> tuple[float, float]:
    """
    Hard-threshold probs → apply border+CC filter → compute Dice & IoU vs true mask.
    True mask is also cropped to interior.
    """
    pred_bin = (probs_np > thr).astype(np.uint8)
    pred_filt = apply_border_and_cc_filter(pred_bin, interior, min_area)
    true_filt = (mask_np > 0.5).astype(np.uint8)
    true_filt[~interior] = 0

    inter = (pred_filt & true_filt).sum()
    pred_sum = pred_filt.sum()
    true_sum = true_filt.sum()
    union = pred_sum + true_sum

    dice = (2 * inter) / (union + 1e-6) if union > 0 else 1.0
    iou = inter / (inter + pred_sum - inter + true_sum - inter + 1e-6) if (pred_sum + true_sum) > 0 else 1.0
    return float(dice), float(iou)


def compute_val_metrics_v2(all_probs: list[np.ndarray],
                            all_masks: list[np.ndarray],
                            all_labels: list[int],
                            thr: float,
                            interior: np.ndarray,
                            min_area: int) -> dict:
    """
    Per-image hard Dice, IoU, and FP rate.
    FP = normal image has at least one CC > min_area after border margin.
    """
    dice_def, iou_def = [], []
    fp_normal = []

    for prob_np, mask_np, label in zip(all_probs, all_masks, all_labels):
        pred_bin = (prob_np > thr).astype(np.uint8)
        pred_filt = apply_border_and_cc_filter(pred_bin, interior, min_area)

        if label == 1:
            d, iou = per_image_hard_dice_iou(prob_np, mask_np, thr, interior, min_area)
            dice_def.append(d)
            iou_def.append(iou)
        else:
            # FP if any pixel survives after CC filter
            fp_normal.append(float(pred_filt.sum() > 0))

    mean_dice = float(np.mean(dice_def)) if dice_def else 0.0
    mean_iou = float(np.mean(iou_def)) if iou_def else 0.0
    fp_rate = float(np.mean(fp_normal)) if fp_normal else 0.0
    score = mean_dice - 0.5 * fp_rate
    return {
        "dice_def": mean_dice,
        "iou_def": mean_iou,
        "fp_rate": fp_rate,
        "score": score,
        "n_defective": len(dice_def),
        "n_normal": len(fp_normal),
    }


# ──────────────────────────────────────────────────────────────────────────────
# 7. TTA INFERENCE
# ──────────────────────────────────────────────────────────────────────────────
def tta_predict(model, imgs: torch.Tensor, device, use_amp: bool) -> torch.Tensor:
    """
    Average sigmoid probabilities over 4 TTA variants:
    original, hflip, vflip, hflip+vflip.
    """
    model.eval()
    variants = [
        imgs,
        torch.flip(imgs, dims=[3]),            # hflip
        torch.flip(imgs, dims=[2]),            # vflip
        torch.flip(imgs, dims=[2, 3]),         # hflip+vflip
    ]
    probs_sum = None
    with torch.no_grad():
        for var in variants:
            var = var.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(var)
            probs = torch.sigmoid(logits).cpu()
            # Un-flip for hflip/vflip variants
            if var is variants[1]:  # hflip → flip back
                probs = torch.flip(probs, dims=[3])
            elif var is variants[2]:  # vflip
                probs = torch.flip(probs, dims=[2])
            elif var is variants[3]:  # both
                probs = torch.flip(probs, dims=[2, 3])

            if probs_sum is None:
                probs_sum = probs
            else:
                probs_sum = probs_sum + probs

    return probs_sum / 4.0  # [B, 1, H, W]


def tta_predict_loader(model, loader, device, use_amp: bool):
    """Run TTA on entire loader, return lists of prob arrays, mask arrays, labels, fnames."""
    all_probs, all_masks, all_labels, all_fnames = [], [], [], []
    model.eval()
    for imgs, masks, labels, fnames in loader:
        probs = tta_predict(model, imgs, device, use_amp)  # [B,1,H,W] cpu
        for i in range(imgs.size(0)):
            all_probs.append(probs[i, 0].numpy())
            all_masks.append(masks[i, 0].numpy())
            all_labels.append(labels[i].item())
            all_fnames.append(fnames[i])
    return all_probs, all_masks, all_labels, all_fnames


# ──────────────────────────────────────────────────────────────────────────────
# 8. THRESHOLD + MIN-AREA GRID TUNING (VALIDATION ONLY)
# ──────────────────────────────────────────────────────────────────────────────
def tune_threshold_and_area(all_probs, all_masks, all_labels, image_size: int,
                             border_margin: int, min_area_start: int,
                             out_dir: Path) -> tuple[float, int, pd.DataFrame]:
    """
    Grid search over threshold ∈ {0.25, 0.30, 0.35, 0.40, 0.45, 0.50}
    and min_area ∈ {100, 150, 200, 300} (scaled for image size).
    Selection: highest score = mean_dice_defective - 0.5 * fp_rate_normal.
    Returns best_thr, best_min_area, sweep_df.
    """
    thresholds = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    # Scale min_area candidates from 256px reference
    scale = (image_size / 256) ** 2
    min_areas = [max(50, int(a * scale)) for a in [100, 150, 200, 300]]

    H, W = image_size, image_size
    interior = make_interior_mask(H, W, border_margin)

    records = []
    best_score = -float("inf")
    best_thr = 0.35
    best_area = min_area_start

    for thr in thresholds:
        for ma in min_areas:
            m = compute_val_metrics_v2(all_probs, all_masks, all_labels, thr, interior, ma)
            records.append({
                "threshold": thr,
                "min_area": ma,
                "dice_def": round(m["dice_def"], 4),
                "iou_def": round(m["iou_def"], 4),
                "fp_rate": round(m["fp_rate"], 4),
                "score": round(m["score"], 4),
            })
            if m["score"] > best_score:
                best_score = m["score"]
                best_thr = thr
                best_area = ma

    df = pd.DataFrame(records)
    df.to_csv(out_dir / "threshold_area_tuning.csv", index=False)
    print(f"\n  Threshold+Area tuning grid saved → {out_dir / 'threshold_area_tuning.csv'}")
    print(f"  Best threshold: {best_thr:.2f}  |  Best min_area: {best_area}  |  Score: {best_score:.4f}")
    return best_thr, best_area, df


# ──────────────────────────────────────────────────────────────────────────────
# 9. SCHEDULER WITH WARMUP
# ──────────────────────────────────────────────────────────────────────────────
class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_epochs: int, total_epochs: int, eta_min: float = 1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.eta_min = eta_min
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]

    def step(self, epoch: int):
        """epoch is 1-indexed."""
        if epoch <= self.warmup_epochs:
            factor = epoch / max(1, self.warmup_epochs)
        else:
            progress = (epoch - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs)
            factor = self.eta_min / self.base_lrs[0] + 0.5 * (1 - self.eta_min / self.base_lrs[0]) * (
                1 + math.cos(math.pi * progress)
            )
        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            pg["lr"] = base_lr * factor


# ──────────────────────────────────────────────────────────────────────────────
# 10. PER-DEFECT-TYPE METRICS
# ──────────────────────────────────────────────────────────────────────────────
def compute_per_defect_metrics(all_probs, all_masks, all_labels, all_fnames,
                                val_manifest: Path,
                                thr: float, interior: np.ndarray, min_area: int) -> pd.DataFrame:
    df_manifest = pd.read_csv(val_manifest)
    fname_to_type = {}
    for _, row in df_manifest.iterrows():
        fname = Path(row["image_path"]).name
        fname_to_type[fname] = Path(row["image_path"]).parent.name

    type_records = {}
    for prob_np, mask_np, label, fname in zip(all_probs, all_masks, all_labels, all_fnames):
        defect_type = fname_to_type.get(fname, "unknown")
        d, iou = per_image_hard_dice_iou(prob_np, mask_np, thr, interior, min_area)
        pred_bin = (prob_np > thr).astype(np.uint8)
        pred_filt = apply_border_and_cc_filter(pred_bin, interior, min_area)
        fp = int(pred_filt.sum() > 0 and label == 0)
        empty = int(pred_filt.sum() == 0 and label == 1)

        if defect_type not in type_records:
            type_records[defect_type] = {
                "n_images": 0, "dice_sum": 0.0, "iou_sum": 0.0,
                "dice0_count": 0, "fp_count": 0, "empty_count": 0,
                "is_defective": label
            }
        type_records[defect_type]["n_images"] += 1
        if label == 1:
            type_records[defect_type]["dice_sum"] += d
            type_records[defect_type]["iou_sum"] += iou
            if d == 0:
                type_records[defect_type]["dice0_count"] += 1
            type_records[defect_type]["empty_count"] += empty
        else:
            type_records[defect_type]["fp_count"] += fp

    rows = []
    for t, r in type_records.items():
        n = r["n_images"]
        is_def = r["is_defective"]
        rows.append({
            "defect_type": t,
            "n_images": n,
            "mean_dice": round(r["dice_sum"] / max(n, 1), 4) if is_def else None,
            "mean_iou": round(r["iou_sum"] / max(n, 1), 4) if is_def else None,
            "n_dice_zero": r["dice0_count"] if is_def else None,
            "n_empty_pred": r["empty_count"] if is_def else None,
            "fp_rate": round(r["fp_count"] / max(n, 1), 4) if not is_def else None,
        })
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────────
# 11. CHECK (0,1) ARTIFACT
# ──────────────────────────────────────────────────────────────────────────────
def check_border_artifact(all_probs, all_labels):
    """Check if normal images still peak at exactly (x=0, y=1) after v2 training."""
    normal_probs = [p for p, lbl in zip(all_probs, all_labels) if lbl == 0]
    if not normal_probs:
        print("  No normal images to check for border artifact.")
        return
    maxima = []
    for p in normal_probs:
        idx = np.unravel_index(np.argmax(p), p.shape)
        maxima.append((p[idx], idx[1], idx[0]))  # (prob, x, y)

    maxima.sort(reverse=True, key=lambda x: x[0])
    print(f"\n  Border Artifact Check on {len(normal_probs)} normal images:")
    print(f"  {'Max Prob':>10} {'x':>5} {'y':>5}")
    for prob, x, y in maxima[:10]:
        marker = " ← ARTIFACT" if (x == 0 and y == 1) else ""
        print(f"  {prob:10.4f} {x:5d} {y:5d}{marker}")

    # Count how many still peak at (0,1)
    n_at_01 = sum(1 for prob, x, y in maxima if x == 0 and y == 1)
    print(f"  Images with peak at (0,1): {n_at_01}/{len(normal_probs)}")
    if n_at_01 == 0:
        print("  ✓ Artifact (0,1) is GONE — reflection padding and augmentation fixed it.")
    else:
        print(f"  ⚠ Artifact still present in {n_at_01} images.")


# ──────────────────────────────────────────────────────────────────────────────
# 12. RECOMPUTE V1 METRICS FAIRLY
# ──────────────────────────────────────────────────────────────────────────────
def recompute_v1_fair(val_manifest: Path, border_margin: int, image_size: int,
                       device, use_amp: bool, out_dir: Path) -> dict | None:
    """
    Load v1 best.ckpt (resnet18, 256px) and re-evaluate on validation with
    the same border+CC filter used by v2 for a fair comparison.
    """
    import segmentation_models_pytorch as smp

    v1_ckpt = PROJECT_ROOT / "ml/outputs/bottle/unet/best.ckpt"
    if not v1_ckpt.is_file():
        print(f"  V1 checkpoint not found at {v1_ckpt}, skipping fair recompute.")
        return None

    print("\n  Recomputing V1 metrics with border+CC filter (fair comparison)...")
    model_v1 = smp.Unet(encoder_name="resnet18", encoder_weights=None, in_channels=3, classes=1)
    ckpt = torch.load(v1_ckpt, map_location=device)
    model_v1.load_state_dict(ckpt["model_state_dict"])
    model_v1 = model_v1.to(device)
    model_v1.eval()

    # V1 uses plain PIL resize, no augmentation (reproduce exactly)
    val_ds_v1 = _V1ValDataset(val_manifest, image_size=256)
    val_loader_v1 = DataLoader(val_ds_v1, batch_size=8, shuffle=False, num_workers=0,
                                collate_fn=collate_fn_v1)
    all_probs_v1, all_masks_v1, all_labels_v1 = [], [], []
    with torch.no_grad():
        for imgs_v1, masks_v1, labels_v1 in val_loader_v1:
            imgs_v1 = imgs_v1.to(device)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits_v1 = model_v1(imgs_v1)
            probs_v1 = torch.sigmoid(logits_v1).cpu()
            for i in range(imgs_v1.size(0)):
                all_probs_v1.append(probs_v1[i, 0].numpy())
                all_masks_v1.append(masks_v1[i, 0].numpy())
                all_labels_v1.append(labels_v1[i].item())

    H = W = 256
    interior_v1 = make_interior_mask(H, W, border_margin)
    # Use v1 threshold = 0.50, min_area = 150 at 256px
    min_area_v1 = 150
    m_v1 = compute_val_metrics_v2(all_probs_v1, all_masks_v1, all_labels_v1,
                                   thr=0.50, interior=interior_v1, min_area=min_area_v1)

    # Also try best threshold from v1 diagnostics (0.30)
    m_v1_30 = compute_val_metrics_v2(all_probs_v1, all_masks_v1, all_labels_v1,
                                      thr=0.30, interior=interior_v1, min_area=min_area_v1)

    print(f"  V1 @thr=0.50 border+CC: Dice={m_v1['dice_def']:.4f} IoU={m_v1['iou_def']:.4f} FP={m_v1['fp_rate']:.4f}")
    print(f"  V1 @thr=0.30 border+CC: Dice={m_v1_30['dice_def']:.4f} IoU={m_v1_30['iou_def']:.4f} FP={m_v1_30['fp_rate']:.4f}")
    return {"thr050": m_v1, "thr030": m_v1_30}


class _V1ValDataset(Dataset):
    """Reproduce v1 preprocessing exactly: PIL resize + normalize [0,1]."""
    def __init__(self, manifest_path, image_size=256):
        self.df = pd.read_csv(manifest_path)
        self.sz = image_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        with Image.open(row["image_path"]) as img:
            img = img.convert("RGB").resize((self.sz, self.sz), Image.BILINEAR)
            img_np = np.array(img, dtype=np.float32) / 255.0
        mask_val = row.get("mask_path", "")
        if pd.isna(mask_val) or str(mask_val).strip() == "":
            mask_np = np.zeros((self.sz, self.sz), dtype=np.float32)
        else:
            with Image.open(str(mask_val).strip()) as m:
                m = m.convert("L").resize((self.sz, self.sz), Image.NEAREST)
                mask_np = (np.array(m, dtype=np.float32) > 0).astype(np.float32)
        img_t = torch.from_numpy(img_np.transpose(2, 0, 1)).float()
        mask_t = torch.from_numpy(mask_np).unsqueeze(0).float()
        label = int(row["is_anomaly"])
        return img_t, mask_t, label


def collate_fn_v1(batch):
    imgs, masks, labels = zip(*batch)
    return torch.stack(imgs), torch.stack(masks), torch.tensor(labels, dtype=torch.long)


# ──────────────────────────────────────────────────────────────────────────────
# 13. GALLERY
# ──────────────────────────────────────────────────────────────────────────────
def generate_gallery_v2(all_probs, all_masks, all_labels, all_fnames,
                         thr: float, interior: np.ndarray, min_area: int,
                         out_dir: Path, suffix: str = ""):
    gallery_dir = out_dir / "gallery"
    gallery_dir.mkdir(exist_ok=True)

    samples = []
    for prob_np, mask_np, label, fname in zip(all_probs, all_masks, all_labels, all_fnames):
        pred_filt = apply_border_and_cc_filter((prob_np > thr).astype(np.uint8), interior, min_area)
        d, _ = per_image_hard_dice_iou(prob_np, mask_np, thr, interior, min_area)
        # Sort: worst defective first (lowest Dice), then FP normals
        sort_key = d if label == 1 else 1.0 + (1.0 - float(pred_filt.sum() == 0))
        samples.append((sort_key, prob_np, mask_np, pred_filt, label, fname))

    samples.sort(key=lambda s: s[0])
    top_n = min(8, len(samples))
    selected = samples[:top_n]

    fig, axes = plt.subplots(top_n, 4, figsize=(14, 3.2 * top_n))
    if top_n == 1:
        axes = axes[np.newaxis, :]

    for r, (_, prob_np, mask_np, pred_np, lbl, fname) in enumerate(selected):
        # Reconstruct image from normalized tensor (approximate: use prob heatmap as proxy)
        for c, (data, title, cmap) in enumerate([
            (prob_np, f"{fname} ({'D' if lbl == 1 else 'N'})", "plasma"),
            (mask_np, "True Mask", "gray"),
            (pred_np.astype(float), f"Pred@{thr:.2f}+CC", "gray"),
            (prob_np * (mask_np > 0.5), "Prob×TrueMask", "plasma"),
        ]):
            ax = axes[r, c]
            ax.imshow(np.clip(data, 0, 1), cmap=cmap)
            ax.set_title(title, fontsize=7)
            ax.axis("off")

    fig.suptitle(f"V2 Gallery – Worst Predictions (thr={thr:.2f} min_area={min_area}){suffix}", fontsize=10)
    plt.tight_layout()
    out_path = gallery_dir / f"validation_gallery{suffix}.png"
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  Gallery saved → {out_path}")
    return out_path


# ──────────────────────────────────────────────────────────────────────────────
# 14. REGISTRY
# ──────────────────────────────────────────────────────────────────────────────
def update_registry(reg_path: Path, entry: dict):
    reg_path.parent.mkdir(parents=True, exist_ok=True)
    registry = []
    if reg_path.is_file():
        try:
            registry = json.loads(reg_path.read_text(encoding="utf-8"))
            if not isinstance(registry, list):
                registry = []
        except Exception:
            registry = []
    # Do NOT overwrite v1 (version="1.0"); replace only if same name AND same version
    updated = [
        r for r in registry
        if not (r.get("name") == entry["name"] and r.get("version") == entry["version"])
    ]
    updated.append(entry)
    reg_path.write_text(json.dumps(updated, indent=2), encoding="utf-8")
    print(f"  Registry updated → {reg_path}")


# ──────────────────────────────────────────────────────────────────────────────
# 15. TRAIN ONE SEED
# ──────────────────────────────────────────────────────────────────────────────
def train_one_seed(seed: int, cfg: dict, device, cuda_avail: bool,
                   dev_name: str, total_vram_mib: float,
                   use_amp: bool) -> dict:
    """Train one seed, return metrics dict."""
    import segmentation_models_pytorch as smp

    image_size = cfg["image_size"]
    border_margin = cfg["border_margin_px"]
    H = W = image_size
    interior = make_interior_mask(H, W, border_margin)

    out_dir = PROJECT_ROOT / cfg["out_dir"] / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*65}")
    print(f"SEED {seed} | image_size={image_size} | encoder={cfg['encoder']} | margin={border_margin}px")
    print(f"{'='*65}")

    # Seed
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if cuda_avail:
        torch.cuda.manual_seed_all(seed)

    # Data
    manifest_train = PROJECT_ROOT / "ml/manifests/manifest_bottle_train.csv"
    manifest_val = PROJECT_ROOT / "ml/manifests/manifest_bottle_val.csv"
    manifest_test = PROJECT_ROOT / "ml/manifests/manifest_bottle_test.csv"

    train_ds = BottleDatasetV2(manifest_train, image_size=image_size, is_train=True)
    val_ds = BottleDatasetV2(manifest_val, image_size=image_size, is_train=False)
    test_ds = BottleDatasetV2(manifest_test, image_size=image_size, is_train=False)

    # Weighted sampler
    train_labels = train_ds.df["is_anomaly"].astype(int).values
    class_counts = np.bincount(train_labels)
    class_weights = 1.0 / np.maximum(class_counts, 1)
    sample_weights = torch.tensor(class_weights[train_labels], dtype=torch.float)
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    bs = cfg["batch_size"]
    fallback_bs = cfg["fallback_batch_size"]
    train_loader = DataLoader(train_ds, batch_size=bs, sampler=sampler,
                               num_workers=0, pin_memory=cuda_avail, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False,
                             num_workers=0, pin_memory=cuda_avail, collate_fn=collate_fn)

    # Model
    try:
        model = smp.Unet(encoder_name=cfg["encoder"], encoder_weights="imagenet",
                          in_channels=3, classes=1)
        print(f"  Model: U-Net ({cfg['encoder']}, ImageNet pretrained)")
    except Exception as e:
        print(f"  Pretrained download failed ({e}). Using random init.")
        model = smp.Unet(encoder_name=cfg["encoder"], encoder_weights=None,
                          in_channels=3, classes=1)
    model = model.to(device)

    # Loss, optimizer, scheduler
    criterion = CombinedLoss(bce_w=cfg["loss_bce_weight"],
                              dice_w=cfg["loss_dice_weight"],
                              focal_w=cfg["loss_focal_weight"])
    optimizer = optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = WarmupCosineScheduler(optimizer, warmup_epochs=cfg["warmup_epochs"],
                                       total_epochs=cfg["epochs"])
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_score = -float("inf")
    best_ckpt = out_dir / "best.ckpt"
    patience_counter = 0
    peak_vram_mib = 0.0

    history = {"epoch": [], "train_loss": [], "val_dice": [], "val_fp": [],
               "score": [], "epoch_time_s": []}

    thresholds_to_print = [0.30, 0.35, 0.40, 0.45, 0.50]

    print(f"\n{'Ep':>4} | {'TrLoss':>7} | {'Dice@30':>7} | {'Dice@40':>7} | {'Dice@50':>7} | "
          f"{'FP@35':>6} | {'Score':>6} | {'Time':>5} | {'VRAM':>6}")
    print("-" * 90)

    try:
        for epoch in range(1, cfg["epochs"] + 1):
            t0 = time.time()
            if cuda_avail:
                torch.cuda.reset_peak_memory_stats(0)

            # Training
            model.train()
            train_loss = 0.0
            for imgs, masks, _, _fnames in train_loader:
                imgs = imgs.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)
                optimizer.zero_grad()
                try:
                    with torch.amp.autocast("cuda", enabled=use_amp):
                        logits = model(imgs)
                        loss = criterion(logits, masks)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    train_loss += loss.item() * imgs.size(0)
                except torch.cuda.OutOfMemoryError:
                    print(f"\n[OOM] at batch_size={bs}. Try batch_size={fallback_bs}.")
                    raise
            train_loss /= len(train_loader.dataset)

            scheduler.step(epoch)

            # Validation with TTA
            all_probs, all_masks_v, all_labels_v, all_fnames_v = tta_predict_loader(
                model, val_loader, device, use_amp)

            # Print Dice at multiple thresholds
            dice_per_thr = {}
            for thr in thresholds_to_print:
                m = compute_val_metrics_v2(all_probs, all_masks_v, all_labels_v,
                                           thr, interior, cfg["min_component_area_start"])
                dice_per_thr[thr] = m

            # Use thr=0.35 for epoch score (will be tuned after training)
            m35 = dice_per_thr[0.35]
            score = m35["score"]

            if cuda_avail:
                cur_peak = torch.cuda.max_memory_allocated(0) / (1024 ** 2)
                peak_vram_mib = max(peak_vram_mib, cur_peak)

            ep_time = time.time() - t0
            if epoch == 1:
                print(f"  [Epoch 1 Profiling] Time: {ep_time:.1f}s | Peak VRAM: {peak_vram_mib:.0f} MiB")

            d30 = dice_per_thr[0.30]["dice_def"]
            d40 = dice_per_thr[0.40]["dice_def"]
            d50 = dice_per_thr[0.50]["dice_def"]
            fp35 = m35["fp_rate"]

            history["epoch"].append(epoch)
            history["train_loss"].append(round(train_loss, 4))
            history["val_dice"].append(round(m35["dice_def"], 4))
            history["val_fp"].append(round(fp35, 4))
            history["score"].append(round(score, 4))
            history["epoch_time_s"].append(round(ep_time, 1))

            print(f"{epoch:4d} | {train_loss:7.4f} | {d30:7.4f} | {d40:7.4f} | {d50:7.4f} | "
                  f"{fp35:6.4f} | {score:6.4f} | {ep_time:4.0f}s | {peak_vram_mib:5.0f}M")

            # Checkpoint selection (must train min_epochs before stopping)
            if score > best_score:
                best_score = score
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "score": score,
                    "image_size": image_size,
                    "encoder": cfg["encoder"],
                    "seed": seed,
                }, best_ckpt)
                patience_counter = 0
                print(f"       → [Saved best.ckpt epoch={epoch} score={score:.4f}]")
            else:
                patience_counter += 1
                if epoch >= cfg["min_epochs"] and patience_counter >= cfg["patience"]:
                    print(f"\nEarly stopping at epoch {epoch} (patience={cfg['patience']}).")
                    break

    except torch.cuda.OutOfMemoryError as oom:
        print(f"\n{'!'*70}\nCUDA OOM! Try --config with batch_size={fallback_bs}\n{'!'*70}")
        raise oom

    # Load best checkpoint for final eval
    print(f"\n  Loading best checkpoint (epoch {torch.load(best_ckpt)['epoch']})...")
    ckpt = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    # Final TTA pass for tuning
    all_probs, all_masks_v, all_labels_v, all_fnames_v = tta_predict_loader(
        model, val_loader, device, use_amp)

    # Tune threshold + min_area on validation only
    best_thr, best_min_area, sweep_df = tune_threshold_and_area(
        all_probs, all_masks_v, all_labels_v,
        image_size=image_size, border_margin=border_margin,
        min_area_start=cfg["min_component_area_start"], out_dir=out_dir)

    # Final metrics with tuned params
    final_interior = make_interior_mask(H, W, border_margin)
    final_m = compute_val_metrics_v2(all_probs, all_masks_v, all_labels_v,
                                      best_thr, final_interior, best_min_area)

    # Per-defect-type table
    per_defect_df = compute_per_defect_metrics(
        all_probs, all_masks_v, all_labels_v, all_fnames_v,
        manifest_val, best_thr, final_interior, best_min_area)
    per_defect_df.to_csv(out_dir / "per_defect_metrics.csv", index=False)
    print("\n  Per-defect-type metrics:")
    print(per_defect_df.to_string(index=False))

    # Border artifact check
    check_border_artifact(all_probs, all_labels_v)

    # Gallery
    generate_gallery_v2(all_probs, all_masks_v, all_labels_v, all_fnames_v,
                         best_thr, final_interior, best_min_area, out_dir, suffix=f"_seed{seed}")

    # Learning curve
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["epoch"], history["train_loss"], label="Train Loss")
    axes[0].set_title("Train Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[1].plot(history["epoch"], history["val_dice"], label="Val Dice@35", color="green")
    axes[1].plot(history["epoch"], history["val_fp"], label="FP Rate@35", color="red")
    axes[1].set_title("Val Dice & FP Rate (thr=0.35)")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(out_dir / "learning_curve.png", dpi=100)
    plt.close(fig)

    result = {
        "seed": seed,
        "best_epoch": int(ckpt["epoch"]),
        "best_thr": best_thr,
        "best_min_area": best_min_area,
        "dice_def": final_m["dice_def"],
        "iou_def": final_m["iou_def"],
        "fp_rate": final_m["fp_rate"],
        "score": final_m["score"],
        "peak_vram_mib": round(peak_vram_mib, 1),
        "per_defect": per_defect_df.to_dict("records"),
        "ckpt_path": str(best_ckpt),
        "out_dir": str(out_dir),
        "epochs_trained": len(history["epoch"]),
    }
    return result


# ──────────────────────────────────────────────────────────────────────────────
# 16. MAIN
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="U-Net V2 Bottle Training (Phase 4A-v2)")
    parser.add_argument("--config", type=str, default=None, help="Path to JSON config file")
    args = parser.parse_args()

    cfg = load_config(args.config)

    print("=" * 65)
    print("V2 CONFIG:")
    for k, v in cfg.items():
        print(f"  {k}: {v}")
    print("=" * 65)

    device, cuda_avail, dev_name, total_vram_mib = setup_device()
    use_amp = cuda_avail and cfg["amp"]

    seeds = cfg["seeds"]
    all_seed_results = []

    out_dir_root = PROJECT_ROOT / cfg["out_dir"]
    out_dir_root.mkdir(parents=True, exist_ok=True)

    for seed in seeds:
        result = train_one_seed(
            seed=seed, cfg=cfg, device=device, cuda_avail=cuda_avail,
            dev_name=dev_name, total_vram_mib=total_vram_mib, use_amp=use_amp
        )
        all_seed_results.append(result)

    # ── Aggregate across seeds ──
    print("\n" + "=" * 65)
    print("MULTI-SEED RESULTS SUMMARY")
    print("=" * 65)
    dices = [r["dice_def"] for r in all_seed_results]
    ious = [r["iou_def"] for r in all_seed_results]
    fps = [r["fp_rate"] for r in all_seed_results]
    scores = [r["score"] for r in all_seed_results]

    print(f"{'Metric':<20} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
    print("-" * 55)
    for name, vals in [("Dice (defective)", dices), ("IoU (defective)", ious),
                        ("FP Rate (normal)", fps), ("Score", scores)]:
        print(f"{name:<20} {np.mean(vals):8.4f} {np.std(vals):8.4f} {np.min(vals):8.4f} {np.max(vals):8.4f}")

    # Find best seed
    best_seed_result = max(all_seed_results, key=lambda r: r["score"])
    best_seed = best_seed_result["seed"]
    print(f"\n  Best seed: {best_seed} (score={best_seed_result['score']:.4f})")

    # Save aggregated summary
    summary = {
        "config": cfg,
        "seeds": seeds,
        "mean_dice": float(np.mean(dices)),
        "std_dice": float(np.std(dices)),
        "mean_iou": float(np.mean(ious)),
        "std_iou": float(np.std(ious)),
        "mean_fp_rate": float(np.mean(fps)),
        "std_fp_rate": float(np.std(fps)),
        "mean_score": float(np.mean(scores)),
        "std_score": float(np.std(scores)),
        "best_seed": best_seed,
        "per_seed": all_seed_results,
        "device": dev_name,
        "total_vram_mib": total_vram_mib,
    }
    summary_path = out_dir_root / f"summary_{cfg['version']}.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"\n  Summary saved → {summary_path}")

    # ── Recompute V1 metrics fairly ──
    manifest_val = PROJECT_ROOT / "ml/manifests/manifest_bottle_val.csv"
    border_margin = cfg["border_margin_px"]
    v1_fair = recompute_v1_fair(manifest_val, border_margin=border_margin,
                                 image_size=256, device=device,
                                 use_amp=use_amp, out_dir=out_dir_root)

    # ── Print final comparison table ──
    print("\n" + "=" * 65)
    print("COMPARISON TABLE (Validation Set)")
    print("=" * 65)
    print(f"{'Metric':<28} {'V1 (naive)':>12} {'V1 (fair@50)':>13} {'V1 (fair@30)':>13} {'V2-'+cfg['image_size'].__str__()+'px':>12}")
    print("-" * 75)
    rows = [
        ("Threshold", "0.50", f"0.50", "0.30", f"tuned({best_seed_result['best_thr']:.2f})"),
        ("Min CC Area", "none", "150px", "150px", f"{best_seed_result['best_min_area']}px"),
        ("Dice (defective)", "0.5034*",
         f"{v1_fair['thr050']['dice_def']:.4f}" if v1_fair else "N/A",
         f"{v1_fair['thr030']['dice_def']:.4f}" if v1_fair else "N/A",
         f"{np.mean(dices):.4f}±{np.std(dices):.4f}"),
        ("IoU (defective)", "0.3379*",
         f"{v1_fair['thr050']['iou_def']:.4f}" if v1_fair else "N/A",
         f"{v1_fair['thr030']['iou_def']:.4f}" if v1_fair else "N/A",
         f"{np.mean(ious):.4f}±{np.std(ious):.4f}"),
        ("FP Rate (normal)", "0.0000",
         f"{v1_fair['thr050']['fp_rate']:.4f}" if v1_fair else "N/A",
         f"{v1_fair['thr030']['fp_rate']:.4f}" if v1_fair else "N/A",
         f"{np.mean(fps):.4f}±{np.std(fps):.4f}"),
    ]
    for row in rows:
        print(f"{row[0]:<28} {row[1]:>12} {row[2]:>13} {row[3]:>13} {row[4]:>12}")
    print("* V1 naive: per-image Dice from diagnostics (batch soft-Dice = 0.2865 was broken)")

    # ── Smoke test on 2 test images (no metrics) ──
    print("\n── Smoke Test: 2 test images (no metrics) ──")
    best_ckpt_path = Path(best_seed_result["ckpt_path"])
    import segmentation_models_pytorch as smp
    model_smoke = smp.Unet(encoder_name=cfg["encoder"], encoder_weights=None,
                            in_channels=3, classes=1)
    ckpt_smoke = torch.load(best_ckpt_path, map_location=device)
    model_smoke.load_state_dict(ckpt_smoke["model_state_dict"])
    model_smoke = model_smoke.to(device)
    model_smoke.eval()

    smoke_loader = DataLoader(BottleDatasetV2(manifest_val.parent.parent /
                                               "manifests/manifest_bottle_test.csv",
                                               image_size=cfg["image_size"], is_train=False),
                               batch_size=2, shuffle=False, num_workers=0, collate_fn=collate_fn)
    smoke_dir = out_dir_root / "test_smoke"
    smoke_dir.mkdir(exist_ok=True)
    with torch.no_grad():
        imgs_t, _, _, fnames_t = next(iter(smoke_loader))
        probs_t = tta_predict(model_smoke, imgs_t, device, use_amp)
        preds_t = (probs_t > best_seed_result["best_thr"]).float()
        for i in range(min(2, preds_t.size(0))):
            pred_map = (preds_t[i, 0].numpy() * 255).astype(np.uint8)
            out_png = smoke_dir / f"smoke_{i}_{fnames_t[i]}"
            Image.fromarray(pred_map).save(out_png)
            print(f"  Smoke prediction → {out_png}")

    # ── Registry update ──
    reg_path = PROJECT_ROOT / "ml/models/registry.json"
    reg_entry = {
        "name": "bottle_unet_v2",
        "version": cfg["version"],
        "category": "bottle",
        "protocol": "Protocol B custom split",
        "image_size": cfg["image_size"],
        "encoder": cfg["encoder"],
        "batch_size": cfg["batch_size"],
        "amp_enabled": use_amp,
        "tta": cfg["tta"],
        "device_gpu": dev_name,
        "seeds": seeds,
        "best_seed": best_seed,
        "selected_threshold": best_seed_result["best_thr"],
        "min_component_area": best_seed_result["best_min_area"],
        "border_margin_px": cfg["border_margin_px"],
        "validation_metrics": {
            "mean_dice_defective": float(np.mean(dices)),
            "std_dice_defective": float(np.std(dices)),
            "mean_iou_defective": float(np.mean(ious)),
            "std_iou_defective": float(np.std(ious)),
            "mean_fp_rate_normal": float(np.mean(fps)),
            "std_fp_rate_normal": float(np.std(fps)),
        },
        "artifacts": {
            "best_seed_ckpt": str(best_ckpt_path),
            "summary": str(summary_path),
        }
    }
    update_registry(reg_path, reg_entry)

    print("\n" + "=" * 65)
    print("ALL DONE — V2 TRAINING COMPLETE")
    print(f"  Outputs: {out_dir_root}")
    print(f"  Registry: {reg_path}")
    print("=" * 65)


if __name__ == "__main__":
    main()
