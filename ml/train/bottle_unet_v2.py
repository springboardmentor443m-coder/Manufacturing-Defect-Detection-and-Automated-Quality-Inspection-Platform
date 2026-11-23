#!/usr/bin/env python
"""
ml/train/bottle_unet_v2.py
Phase 4A-v2 – U-Net v2 training for the 'bottle' category.

Key Features & Protocol B Compliance:
  1. Per-image hard-thresholded Dice on defective validation images as the selection metric
     (NOT batch-averaged soft Dice).
  2. Border margin of 4px (at 256px, scaled proportionally for other sizes) ignored
     when computing components, metrics, and false-positive decisions.
  3. Minimum connected-component area filter (starts at 150px at 256x256), tuned on validation only.
  4. Selection score: mean per-image Dice (defective) - 0.5 * FP rate (normal).
     A normal image counts as FP ONLY if it has a connected component above min_area
     outside the border margin.
  5. Reflection padding (cv2.BORDER_REFLECT_101) in all padding augmentations to eliminate
     the boundary zero-padding artifact.
  6. ResNet34 encoder (ImageNet pretrained), batch size 8 with automatic fallback to 4 on CUDA OOM.
  7. Configurable loss: BCEWithLogits + SoftDice + Focal (or Tversky with configurable beta).
  8. AdamW (lr 3e-4, weight decay 1e-4) with 3 warmup epochs followed by Cosine Annealing.
  9. Up to 100 epochs, early stopping patience 20, minimum 30 epochs (or user-overridden epochs).
 10. WeightedRandomSampler ensuring ~50% defective samples per batch.
 11. Independent seed execution: each seed saves its own best.ckpt, config.json, learning_curve.png,
     val_results.csv, and per_defect_metrics.csv in ml/outputs/bottle/unet_v2/seed_<seed>/.
 12. Test manifest is NEVER used for training/tuning/model selection; only a 2-image smoke test
     with zero metrics recorded is run after training.

Usage:
    python ml/train/bottle_unet_v2.py --config ml/configs/bottle_unet_v2_256.json --seed 42
"""

import argparse
import hashlib
import json
import math
import random
import sys

# Ensure UTF-8 output on Windows consoles/pipes to prevent cp1252 charmap encoding errors
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

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
    "loss_type": "combined",
    "loss_bce_weight": 0.3,
    "loss_dice_weight": 0.5,
    "loss_focal_weight": 0.2,
    "tversky_alpha": 0.3,
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
        cp = Path(config_path)
        if not cp.is_file():
            cp = PROJECT_ROOT / config_path
        if cp.is_file():
            with open(cp, "r") as f:
                override = json.load(f)
            cfg.update(override)
        else:
            print(f"Warning: Config file {config_path} not found. Using defaults.")
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
        print("DEVICE CONFIGURATION: GPU")
        print(f"  GPU Name        : {dev_name}")
        print(f"  Total VRAM      : {total_vram_mib:.0f} MiB")
        print(f"  CUDA Version    : {getattr(torch.version, 'cuda', 'unknown')}")
        print("=" * 65)
    else:
        device = torch.device("cpu")
        dev_name = "CPU"
        total_vram_mib = 0.0
        print("WARNING: CUDA not available. Falling back to CPU mode.")
    return device, cuda, dev_name, total_vram_mib


# ──────────────────────────────────────────────────────────────────────────────
# 2. DATA AUGMENTATION (REFLECTION PADDING)
# ──────────────────────────────────────────────────────────────────────────────
def build_augmentation(image_size: int, is_train: bool):
    """
    Builds data augmentation pipeline using Albumentations.
    Uses cv2.BORDER_REFLECT_101 for any operation that pads,
    eliminating constant zero-border artifacts.
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
                shift_limit=0.05,
                scale_limit=0.10,
                rotate_limit=15,
                border_mode=cv2.BORDER_REFLECT_101,
                p=0.5
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
    """Protocol B Dataset with Albumentations augmentation & reflection padding."""

    def __init__(self, manifest_path: Path, image_size: int, is_train: bool = False):
        self.df = pd.read_csv(manifest_path)
        self.image_size = image_size
        self.transform = build_augmentation(image_size, is_train)
        for col in ("image_path", "is_anomaly"):
            if col not in self.df.columns:
                raise ValueError(f"Manifest {manifest_path} missing column '{col}'")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int):
        row = self.df.iloc[index]

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
        img_aug = augmented["image"]
        mask_aug = augmented["mask"]

        img_t = torch.from_numpy(img_aug.transpose(2, 0, 1)).float()
        mask_t = torch.from_numpy(mask_aug.astype(np.float32)).unsqueeze(0)
        label = int(row["is_anomaly"])
        fname = Path(row["image_path"]).name
        defect_type = str(row.get("defect_type", Path(row["image_path"]).parent.name))
        return img_t, mask_t, label, fname, defect_type


def collate_fn(batch):
    imgs, masks, labels, fnames, defect_types = zip(*batch)
    return (torch.stack(imgs), torch.stack(masks),
            torch.tensor(labels, dtype=torch.long), list(fnames), list(defect_types))


# ──────────────────────────────────────────────────────────────────────────────
# 4. LOSS FUNCTIONS (CONFIGURABLE: BCE+DICE+FOCAL OR TVERSKY)
# ──────────────────────────────────────────────────────────────────────────────
def soft_dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    p = probs.reshape(probs.size(0), -1)
    t = targets.reshape(targets.size(0), -1)
    inter = (p * t).sum(dim=1)
    union = p.sum(dim=1) + t.sum(dim=1)
    dice = 1.0 - (2.0 * inter + eps) / (union + eps)
    return dice.mean()


def focal_loss(logits: torch.Tensor, targets: torch.Tensor, gamma: float = 2.0) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    pt = torch.where(targets > 0.5, probs, 1.0 - probs)
    fl = ((1.0 - pt) ** gamma) * bce
    return fl.mean()


def tversky_loss(logits: torch.Tensor, targets: torch.Tensor,
                 alpha: float = 0.3, beta: float = 0.7, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    p = probs.reshape(probs.size(0), -1)
    t = targets.reshape(targets.size(0), -1)
    true_pos = (p * t).sum(dim=1)
    false_neg = ((1.0 - p) * t).sum(dim=1)
    false_pos = (p * (1.0 - t)).sum(dim=1)
    tversky = (true_pos + eps) / (true_pos + alpha * false_pos + beta * false_neg + eps)
    return (1.0 - tversky).mean()


class ConfiguredLoss(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.loss_type = cfg.get("loss_type", "combined")
        self.bce_w = cfg.get("loss_bce_weight", 0.3)
        self.dice_w = cfg.get("loss_dice_weight", 0.5)
        self.focal_w = cfg.get("loss_focal_weight", 0.2)
        self.tversky_alpha = cfg.get("tversky_alpha", 0.3)
        self.tversky_beta = cfg.get("tversky_beta", 0.7)
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.loss_type == "tversky":
            return (self.bce_w * self.bce(logits, targets) +
                    (1.0 - self.bce_w) * tversky_loss(logits, targets,
                                                      alpha=self.tversky_alpha,
                                                      beta=self.tversky_beta))
        return (self.bce_w * self.bce(logits, targets) +
                self.dice_w * soft_dice_loss(logits, targets) +
                self.focal_w * focal_loss(logits, targets))


# ──────────────────────────────────────────────────────────────────────────────
# 5. BORDER MARGIN & CONNECTED COMPONENT FILTER
# ──────────────────────────────────────────────────────────────────────────────
def make_interior_mask(H: int, W: int, margin: int) -> np.ndarray:
    """Returns boolean mask that is True inside the border margin."""
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
    2. Filter out connected components smaller than min_area.
    """
    from scipy import ndimage as ndi
    result = pred_bin.copy()
    result[~interior] = 0

    labeled, n_cc = ndi.label(result)
    if n_cc > 0:
        sizes = ndi.sum(result, labeled, range(1, n_cc + 1))
        for cc_idx, sz in enumerate(sizes, start=1):
            if sz < min_area:
                result[labeled == cc_idx] = 0
    return result


# ──────────────────────────────────────────────────────────────────────────────
# 6. PER-IMAGE HARD-THRESHOLDED METRICS (SELECTION RULE)
# ──────────────────────────────────────────────────────────────────────────────
def per_image_hard_dice_iou(prob_np: np.ndarray, mask_np: np.ndarray,
                             thr: float, interior: np.ndarray,
                             min_area: int) -> tuple[float, float, bool]:
    """
    Hard-threshold predicted probability map, apply border margin & CC filter,
    then compute per-image Dice and IoU against true mask (also masked to interior).
    Returns (dice, iou, is_empty_prediction).
    """
    pred_bin = (prob_np > thr).astype(np.uint8)
    pred_filt = apply_border_and_cc_filter(pred_bin, interior, min_area)

    true_filt = (mask_np > 0.5).astype(np.uint8)
    true_filt[~interior] = 0

    inter = (pred_filt & true_filt).sum()
    pred_sum = pred_filt.sum()
    true_sum = true_filt.sum()
    union = pred_sum + true_sum

    is_empty = bool(pred_sum == 0)

    if union > 0:
        dice = float(2.0 * inter / (union + 1e-6))
    else:
        dice = 1.0

    denom_iou = pred_sum + true_sum - inter
    if denom_iou > 0:
        iou = float(inter / (denom_iou + 1e-6))
    else:
        iou = 1.0

    return dice, iou, is_empty


def compute_val_metrics_v2(all_probs: list[np.ndarray],
                            all_masks: list[np.ndarray],
                            all_labels: list[int],
                            thr: float,
                            interior: np.ndarray,
                            min_area: int) -> dict:
    dice_def, iou_def = [], []
    fp_normal = []
    empty_preds = 0

    for prob_np, mask_np, label in zip(all_probs, all_masks, all_labels):
        if label == 1:
            d, iou, is_empty = per_image_hard_dice_iou(prob_np, mask_np, thr, interior, min_area)
            dice_def.append(d)
            iou_def.append(iou)
            if is_empty:
                empty_preds += 1
        else:
            pred_bin = (prob_np > thr).astype(np.uint8)
            pred_filt = apply_border_and_cc_filter(pred_bin, interior, min_area)
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
        "empty_preds": empty_preds,
        "n_defective": len(dice_def),
        "n_normal": len(fp_normal),
    }


# ──────────────────────────────────────────────────────────────────────────────
# 7. TTA (TEST-TIME AUGMENTATION) INFERENCE
# ──────────────────────────────────────────────────────────────────────────────
def tta_predict(model, imgs: torch.Tensor, device, use_amp: bool) -> torch.Tensor:
    model.eval()
    variants = [
        imgs,
        torch.flip(imgs, dims=[3]),
        torch.flip(imgs, dims=[2]),
        torch.flip(imgs, dims=[2, 3]),
    ]
    probs_sum = None
    with torch.no_grad():
        for i, var in enumerate(variants):
            var = var.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(var)
            probs = torch.sigmoid(logits).cpu()
            if i == 1:
                probs = torch.flip(probs, dims=[3])
            elif i == 2:
                probs = torch.flip(probs, dims=[2])
            elif i == 3:
                probs = torch.flip(probs, dims=[2, 3])

            probs_sum = probs if probs_sum is None else probs_sum + probs

    if probs_sum is None:
        return torch.zeros((imgs.size(0), 1, imgs.size(2), imgs.size(3)), dtype=torch.float32)
    return probs_sum / 4.0


def tta_predict_loader(model, loader, device, use_amp: bool):
    all_probs, all_masks, all_labels, all_fnames, all_defect_types = [], [], [], [], []
    model.eval()
    for imgs, masks, labels, fnames, defect_types in loader:
        probs = tta_predict(model, imgs, device, use_amp)
        for i in range(imgs.size(0)):
            all_probs.append(probs[i, 0].numpy())
            all_masks.append(masks[i, 0].numpy())
            all_labels.append(labels[i].item())
            all_fnames.append(fnames[i])
            all_defect_types.append(defect_types[i])
    return all_probs, all_masks, all_labels, all_fnames, all_defect_types


# ──────────────────────────────────────────────────────────────────────────────
# 8. THRESHOLD & MIN-AREA GRID TUNING (VALIDATION ONLY)
# ──────────────────────────────────────────────────────────────────────────────
def tune_threshold_and_area(all_probs, all_masks, all_labels, image_size: int,
                             border_margin: int, min_area_start: int,
                             out_dir: Path) -> tuple[float, int, pd.DataFrame]:
    thresholds = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    scale = (image_size / 256) ** 2
    min_areas = [max(50, int(a * scale)) for a in [100, 150, 200, 300]]

    H = W = image_size
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
                "empty_preds": m["empty_preds"],
                "score": round(m["score"], 4),
            })
            if m["score"] > best_score:
                best_score = m["score"]
                best_thr = thr
                best_area = ma

    df = pd.DataFrame(records)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "threshold_area_tuning.csv", index=False)
    print(f"\n  Threshold & min_area grid saved -> {out_dir / 'threshold_area_tuning.csv'}")
    print(f"  Tuned Result: thr={best_thr:.2f}, min_area={best_area}px | Score: {best_score:.4f}")
    return best_thr, best_area, df


# ──────────────────────────────────────────────────────────────────────────────
# 9. SCHEDULER: LINEAR WARMUP + COSINE ANNEALING
# ──────────────────────────────────────────────────────────────────────────────
class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_epochs: int, total_epochs: int, eta_min: float = 1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = max(1, warmup_epochs)
        self.total_epochs = total_epochs
        self.eta_min = eta_min
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]

    def step(self, epoch: int):
        if epoch <= self.warmup_epochs:
            factor = epoch / self.warmup_epochs
        else:
            progress = (epoch - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs)
            factor = self.eta_min / self.base_lrs[0] + 0.5 * (1.0 - self.eta_min / self.base_lrs[0]) * (
                1.0 + math.cos(math.pi * progress)
            )
        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            pg["lr"] = base_lr * factor


# ──────────────────────────────────────────────────────────────────────────────
# 10. PER-DEFECT-TYPE METRICS TABLE
# ──────────────────────────────────────────────────────────────────────────────
def compute_per_defect_metrics(all_probs, all_masks, all_labels, all_defect_types,
                                thr: float, interior: np.ndarray, min_area: int) -> pd.DataFrame:
    type_records = {}
    for prob_np, mask_np, label, defect_type in zip(all_probs, all_masks, all_labels, all_defect_types):
        d, iou, is_empty = per_image_hard_dice_iou(prob_np, mask_np, thr, interior, min_area)
        pred_bin = (prob_np > thr).astype(np.uint8)
        pred_filt = apply_border_and_cc_filter(pred_bin, interior, min_area)
        fp = int(pred_filt.sum() > 0 and label == 0)

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
            if is_empty:
                type_records[defect_type]["empty_count"] += 1
        else:
            type_records[defect_type]["fp_count"] += fp

    rows = []
    for t, r in type_records.items():
        n = r["n_images"]
        is_def = bool(r["is_defective"])
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
# 11. BORDER ARTIFACT CHECK ON NORMAL IMAGES
# ──────────────────────────────────────────────────────────────────────────────
def check_border_artifact(all_probs, all_labels):
    normal_probs = [p for p, lbl in zip(all_probs, all_labels) if lbl == 0]
    if not normal_probs:
        return
    maxima = []
    for p in normal_probs:
        idx = np.unravel_index(np.argmax(p), p.shape)
        maxima.append((float(p[idx]), int(idx[1]), int(idx[0])))

    maxima.sort(reverse=True, key=lambda x: x[0])
    print(f"\n  Border Artifact Check (Top Normal Image Probability Peaks):")
    print(f"  {'Max Prob':>10} {'x':>6} {'y':>6}")
    print("  " + "-" * 26)
    for prob, x, y in maxima[:5]:
        flag = " <- ARTIFACT AT (0,1)!" if (x == 0 and y == 1) else ""
        print(f"  {prob:10.4f} {x:6d} {y:6d}{flag}")

    n_at_01 = sum(1 for _, x, y in maxima if x == 0 and y == 1)
    val_at_01 = [p[1, 0] for p in normal_probs]
    mean_val_01 = float(np.mean(val_at_01))
    max_val_01 = float(np.max(val_at_01))
    print(f"  Images peaking at (x=0, y=1): {n_at_01}/{len(normal_probs)}")
    print(f"  Probability at (x=0, y=1) across normal images: mean={mean_val_01:.4f}, max={max_val_01:.4f}")
    if n_at_01 == 0:
        print("  [SUCCESS] (x=0, y=1) border artifact eliminated by reflection padding.")
    else:
        print(f"  [NOTE] Peak at (0,1) observed in {n_at_01} images (filtered by 4px margin).")


# ──────────────────────────────────────────────────────────────────────────────
# 12. RUN SINGLE SEED WITH AUTOMATIC OOM FALLBACK
# ──────────────────────────────────────────────────────────────────────────────
def run_seed_training(seed: int, cfg: dict, batch_size: int, device, cuda_avail: bool,
                      dev_name: str, total_vram_mib: float, use_amp: bool) -> dict:
    import segmentation_models_pytorch as smp

    image_size = cfg["image_size"]
    border_margin = cfg["border_margin_px"]
    H = W = image_size
    interior = make_interior_mask(H, W, border_margin)

    out_dir = PROJECT_ROOT / cfg["out_dir"] / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 65)
    print(f"RUNNING SEED {seed}")
    print(f"  Image Size  : {image_size}x{image_size}")
    print(f"  Encoder     : {cfg['encoder']}")
    print(f"  Batch Size  : {batch_size}")
    print(f"  Epochs      : {cfg['epochs']} (min={cfg['min_epochs']}, patience={cfg['patience']})")
    print(f"  Output Dir  : {out_dir}")
    print("=" * 65)

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if cuda_avail:
        torch.cuda.manual_seed_all(seed)

    manifest_train = PROJECT_ROOT / "ml/manifests/manifest_bottle_train.csv"
    manifest_val = PROJECT_ROOT / "ml/manifests/manifest_bottle_val.csv"
    manifest_test = PROJECT_ROOT / "ml/manifests/manifest_bottle_test.csv"

    train_ds = BottleDatasetV2(manifest_train, image_size=image_size, is_train=True)
    val_ds = BottleDatasetV2(manifest_val, image_size=image_size, is_train=False)

    train_labels = train_ds.df["is_anomaly"].astype(int).values
    class_counts = np.bincount(train_labels)
    class_weights = 1.0 / np.maximum(class_counts, 1)
    sample_weights_list = [float(class_weights[lbl]) for lbl in train_labels]
    sampler = WeightedRandomSampler(sample_weights_list, len(sample_weights_list), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                               num_workers=0, pin_memory=cuda_avail, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                             num_workers=0, pin_memory=cuda_avail, collate_fn=collate_fn)

    try:
        model = smp.Unet(encoder_name=cfg["encoder"], encoder_weights="imagenet",
                          in_channels=3, classes=1)
        print(f"  Model: U-Net ({cfg['encoder']}, ImageNet pretrained)")
    except Exception as e:
        print(f"  Pretrained weights download failed ({e}). Initializing randomly.")
        model = smp.Unet(encoder_name=cfg["encoder"], encoder_weights=None,
                          in_channels=3, classes=1)
    model = model.to(device)

    criterion = ConfiguredLoss(cfg)
    optimizer = optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = WarmupCosineScheduler(optimizer, warmup_epochs=cfg["warmup_epochs"],
                                       total_epochs=cfg["epochs"])
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_score = -float("inf")
    best_ckpt = out_dir / "best.ckpt"
    patience_counter = 0
    peak_vram_mib = 0.0

    history = {
        "epoch": [], "train_loss": [], "val_dice35": [], "val_fp35": [],
        "score35": [], "epoch_time_s": []
    }

    print(f"\n{'Ep':>4} | {'TrLoss':>7} | {'Dice@30':>7} | {'Dice@40':>7} | {'Dice@50':>7} | "
          f"{'FP@35':>6} | {'Score':>6} | {'Time':>5} | {'VRAM':>6}")
    print("-" * 92)

    for epoch in range(1, cfg["epochs"] + 1):
        t0 = time.time()
        if cuda_avail:
            torch.cuda.reset_peak_memory_stats(0)

        model.train()
        train_loss = 0.0
        for imgs, masks, _, _, _ in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(imgs)
                loss = criterion(logits, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item() * imgs.size(0)

        train_loss /= len(train_ds)
        scheduler.step(epoch)

        all_probs, all_masks_v, all_labels_v, all_fnames_v, all_defect_types_v = tta_predict_loader(
            model, val_loader, device, use_amp)

        d30 = compute_val_metrics_v2(all_probs, all_masks_v, all_labels_v, 0.30, interior,
                                     cfg["min_component_area_start"])["dice_def"]
        d40 = compute_val_metrics_v2(all_probs, all_masks_v, all_labels_v, 0.40, interior,
                                     cfg["min_component_area_start"])["dice_def"]
        d50 = compute_val_metrics_v2(all_probs, all_masks_v, all_labels_v, 0.50, interior,
                                     cfg["min_component_area_start"])["dice_def"]

        m35 = compute_val_metrics_v2(all_probs, all_masks_v, all_labels_v, 0.35, interior,
                                     cfg["min_component_area_start"])
        score = m35["score"]

        if cuda_avail:
            cur_peak = torch.cuda.max_memory_allocated(0) / (1024 ** 2)
            peak_vram_mib = max(peak_vram_mib, cur_peak)

        ep_time = time.time() - t0
        if epoch == 1:
            print(f"  [Epoch 1 Profiling] Time: {ep_time:.1f}s | Peak VRAM: {peak_vram_mib:.0f} MiB")

        history["epoch"].append(epoch)
        history["train_loss"].append(round(train_loss, 4))
        history["val_dice35"].append(round(m35["dice_def"], 4))
        history["val_fp35"].append(round(m35["fp_rate"], 4))
        history["score35"].append(round(score, 4))
        history["epoch_time_s"].append(round(ep_time, 1))

        print(f"{epoch:4d} | {train_loss:7.4f} | {d30:7.4f} | {d40:7.4f} | {d50:7.4f} | "
              f"{m35['fp_rate']:6.4f} | {score:6.4f} | {ep_time:4.0f}s | {peak_vram_mib:5.0f}M")

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
            print(f"       -> [Saved best checkpoint epoch={epoch} score={score:.4f}]")
        else:
            patience_counter += 1
            if epoch >= cfg["min_epochs"] and patience_counter >= cfg["patience"]:
                print(f"\nEarly stopping triggered at epoch {epoch} (patience={cfg['patience']}).")
                break

    print(f"\nLoading best checkpoint for final evaluation...")
    ckpt = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    all_probs, all_masks_v, all_labels_v, all_fnames_v, all_defect_types_v = tta_predict_loader(
        model, val_loader, device, use_amp)

    best_thr, best_min_area, _ = tune_threshold_and_area(
        all_probs, all_masks_v, all_labels_v,
        image_size=image_size, border_margin=border_margin,
        min_area_start=cfg["min_component_area_start"], out_dir=out_dir)

    final_m = compute_val_metrics_v2(all_probs, all_masks_v, all_labels_v,
                                      best_thr, interior, best_min_area)

    per_defect_df = compute_per_defect_metrics(
        all_probs, all_masks_v, all_labels_v, all_defect_types_v,
        best_thr, interior, best_min_area)
    per_defect_df.to_csv(out_dir / "per_defect_metrics.csv", index=False)

    print("\nPer-Defect-Type Validation Metrics:")
    print(per_defect_df.to_string(index=False))

    check_border_artifact(all_probs, all_labels_v)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["epoch"], history["train_loss"], label="Train Loss", marker="o", markersize=3)
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend()

    axes[1].plot(history["epoch"], history["val_dice35"], label="Val Dice (Defective)", color="green", marker="o", markersize=3)
    axes[1].plot(history["epoch"], history["val_fp35"], label="Val FP Rate (Normal)", color="red", marker="x", markersize=3)
    axes[1].set_title("Validation Metrics (thr=0.35)")
    axes[1].set_xlabel("Epoch")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend()

    plt.tight_layout()
    curve_path = out_dir / "learning_curve.png"
    plt.savefig(curve_path, dpi=120)
    plt.close(fig)

    val_res = pd.DataFrame([{
        "seed": seed,
        "best_epoch": int(ckpt["epoch"]),
        "best_threshold": best_thr,
        "best_min_area": best_min_area,
        "dice_defective": round(final_m["dice_def"], 4),
        "iou_defective": round(final_m["iou_def"], 4),
        "fp_rate_normal": round(final_m["fp_rate"], 4),
        "empty_defective_preds": final_m["empty_preds"],
        "selection_score": round(final_m["score"], 4),
        "peak_vram_mib": round(peak_vram_mib, 1),
        "avg_epoch_time_s": round(float(np.mean(history["epoch_time_s"])), 1),
    }])
    val_res.to_csv(out_dir / "val_results.csv", index=False)

    seed_config = cfg.copy()
    seed_config.update({
        "active_seed": seed,
        "best_epoch": int(ckpt["epoch"]),
        "best_threshold": best_thr,
        "best_min_area": best_min_area,
        "final_metrics": final_m,
        "peak_vram_mib": round(peak_vram_mib, 1),
        "epochs_trained": len(history["epoch"]),
    })
    (out_dir / "config.json").write_text(json.dumps(seed_config, indent=2, default=str), encoding="utf-8")

    print("\n-- 2-Image Test Smoke Check (No Metrics Recorded) --")
    test_ds = BottleDatasetV2(manifest_test, image_size=image_size, is_train=False)
    smoke_loader = DataLoader(test_ds, batch_size=2, shuffle=False, num_workers=0, collate_fn=collate_fn)
    smoke_dir = out_dir / "test_smoke"
    smoke_dir.mkdir(exist_ok=True)
    with torch.no_grad():
        imgs_test, _, _, fnames_test, _ = next(iter(smoke_loader))
        probs_test = tta_predict(model, imgs_test, device, use_amp)
        preds_test = (probs_test > best_thr).float()
        for idx in range(min(2, preds_test.size(0))):
            pred_arr = (preds_test[idx, 0].numpy() * 255).astype(np.uint8)
            out_img = smoke_dir / f"smoke_{idx}_{fnames_test[idx]}"
            Image.fromarray(pred_arr).save(out_img)
            print(f"  Saved smoke image -> {out_img}")

    return {
        "seed": seed,
        "out_dir": str(out_dir),
        "ckpt_path": str(best_ckpt),
        "best_epoch": int(ckpt["epoch"]),
        "best_thr": best_thr,
        "best_min_area": best_min_area,
        "dice_def": final_m["dice_def"],
        "iou_def": final_m["iou_def"],
        "fp_rate": final_m["fp_rate"],
        "empty_preds": final_m["empty_preds"],
        "score": final_m["score"],
        "peak_vram_mib": peak_vram_mib,
        "avg_epoch_time_s": float(np.mean(history["epoch_time_s"])),
    }


def train_seed(seed: int, cfg: dict, device, cuda_avail: bool,
               dev_name: str, total_vram_mib: float, use_amp: bool) -> dict:
    bs = cfg["batch_size"]
    fallback_bs = cfg["fallback_batch_size"]

    try:
        return run_seed_training(seed, cfg, bs, device, cuda_avail, dev_name, total_vram_mib, use_amp)
    except torch.cuda.OutOfMemoryError as oom:
        if bs > fallback_bs:
            print("\n" + "!" * 70)
            print(f"[CUDA OOM] Caught at batch_size={bs}!")
            print(f"Automatic Fallback: Freeing GPU cache and restarting with batch_size={fallback_bs}...")
            print("!" * 70 + "\n")
            if cuda_avail:
                torch.cuda.empty_cache()
            return run_seed_training(seed, cfg, fallback_bs, device, cuda_avail, dev_name, total_vram_mib, use_amp)
        else:
            raise oom


# ──────────────────────────────────────────────────────────────────────────────
# 13. MAIN CLI
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="U-Net v2 Bottle Training (Phase 4A-v2)")
    parser.add_argument("--config", type=str, default="ml/configs/bottle_unet_v2_256.json",
                        help="Path to JSON config file")
    parser.add_argument("--seed", type=int, default=None,
                        help="Run a specific random seed (e.g. 42, 43, 44). If omitted, runs all seeds in config.")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override number of training epochs (e.g. 3 for quick test).")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override mini-batch size.")
    parser.add_argument("--no-amp", action="store_true",
                        help="Disable automatic mixed precision.")
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.epochs is not None:
        cfg["epochs"] = args.epochs
        if args.epochs < cfg["min_epochs"]:
            cfg["min_epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size
    if args.no_amp:
        cfg["amp"] = False

    device, cuda_avail, dev_name, total_vram_mib = setup_device()
    use_amp = cuda_avail and cfg["amp"]

    if args.seed is not None:
        seeds_to_run = [args.seed]
    else:
        seeds_to_run = cfg["seeds"]

    print("Active Configuration:")
    print(f"  Encoder         : {cfg['encoder']}")
    print(f"  Image Size      : {cfg['image_size']}")
    print(f"  Batch Size      : {cfg['batch_size']} (fallback={cfg['fallback_batch_size']})")
    print(f"  Epochs          : {cfg['epochs']} (min={cfg['min_epochs']})")
    print(f"  Seeds           : {seeds_to_run}")
    print(f"  AMP Enabled     : {use_amp}")

    results = []
    for s in seeds_to_run:
        res = train_seed(s, cfg, device, cuda_avail, dev_name, total_vram_mib, use_amp)
        results.append(res)

    print("\n" + "=" * 65)
    print("FINISHED EXECUTION")
    for r in results:
        print(f"  Seed {r['seed']}: Best Ep={r['best_epoch']}, Thr={r['best_thr']:.2f}, "
              f"Dice={r['dice_def']:.4f}, IoU={r['iou_def']:.4f}, FP={r['fp_rate']:.4f}, "
              f"Empty={r['empty_preds']}")
    print("=" * 65)


if __name__ == "__main__":
    main()
