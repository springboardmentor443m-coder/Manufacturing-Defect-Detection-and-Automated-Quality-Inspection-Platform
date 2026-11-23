#!/usr/bin/env python
"""
ml/train/bottle_unet.py
Phase 4A – U-Net training for the 'bottle' category (Protocol B custom split).
GPU-accelerated with automatic mixed precision (AMP), OOM recovery guidance,
reproducible seeding, validation-only threshold tuning, and registry recording.

Usage:
    python ml/train/bottle_unet.py --image-size 256 --encoder resnet18 --batch-size 8 --epochs 15 --patience 5
"""

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless – no display needed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

# ──────────────────────────────────────────────────────────────────────────────
# 1. ENVIRONMENT & DEVICE SETUP
# ──────────────────────────────────────────────────────────────────────────────
def setup_device():
    cuda = torch.cuda.is_available()
    if cuda:
        device = torch.device("cuda")
        dev_name = torch.cuda.get_device_name(0)
        total_vram_mib = torch.cuda.get_device_properties(0).total_memory / (1024**2)
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
        print("!" * 65)
        print("WARNING: CUDA is not available. Falling back to CPU mode.")
        print("Training will be substantially slower.")
        print("!" * 65)
    print()
    return device, cuda, dev_name, total_vram_mib


# ──────────────────────────────────────────────────────────────────────────────
# 2. DATASET CLASS
# ──────────────────────────────────────────────────────────────────────────────
class BottleDataset(Dataset):
    """
    Protocol B Dataset.
    Columns expected: image_path, mask_path, is_anomaly
    Handles missing/empty mask_path for normal images by creating an all-zero mask.
    Resizes images & masks dynamically to image_size x image_size.
    """
    def __init__(self, manifest_path: Path, image_size: int = 256):
        self.df = pd.read_csv(manifest_path)
        self.image_size = image_size
        for col in ("image_path", "is_anomaly"):
            if col not in self.df.columns:
                raise ValueError(f"Manifest {manifest_path} missing required column '{col}'")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = self._load_img(row["image_path"])

        mask_val = row.get("mask_path", "")
        if pd.isna(mask_val) or str(mask_val).strip() == "":
            mask = np.zeros((self.image_size, self.image_size), dtype=np.float32)
        else:
            mask = self._load_mask(str(mask_val).strip())

        img_t = torch.from_numpy(img.transpose(2, 0, 1)).float()  # [3, H, W] in [0, 1]
        mask_t = torch.from_numpy(mask).unsqueeze(0).float()       # [1, H, W] in {0, 1}
        label = int(row["is_anomaly"])
        return img_t, mask_t, label

    def _load_img(self, path: str) -> np.ndarray:
        with Image.open(path) as img:
            img = img.convert("RGB")
            if img.size != (self.image_size, self.image_size):
                img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
            return np.asarray(img, dtype=np.float32) / 255.0

    def _load_mask(self, path: str) -> np.ndarray:
        with Image.open(path) as mask:
            mask = mask.convert("L")
            if mask.size != (self.image_size, self.image_size):
                mask = mask.resize((self.image_size, self.image_size), Image.NEAREST)
            arr = np.asarray(mask, dtype=np.float32)
            return (arr > 0).astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# 3. LOSS FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────
def soft_dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Soft Dice loss handling empty masks gracefully (no NaN, normal images contribute smoothly).
    """
    probs = torch.sigmoid(logits)
    probs_f = probs.reshape(probs.size(0), -1)
    targets_f = targets.reshape(targets.size(0), -1)
    intersection = (probs_f * targets_f).sum(dim=1)
    union = probs_f.sum(dim=1) + targets_f.sum(dim=1)
    dice = 1.0 - (2.0 * intersection + eps) / (union + eps)
    return dice.mean()


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight: float = 0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.bce_weight * self.bce(logits, targets) + (1.0 - self.bce_weight) * soft_dice_loss(logits, targets)


# ──────────────────────────────────────────────────────────────────────────────
# 4. METRICS COMPUTATION
# ──────────────────────────────────────────────────────────────────────────────
def compute_metrics(logits: torch.Tensor,
                    masks: torch.Tensor,
                    labels: torch.Tensor,
                    thresh: float) -> dict:
    """
    Validation metrics:
      - dice_def: Dice on defective validation images
      - iou_def: IoU on defective validation images
      - fp_rate: Fraction of normal images with ANY pixel above threshold
      - mean_pos_norm: Mean fraction of predicted-positive pixels on normal images
      - overall_pos: Overall fraction of predicted-positive pixels (to catch all-black failure)
    """
    with torch.no_grad():
        probs = torch.sigmoid(logits)
        preds = (probs > thresh).float()

        defective = labels.bool()
        normal = ~defective

        if defective.any():
            d_probs = probs[defective]
            d_masks = masks[defective]
            inter = (d_probs * d_masks).sum()
            union = d_probs.sum() + d_masks.sum()
            dice_def = (2.0 * inter / (union + 1e-6)).item()
            iou_def = (inter / (union - inter + 1e-6)).item()
        else:
            dice_def = iou_def = 0.0

        if normal.any():
            n_preds = preds[normal]
            # Image has FP if any pixel is positive
            fp_per_img = (n_preds.flatten(1).sum(dim=1) > 0).float()
            fp_rate = fp_per_img.mean().item()
            mean_pos_norm = n_preds.float().mean().item()
        else:
            fp_rate = mean_pos_norm = 0.0

        overall_pos = preds.mean().item()

    return {
        "dice_def": dice_def,
        "iou_def": iou_def,
        "fp_rate": fp_rate,
        "mean_pos_norm": mean_pos_norm,
        "overall_pos": overall_pos,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 5. SANITY OVERFIT TEST
# ──────────────────────────────────────────────────────────────────────────────
def sanity_test(model, dataset, device, lr: float, use_amp: bool):
    """
    Overfits a tiny batch of 8 images for 100 steps.
    Confirms the loss drops and the model can predict defect pixels before full training.
    """
    print("── Step 3 Sanity Test: Overfitting 8 samples (100 steps) ──")
    indices = list(range(min(8, len(dataset))))
    subset = torch.utils.data.Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=8, shuffle=False, num_workers=0)

    criterion = BCEDiceLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    model.train()
    first_loss = None
    last_loss = None

    for step in range(100):
        for imgs, masks, _ in loader:
            imgs = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(imgs)
                loss = criterion(logits, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if step == 0:
                first_loss = loss.item()
            last_loss = loss.item()

        if step % 25 == 0 or step == 99:
            print(f"  Step {step:3d}/100 | Loss: {last_loss:.4f}")

    if last_loss >= first_loss - 0.04:
        print(f"\n[FAIL] Sanity test failed! Loss did not drop significantly: {first_loss:.4f} -> {last_loss:.4f}")
        sys.exit(1)

    print(f"  [PASS] Loss decreased from {first_loss:.4f} to {last_loss:.4f}. Model can overfit defects.\n")

    # Reset model parameters to cleanly start training
    for m in model.modules():
        if hasattr(m, "reset_parameters"):
            m.reset_parameters()


# ──────────────────────────────────────────────────────────────────────────────
# 6. THRESHOLD SWEEP (VALIDATION SET ONLY)
# ──────────────────────────────────────────────────────────────────────────────
def tune_threshold(model, val_loader, device, out_dir: Path):
    """
    Sweeps thresholds 0.1 to 0.9 on VALIDATION data only.
    Selection rule: highest Dice on defective images while keeping normal false-positive rate <= 0.05.
    If none achieves <= 0.05, selects the maximum Dice score.
    """
    print("── Post-Training Validation Threshold Sweep (0.1 to 0.9) ──")
    model.eval()

    all_logits, all_masks, all_labels = [], [], []
    with torch.no_grad():
        for imgs, masks, labels in val_loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            all_logits.append(logits.cpu())
            all_masks.append(masks)
            all_labels.append(labels)

    all_logits = torch.cat(all_logits)
    all_masks = torch.cat(all_masks)
    all_labels = torch.cat(all_labels)

    sweep_records = []
    thresholds = np.round(np.arange(0.10, 0.91, 0.05), 2)

    for thr in thresholds:
        m = compute_metrics(all_logits, all_masks, all_labels, thresh=float(thr))
        sweep_records.append({
            "threshold": thr,
            "dice_def": round(m["dice_def"], 4),
            "iou_def": round(m["iou_def"], 4),
            "fp_rate": round(m["fp_rate"], 4),
            "mean_pos_norm": round(m["mean_pos_norm"], 6),
            "overall_pos": round(m["overall_pos"], 6),
        })

    df = pd.DataFrame(sweep_records)
    df.to_csv(out_dir / "threshold_sweep.csv", index=False)

    # Filter by low false-positive rate (<= 5%)
    low_fp = df[df["fp_rate"] <= 0.05]
    if not low_fp.empty:
        best_row = low_fp.loc[low_fp["dice_def"].idxmax()]
    else:
        best_row = df.loc[df["dice_def"].idxmax()]

    best_thr = float(best_row["threshold"])
    print(f"  Selected Threshold: {best_thr:.2f}")
    print(f"  Validation Dice (Defective): {best_row['dice_def']:.4f}")
    print(f"  Validation IoU  (Defective): {best_row['iou_def']:.4f}")
    print(f"  Normal False-Positive Rate : {best_row['fp_rate']:.4f}")
    print(f"  Fraction Positive Pixels   : {best_row['overall_pos']:.6f}\n")
    return best_thr, df, best_row.to_dict()


# ──────────────────────────────────────────────────────────────────────────────
# 7. GALLERY GENERATION (IMAGE / TRUE MASK / PRED MASK / OVERLAY)
# ──────────────────────────────────────────────────────────────────────────────
def generate_gallery(model, val_dataset, device, threshold: float, out_dir: Path):
    gallery_dir = out_dir / "gallery"
    gallery_dir.mkdir(exist_ok=True)
    loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0)
    model.eval()

    samples = []
    with torch.no_grad():
        for imgs, masks, labels in loader:
            logits = model(imgs.to(device))
            probs = torch.sigmoid(logits).cpu()
            for i in range(imgs.size(0)):
                img_np = imgs[i].numpy().transpose(1, 2, 0)
                mask_np = masks[i, 0].numpy()
                prob_np = probs[i, 0].numpy()
                pred_np = (prob_np > threshold).astype(np.float32)
                lbl = labels[i].item()

                inter = (prob_np * mask_np).sum()
                union = prob_np.sum() + mask_np.sum()
                dice = (2 * inter / (union + 1e-6)) if lbl == 1 else (1.0 - pred_np.mean())
                samples.append((dice, img_np, mask_np, pred_np, lbl))

    # Sort worst predictions first so at least 3 poor predictions are visible
    samples.sort(key=lambda s: s[0])
    top_n = min(8, len(samples))
    selected = samples[:top_n]

    fig, axes = plt.subplots(top_n, 4, figsize=(14, 3.2 * top_n))
    if top_n == 1:
        axes = axes[np.newaxis, :]

    for r, (dice, img_np, mask_np, pred_np, lbl) in enumerate(selected):
        overlay = img_np.copy()
        # Red = Ground Truth, Green = Predicted
        overlay[mask_np > 0.5] = overlay[mask_np > 0.5] * 0.4 + np.array([1, 0, 0]) * 0.6
        overlay[pred_np > 0.5] = overlay[pred_np > 0.5] * 0.4 + np.array([0, 1, 0]) * 0.6

        for c, (data, title, cmap) in enumerate([
            (img_np, f"Image ({'Defective' if lbl==1 else 'Normal'})", None),
            (mask_np, "True Mask", "gray"),
            (pred_np, f"Pred Mask (thr={threshold:.2f})", "gray"),
            (overlay, f"Overlay (Dice: {dice:.3f})", None),
        ]):
            ax = axes[r, c]
            ax.imshow(np.clip(data, 0, 1), cmap=cmap)
            ax.set_title(title, fontsize=8)
            ax.axis("off")

    red_patch = mpatches.Patch(color="red", label="True Defect")
    green_patch = mpatches.Patch(color="green", label="Predicted Defect")
    fig.legend(handles=[red_patch, green_patch], loc="lower center", ncol=2, fontsize=10, frameon=False)
    fig.suptitle(f"Validation Sample Gallery (Worst Predictions First) - Threshold {threshold:.2f}", fontsize=12)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])

    gallery_path = gallery_dir / "validation_gallery.png"
    plt.savefig(gallery_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Gallery saved to: {gallery_path}")
    return gallery_path


# ──────────────────────────────────────────────────────────────────────────────
# 8. REGISTRY MANAGEMENT
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

    # Preserve other models; replace existing entry if same model & version
    updated = [
        r for r in registry
        if not (r.get("name") == entry["name"] and r.get("version") == entry["version"])
    ]
    updated.append(entry)
    reg_path.write_text(json.dumps(updated, indent=2), encoding="utf-8")
    print(f"  Updated model registry: {reg_path}")


def get_file_md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


# ──────────────────────────────────────────────────────────────────────────────
# 9. MAIN PIPELINE
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="U-Net Bottle Training (Protocol B)")
    parser.add_argument("--image-size", type=int, default=256, help="Input image dimension (H=W)")
    parser.add_argument("--encoder", type=str, default="resnet18", help="Encoder backbone architecture")
    parser.add_argument("--batch-size", type=int, default=8, help="Mini-batch size")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for AdamW")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--no-amp", action="store_true", help="Disable automatic mixed precision")
    parser.add_argument("--out-dir", type=str, default="ml/outputs/bottle/unet", help="Artifacts directory")
    args = parser.parse_args()

    # 1. Device Setup
    device, cuda_avail, dev_name, total_vram_mib = setup_device()
    use_amp = cuda_avail and (not args.no_amp)

    # 2. Seeding
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if cuda_avail:
        torch.cuda.manual_seed_all(args.seed)

    # 3. Path Validation
    project_root = Path(__file__).resolve().parents[2]
    out_dir = project_root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_train = project_root / "ml/manifests/manifest_bottle_train.csv"
    manifest_val = project_root / "ml/manifests/manifest_bottle_val.csv"
    manifest_test = project_root / "ml/manifests/manifest_bottle_test.csv"

    for p in (manifest_train, manifest_val, manifest_test):
        if not p.is_file():
            sys.exit(f"ERROR: Manifest missing at {p}")

    train_hash = get_file_md5(manifest_train)
    val_hash = get_file_md5(manifest_val)

    # 4. Data Loading
    print("Loading manifests...")
    train_ds = BottleDataset(manifest_train, image_size=args.image_size)
    val_ds = BottleDataset(manifest_val, image_size=args.image_size)
    test_ds = BottleDataset(manifest_test, image_size=args.image_size)

    # WeightedRandomSampler: training set has 39 defective vs 178 normal
    train_labels = train_ds.df["is_anomaly"].astype(int).values
    class_counts = np.bincount(train_labels)  # [n_normal, n_defective]
    class_weights = 1.0 / np.maximum(class_counts, 1)
    sample_weights = torch.tensor(class_weights[train_labels], dtype=torch.float)

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    sampling_ratio_doc = f"Class counts normal={class_counts[0]}, defective={class_counts[1]}. Balanced ~50:50 per batch."
    print(f"Sampling setup: {sampling_ratio_doc}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=0,  # Windows requirement
        pin_memory=cuda_avail
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=cuda_avail
    )

    # 5. Model Instantiation
    import segmentation_models_pytorch as smp
    try:
        model = smp.Unet(
            encoder_name=args.encoder,
            encoder_weights="imagenet",
            in_channels=3,
            classes=1
        )
        print(f"Model initialized: U-Net ({args.encoder}, ImageNet pretrained)")
    except Exception as e:
        print(f"Pretrained weights download failed ({e}). Initializing with random weights.")
        model = smp.Unet(
            encoder_name=args.encoder,
            encoder_weights=None,
            in_channels=3,
            classes=1
        )

    model = model.to(device)

    # 6. Sanity Overfit Test
    sanity_test(model, train_ds, device, lr=args.lr, use_amp=use_amp)

    # 7. Optimizer, Scaler, Scheduler
    criterion = BCEDiceLoss(bce_weight=0.5)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    best_score = -float("inf")
    best_ckpt = out_dir / "best.ckpt"
    patience_counter = 0

    history = {
        "epoch": [], "train_loss": [], "val_loss": [],
        "val_dice": [], "val_iou": [], "val_fp": [],
        "overall_pos": [], "epoch_time_s": []
    }

    peak_vram_mib = 0.0

    print("── Full Training Loop ──")
    print(f"{'Epoch':>5} | {'Train Loss':>10} | {'Val Loss':>9} | {'Val Dice':>8} | {'Val IoU':>7} | {'FP Rate':>7} | {'Pos Pix':>7} | {'Time':>5}")
    print("-" * 80)

    try:
        for epoch in range(1, args.epochs + 1):
            t_start = time.time()
            if cuda_avail:
                torch.cuda.reset_peak_memory_stats(0)

            # Training
            model.train()
            train_loss = 0.0
            for imgs, masks, _ in train_loader:
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

            train_loss /= len(train_loader.dataset)

            # Validation
            model.eval()
            val_loss = 0.0
            all_logits, all_masks, all_labels = [], [], []

            with torch.no_grad():
                for imgs, masks, labels in val_loader:
                    imgs = imgs.to(device, non_blocking=True)
                    masks = masks.to(device, non_blocking=True)

                    with torch.amp.autocast("cuda", enabled=use_amp):
                        logits = model(imgs)
                        loss = criterion(logits, masks)

                    val_loss += loss.item() * imgs.size(0)
                    all_logits.append(logits.cpu())
                    all_masks.append(masks.cpu())
                    all_labels.append(labels)

            val_loss /= len(val_loader.dataset)
            epoch_time = time.time() - t_start

            # Metrics at default threshold 0.50
            cat_logits = torch.cat(all_logits)
            cat_masks = torch.cat(all_masks)
            cat_labels = torch.cat(all_labels)
            m = compute_metrics(cat_logits, cat_masks, cat_labels, thresh=0.50)

            if cuda_avail:
                cur_peak = torch.cuda.max_memory_allocated(0) / (1024**2)
                if cur_peak > peak_vram_mib:
                    peak_vram_mib = cur_peak

            if epoch == 1:
                print(f"[Epoch 1 Profiling] Time: {epoch_time:.1f}s | Peak VRAM: {peak_vram_mib:.0f} MiB / {total_vram_mib:.0f} MiB")

            # Selection rule: Dice on defective - 0.5 * False Positive rate on normal
            score = m["dice_def"] - 0.5 * m["fp_rate"]

            history["epoch"].append(epoch)
            history["train_loss"].append(round(train_loss, 4))
            history["val_loss"].append(round(val_loss, 4))
            history["val_dice"].append(round(m["dice_def"], 4))
            history["val_iou"].append(round(m["iou_def"], 4))
            history["val_fp"].append(round(m["fp_rate"], 4))
            history["overall_pos"].append(round(m["overall_pos"], 5))
            history["epoch_time_s"].append(round(epoch_time, 1))

            print(f"{epoch:5d} | {train_loss:10.4f} | {val_loss:9.4f} | {m['dice_def']:8.4f} | {m['iou_def']:7.4f} | {m['fp_rate']:7.4f} | {m['overall_pos']:7.4f} | {epoch_time:4.0f}s")

            # Checkpoint saving
            if score > best_score:
                best_score = score
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "score": score,
                    "val_metrics": m,
                    "encoder": args.encoder,
                    "image_size": args.image_size
                }, best_ckpt)
                patience_counter = 0
                print(f"        -> [Saved best checkpoint (score: {score:.4f})]")
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"\nEarly stopping triggered after {epoch} epochs (patience={args.patience}).")
                    break

            scheduler.step(val_loss)

    except torch.cuda.OutOfMemoryError as oom:
        print("\n" + "!" * 70)
        print("CUDA OUT OF MEMORY ERROR CAUGHT!")
        print(f"Attempted Batch Size: {args.batch_size}, Image Size: {args.image_size}")
        print("RECOMMENDATION:")
        if args.batch_size > 2:
            next_bs = max(2, args.batch_size // 2)
            print(f"  -> Re-run with smaller batch size: --batch-size {next_bs}")
        else:
            print("  -> Re-run with smaller image size: --image-size 224 or --batch-size 2")
        print("!" * 70 + "\n")
        raise oom

    # 8. Load Best Checkpoint for Threshold Tuning
    print(f"\nLoading best checkpoint from {best_ckpt}...")
    ckpt = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    # 9. Validation-Only Threshold Tuning
    best_thr, sweep_df, best_val_metrics = tune_threshold(model, val_loader, device, out_dir)

    # 10. Generate Learning Curve Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
    ax1.plot(history["epoch"], history["train_loss"], label="Train Loss", marker="o", markersize=3)
    ax1.plot(history["epoch"], history["val_loss"], label="Val Loss", marker="s", markersize=3)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training & Validation Loss")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend()

    ax2.plot(history["epoch"], history["val_dice"], label="Val Dice (Defective)", color="green", marker="o", markersize=3)
    ax2.plot(history["epoch"], history["val_fp"], label="Val FP Rate (Normal)", color="red", marker="x", markersize=3)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Metric Value")
    ax2.set_title("Validation Dice & False-Positive Rate")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend()

    curve_path = out_dir / "learning_curve.png"
    plt.tight_layout()
    plt.savefig(curve_path, dpi=120)
    plt.close(fig)
    print(f"  Learning curves saved to: {curve_path}")

    # 11. Validation Results Table
    val_table = pd.DataFrame([{
        "selected_threshold": best_thr,
        "dice_defective": best_val_metrics["dice_def"],
        "iou_defective": best_val_metrics["iou_def"],
        "fp_rate_normal": best_val_metrics["fp_rate"],
        "mean_pos_normal": best_val_metrics["mean_pos_norm"],
        "overall_positive_fraction": best_val_metrics["overall_pos"]
    }])
    val_results_path = out_dir / "val_results.csv"
    val_table.to_csv(val_results_path, index=False)

    # 12. Gallery Generation
    gallery_path = generate_gallery(model, val_ds, device, best_thr, out_dir)

    # 13. Save Configuration JSON
    import timm
    config_dict = {
        "seed": args.seed,
        "image_size": args.image_size,
        "encoder": args.encoder,
        "batch_size": args.batch_size,
        "epochs_requested": args.epochs,
        "epochs_trained": len(history["epoch"]),
        "lr": args.lr,
        "patience": args.patience,
        "amp_enabled": use_amp,
        "device": dev_name,
        "total_vram_mib": round(total_vram_mib, 1),
        "peak_vram_mib": round(peak_vram_mib, 1),
        "selection_rule": "score = dice_def - 0.5 * fp_rate_normal",
        "selected_threshold": best_thr,
        "sampling_ratio": sampling_ratio_doc,
        "torch_version": torch.__version__,
        "smp_version": smp.__version__,
        "timm_version": timm.__version__,
        "manifest_hashes": {
            "train_md5": train_hash,
            "val_md5": val_hash
        }
    }
    config_path = out_dir / "config.json"
    config_path.write_text(json.dumps(config_dict, indent=2), encoding="utf-8")
    print(f"  Configuration JSON saved to: {config_path}")

    # 14. Model Registry Entry
    reg_entry = {
        "name": "bottle_unet",
        "version": "1.0",
        "category": "bottle",
        "protocol": "Protocol B custom split",
        "image_size": args.image_size,
        "encoder": args.encoder,
        "batch_size": args.batch_size,
        "amp_enabled": use_amp,
        "device_gpu": dev_name,
        "manifest_hash": train_hash,
        "selected_threshold": best_thr,
        "validation_metrics": {
            "dice_defective": best_val_metrics["dice_def"],
            "iou_defective": best_val_metrics["iou_def"],
            "fp_rate_normal": best_val_metrics["fp_rate"],
            "overall_positive_fraction": best_val_metrics["overall_pos"]
        },
        "artifacts": {
            "checkpoint": str(best_ckpt),
            "config": str(config_path),
            "learning_curve": str(curve_path),
            "gallery": str(gallery_path),
            "val_results": str(val_results_path)
        }
    }
    update_registry(project_root / "ml/models/registry.json", reg_entry)

    # 15. Smoke Test on 2 Test Images (No metrics recorded)
    print("── Smoke Test: Running inference on 2 test images (no metrics) ──")
    smoke_loader = DataLoader(test_ds, batch_size=2, shuffle=False, num_workers=0)
    smoke_dir = out_dir / "test_smoke"
    smoke_dir.mkdir(exist_ok=True)

    with torch.no_grad():
        imgs_test, _, _ = next(iter(smoke_loader))
        logits_test = model(imgs_test.to(device))
        probs_test = torch.sigmoid(logits_test).cpu()
        preds_test = (probs_test > best_thr).float()

    for idx in range(min(2, preds_test.size(0))):
        pred_map = (preds_test[idx, 0].numpy() * 255).astype(np.uint8)
        out_png = smoke_dir / f"smoke_test_{idx}_pred.png"
        Image.fromarray(pred_map).save(out_png)
        print(f"  Saved smoke prediction -> {out_png}")

    print("\n" + "=" * 65)
    print("ALL STEPS COMPLETED SUCCESSFULLY!")
    print(f"Best checkpoint : {best_ckpt}")
    print(f"Gallery path    : {gallery_path}")
    print(f"Config JSON     : {config_path}")
    print("=" * 65)


if __name__ == "__main__":
    main()
