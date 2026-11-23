#!/usr/bin/env python
"""
ml/verify_v2.py
Comprehensive Verification Suite for Phase 4A-v2 U-Net (bottle).
READ-ONLY: Does not retrain or modify checkpoints.

Performs:
  1. Split Integrity & Manifest Hash Verification
  2. Zero Leakage Verification
  3. Per-Image Validation Table on all 12 defective validation images for all seeds
  4. Border Artifact Measurement on all 31 normal validation images
  5. Checkpoint Selection Analysis & Epoch Trajectories
"""

import hashlib
import json
import math
import sys
from pathlib import Path

# Ensure UTF-8 output on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
import segmentation_models_pytorch as smp
from scipy import ndimage as ndi

PROJECT_ROOT = Path(__file__).resolve().parents[1]

try:
    from PIL.Image import Resampling
    RESAMPLE_BILINEAR = Resampling.BILINEAR
    RESAMPLE_NEAREST = Resampling.NEAREST
except (ImportError, AttributeError):
    RESAMPLE_BILINEAR = 2
    RESAMPLE_NEAREST = 0


# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def md5_file(path: Path) -> str:
    h = hashlib.md5()
    h.update(path.read_bytes())
    return h.hexdigest()


def make_interior_mask(H: int, W: int, margin: int) -> np.ndarray:
    m = np.zeros((H, W), dtype=bool)
    if margin > 0:
        m[margin:H - margin, margin:W - margin] = True
    else:
        m[:] = True
    return m


def apply_border_and_cc(pred_bin: np.ndarray, interior: np.ndarray, min_area: int) -> np.ndarray:
    res = pred_bin.copy()
    res[~interior] = 0
    labeled, n_cc = ndi.label(res)
    if n_cc > 0:
        sizes = ndi.sum(res, labeled, range(1, n_cc + 1))
        for idx, sz in enumerate(sizes, start=1):
            if sz < min_area:
                res[labeled == idx] = 0
    return res


# ──────────────────────────────────────────────────────────────────────────────
# DATASET
# ──────────────────────────────────────────────────────────────────────────────
class BottleValDataset(Dataset):
    def __init__(self, manifest_path: Path, image_size: int = 256):
        self.df = pd.read_csv(manifest_path)
        self.sz = image_size
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int):
        row = self.df.iloc[index]
        with Image.open(row["image_path"]) as pil_img:
            img = pil_img.convert("RGB").resize((self.sz, self.sz), RESAMPLE_BILINEAR)
            img_np = np.array(img, dtype=np.float32) / 255.0
            # [H, W, 3] -> [3, H, W] normalized
            img_t = torch.from_numpy(img_np.transpose(2, 0, 1)).float()
            img_t = (img_t - torch.from_numpy(self.mean)) / torch.from_numpy(self.std)

        mask_val = row.get("mask_path", "")
        if pd.isna(mask_val) or str(mask_val).strip() == "":
            mask_np = np.zeros((self.sz, self.sz), dtype=np.float32)
        else:
            with Image.open(str(mask_val).strip()) as pil_mask:
                m = pil_mask.convert("L").resize((self.sz, self.sz), RESAMPLE_NEAREST)
                mask_np = (np.array(m, dtype=np.float32) > 0).astype(np.float32)

        mask_t = torch.from_numpy(mask_np).unsqueeze(0).float()
        label = int(row["is_anomaly"])
        fname = Path(row["image_path"]).name
        defect_type = str(row.get("defect_type", Path(row["image_path"]).parent.name))
        return img_t, mask_t, label, fname, defect_type


def collate_fn(batch):
    imgs, masks, labels, fnames, defect_types = zip(*batch)
    return torch.stack(imgs), torch.stack(masks), torch.tensor(labels, dtype=torch.long), list(fnames), list(defect_types)


def tta_predict(model, imgs: torch.Tensor, device) -> torch.Tensor:
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
            var = var.to(device)
            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
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


# ──────────────────────────────────────────────────────────────────────────────
# MAIN VERIFICATION
# ──────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 80)
    print("PHASE 4A-v2 INDEPENDENT AUDIT & VERIFICATION REPORT (BOTTLE U-NET)")
    print("=" * 80)

    # ── 1. SPLIT INTEGRITY ──
    print("\n" + "=" * 80)
    print("SECTION 1: SPLIT INTEGRITY & MANIFEST AUDIT")
    print("=" * 80)

    train_p = PROJECT_ROOT / "ml/manifests/manifest_bottle_train.csv"
    val_p = PROJECT_ROOT / "ml/manifests/manifest_bottle_val.csv"
    test_p = PROJECT_ROOT / "ml/manifests/manifest_bottle_test.csv"

    manifests = {"train": train_p, "val": val_p, "test": test_p}
    for name, p in manifests.items():
        df = pd.read_csv(p)
        sha = sha256_file(p)
        md5 = md5_file(p)
        print(f"  {name.upper():<5} : {p}")
        print(f"          Rows   : {len(df)}")
        print(f"          SHA256 : {sha}")
        print(f"          MD5    : {md5}")

    df_val = pd.read_csv(val_p)
    val_counts = df_val["defect_type"].value_counts().to_dict()
    print("\n  Direct Defect Type Counts in manifest_bottle_val.csv:")
    for dt, cnt in sorted(val_counts.items()):
        print(f"    - {dt:<15} : {cnt} images")
    print(f"    TOTAL VALIDATION : {len(df_val)} images (Normal: {(df_val['is_anomaly']==0).sum()}, Defective: {(df_val['is_anomaly']==1).sum()})")

    # ── 2. ZERO LEAKAGE AUDIT ──
    print("\n" + "=" * 80)
    print("SECTION 2: DATA LEAKAGE AUDIT")
    print("=" * 80)

    df_tr = pd.read_csv(train_p)
    df_te = pd.read_csv(test_p)

    tr_paths = set(df_tr["image_path"].apply(lambda x: str(Path(x).resolve())))
    va_paths = set(df_val["image_path"].apply(lambda x: str(Path(x).resolve())))
    te_paths = set(df_te["image_path"].apply(lambda x: str(Path(x).resolve())))

    tr_va_overlap = len(tr_paths.intersection(va_paths))
    tr_te_overlap = len(tr_paths.intersection(te_paths))
    va_te_overlap = len(va_paths.intersection(te_paths))

    print(f"  Train & Val  overlap : {tr_va_overlap} (Expected: 0)")
    print(f"  Train & Test overlap : {tr_te_overlap} (Expected: 0)")
    print(f"  Val & Test   overlap : {va_te_overlap} (Expected: 0)")
    if tr_va_overlap == 0 and tr_te_overlap == 0 and va_te_overlap == 0:
        print("  [PASS] Zero image leakage confirmed across all splits.")
    else:
        print("  [FAIL] LEAKAGE DETECTED!")

    # ── 3. PER-IMAGE VALIDATION EVALUATION ──
    print("\n" + "=" * 80)
    print("SECTION 3: PER-IMAGE DEFECTIVE VALIDATION TABLE ACROSS ALL SEEDS")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val_ds = BottleValDataset(val_p, image_size=256)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=0, collate_fn=collate_fn)

    interior = make_interior_mask(256, 256, margin=4)

    seeds = [42, 43, 44]
    seed_models = {}
    seed_configs = {}

    for s in seeds:
        ckpt_p = PROJECT_ROOT / f"ml/outputs/bottle/unet_v2/seed_{s}/best.ckpt"
        cfg_p = PROJECT_ROOT / f"ml/outputs/bottle/unet_v2/seed_{s}/config.json"
        if not ckpt_p.is_file():
            print(f"  Warning: Checkpoint for seed {s} not found at {ckpt_p}")
            continue

        model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1)
        ckpt = torch.load(ckpt_p, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        model = model.to(device)
        model.eval()
        seed_models[s] = (model, ckpt)

        with open(cfg_p, "r") as f:
            seed_configs[s] = json.load(f)

    # Evaluate each seed model on validation set
    all_seed_evals = {}
    for s, (model, ckpt) in seed_models.items():
        thr = seed_configs[s].get("best_threshold", 0.30)
        min_area = seed_configs[s].get("best_min_area", 100)

        records = []
        all_probs_list = []
        all_masks_list = []
        all_labels_list = []

        with torch.no_grad():
            for imgs, masks, labels, fnames, defect_types in val_loader:
                probs = tta_predict(model, imgs, device)
                for i in range(imgs.size(0)):
                    p_np = probs[i, 0].numpy()
                    m_np = masks[i, 0].numpy()
                    lbl = labels[i].item()
                    fn = fnames[i]
                    dt = defect_types[i]

                    all_probs_list.append(p_np)
                    all_masks_list.append(m_np)
                    all_labels_list.append(lbl)

                    if lbl == 1:
                        # Defective
                        pred_bin = (p_np > thr).astype(np.uint8)
                        pred_filt = apply_border_and_cc(pred_bin, interior, min_area)
                        true_filt = (m_np > 0.5).astype(np.uint8)
                        true_filt[~interior] = 0

                        inter = (pred_filt & true_filt).sum()
                        pred_area = pred_filt.sum()
                        true_area = true_filt.sum()
                        union = pred_area + true_area

                        dice = float(2.0 * inter / (union + 1e-6)) if union > 0 else 1.0
                        iou = float(inter / (union - inter + 1e-6)) if (union - inter) > 0 else 1.0

                        records.append({
                            "seed": s,
                            "filename": fn,
                            "defect_type": dt,
                            "dice": round(dice, 4),
                            "iou": round(iou, 4),
                            "pred_area": int(pred_area),
                            "true_area": int(true_area),
                            "empty": int(pred_area == 0),
                        })

        df_def_eval = pd.DataFrame(records)
        all_seed_evals[s] = {
            "defective_table": df_def_eval,
            "all_probs": all_probs_list,
            "all_masks": all_masks_list,
            "all_labels": all_labels_list,
            "thr": thr,
            "min_area": min_area,
            "ckpt_epoch": ckpt["epoch"],
            "ckpt_score": ckpt["score"],
        }

    # Print Per-Image Table for each seed
    for s in seeds:
        if s not in all_seed_evals:
            continue
        ev = all_seed_evals[s]
        df_rec = ev["defective_table"]
        print(f"\n--- SEED {s} (Saved Epoch {ev['ckpt_epoch']} | Thr={ev['thr']:.2f} | MinArea={ev['min_area']}px | 4px Margin) ---")
        print(f"{'Filename':<12} {'Defect Type':<16} {'Dice':>8} {'IoU':>8} {'PredArea':>10} {'TrueArea':>10} {'Empty':>6}")
        print("-" * 74)
        for _, r in df_rec.iterrows():
            print(f"{r['filename']:<12} {r['defect_type']:<16} {r['dice']:8.4f} {r['iou']:8.4f} {r['pred_area']:10d} {r['true_area']:10d} {r['empty']:6d}")

        print("\n  Per-Defect-Type Mean Dice for Seed " + str(s) + ":")
        for dt, group in df_rec.groupby("defect_type"):
            print(f"    - {dt:<15} ({len(group)} imgs): Mean Dice = {group['dice'].mean():.4f}, Mean IoU = {group['iou'].mean():.4f}, Empties = {group['empty'].sum()}")
        print(f"    -> OVERALL DEFECTIVE ({len(df_rec)} imgs): Mean Dice = {df_rec['dice'].mean():.4f}, Mean IoU = {df_rec['iou'].mean():.4f}")

    # ── 4. BORDER ARTIFACT MEASUREMENT ──
    print("\n" + "=" * 80)
    print("SECTION 4: BORDER ARTIFACT MEASUREMENTS (ON ALL 31 NORMAL IMAGES)")
    print("=" * 80)

    for s in seeds:
        if s not in all_seed_evals:
            continue
        ev = all_seed_evals[s]
        all_probs = ev["all_probs"]
        all_labels = ev["all_labels"]

        normal_probs = [p for p, lbl in zip(all_probs, all_labels) if lbl == 0]

        max_in_full = []
        max_in_interior = []
        prob_at_01 = []

        for p in normal_probs:
            idx = np.unravel_index(np.argmax(p), p.shape)
            max_in_full.append((float(p[idx]), int(idx[1]), int(idx[0])))

            p_int = p.copy()
            p_int[~interior] = 0.0
            max_in_interior.append(float(np.max(p_int)))

            prob_at_01.append(float(p[1, 0]))

        peaks_at_01 = sum(1 for _, x, y in max_in_full if x == 0 and y == 1)

        print(f"\n--- SEED {s} Normal Images ({len(normal_probs)} images) ---")
        print(f"  Images peaking at exact (x=0, y=1) : {peaks_at_01}/{len(normal_probs)} (V1 baseline was 31/31!)")
        print(f"  Probability at (x=0, y=1)           : Mean = {np.mean(prob_at_01):.4f}, Max = {np.max(prob_at_01):.4f} (V1 was ~0.4614)")
        print(f"  Max probability inside full image   : Mean = {np.mean([m[0] for m in max_in_full]):.4f}, Max = {np.max([m[0] for m in max_in_full]):.4f}")
        print(f"  Max probability inside 4px interior : Mean = {np.mean(max_in_interior):.4f}, Max = {np.max(max_in_interior):.4f}")

        # Top 5 normal peaks
        max_in_full.sort(reverse=True, key=lambda x: x[0])
        print("  Top 5 normal image peaks (prob, x, y):")
        for prob_val, x_coord, y_coord in max_in_full[:5]:
            marker = " <- ARTIFACT AT (0,1)" if (x_coord == 0 and y_coord == 1) else ""
            print(f"    Prob: {prob_val:.4f} at (x={x_coord:3d}, y={y_coord:3d}){marker}")

    # ── 5. CHECKPOINT SELECTION & NEIGHBOURING EPOCHS ──
    print("\n" + "=" * 80)
    print("SECTION 5: CHECKPOINT SELECTION & EPOCH TRAJECTORY AUDIT")
    print("=" * 80)

    for s in seeds:
        log_f = PROJECT_ROOT / f"ml/outputs/bottle/unet_v2/log_seed{s}.txt"
        if not log_f.is_file():
            continue

        print(f"\n--- SEED {s} Selection Trajectory ---")
        epochs_data = []
        for line in log_f.read_text(encoding="utf-8", errors="replace").splitlines():
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 8:
                try:
                    ep = int(parts[0])
                    tr_loss = float(parts[1])
                    d30 = float(parts[2])
                    d40 = float(parts[3])
                    d50 = float(parts[4])
                    fp35 = float(parts[5])
                    score = float(parts[6])
                    sec = parts[7]
                    epochs_data.append((ep, tr_loss, d30, d40, d50, fp35, score, sec))
                except ValueError:
                    continue

        if epochs_data:
            chosen_ep = seed_models[s][1]["epoch"]
            print(f"  Total epochs trained : {len(epochs_data)}")
            print(f"  Checkpoint saved ep  : Epoch {chosen_ep}")
            print("  Score at neighbouring epochs around chosen checkpoint:")
            print(f"    {'Epoch':>6} | {'TrLoss':>7} | {'Dice@30':>8} | {'Dice@40':>8} | {'Dice@50':>8} | {'FP@35':>7} | {'Score':>7}")
            print("    " + "-" * 65)

            # Show window around chosen epoch
            start_w = max(0, chosen_ep - 4)
            end_w = min(len(epochs_data), chosen_ep + 4)
            for item in epochs_data[start_w:end_w]:
                flag = "  <-- CHOSEN BEST" if item[0] == chosen_ep else ""
                print(f"    {item[0]:6d} | {item[1]:7.4f} | {item[2]:8.4f} | {item[3]:8.4f} | {item[4]:8.4f} | {item[5]:7.4f} | {item[6]:7.4f}{flag}")

    print("\n" + "=" * 80)
    print("VERIFICATION RUN COMPLETED SUCCESSFULLY.")
    print("=" * 80)


if __name__ == "__main__":
    main()
