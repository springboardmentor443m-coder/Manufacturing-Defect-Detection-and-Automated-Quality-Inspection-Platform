"""
ml/diagnose_unet_v1.py
Comprehensive Diagnostic Task for U-Net v1 (bottle).
READ-ONLY: Does not modify checkpoints, manifests, config, or registry.
Operates strictly on the VALIDATION manifest (43 images).
Saves artifacts to ml/outputs/bottle/unet/diagnostics/.
"""

import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import segmentation_models_pytorch as smp

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
VAL_MANIFEST = PROJECT_ROOT / "ml/manifests/manifest_bottle_val.csv"
CKPT_PATH = PROJECT_ROOT / "ml/outputs/bottle/unet/best.ckpt"
OUT_DIR = PROJECT_ROOT / "ml/outputs/bottle/unet/diagnostics"
OUT_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = 256
SAVED_THRESHOLD = 0.50

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Validation Manifest
val_df = pd.read_csv(VAL_MANIFEST)

# Load Model
print(f"Loading checkpoint from: {CKPT_PATH}")
ckpt = torch.load(CKPT_PATH, map_location=device)
model = smp.Unet(
    encoder_name="resnet18",
    encoder_weights=None,
    in_channels=3,
    classes=1
)
model.load_state_dict(ckpt["model_state_dict"])
model = model.to(device)
model.eval()

# Helper Functions
def load_image(path, size=IMG_SIZE):
    with Image.open(path) as img:
        img_rgb = img.convert("RGB")
        img_resized = img_rgb.resize((size, size), Image.BILINEAR)
        arr = np.asarray(img_resized, dtype=np.float32) / 255.0
        return arr, np.asarray(img_rgb)

def load_mask(path, size=IMG_SIZE):
    if not path or pd.isna(path) or str(path).strip() == "":
        return np.zeros((size, size), dtype=np.float32), None
    with Image.open(path) as m:
        m_l = m.convert("L")
        orig_arr = (np.asarray(m_l) > 0).astype(np.float32)
        m_resized = m_l.resize((size, size), Image.NEAREST)
        resized_arr = (np.asarray(m_resized) > 0).astype(np.float32)
        return resized_arr, orig_arr

# Run Inference on all 43 Validation Images
results = []
print("Running inference on 43 validation images...")
with torch.no_grad():
    for idx, row in val_df.iterrows():
        img_256, orig_img = load_image(row["image_path"], IMG_SIZE)
        mask_256, orig_mask = load_mask(row["mask_path"], IMG_SIZE)
        
        img_t = torch.from_numpy(img_256.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
        logit = model(img_t)
        prob = torch.sigmoid(logit).squeeze().cpu().numpy()
        
        fname = Path(row["image_path"]).name
        results.append({
            "idx": idx,
            "filename": fname,
            "image_path": row["image_path"],
            "mask_path": row["mask_path"],
            "defect_type": row["defect_type"],
            "is_anomaly": int(row["is_anomaly"]),
            "img_256": img_256,
            "mask_256": mask_256,
            "orig_mask": orig_mask,
            "prob_256": prob
        })

print(f"Inference complete on {len(results)} images.\n")

# ==============================================================================
# 1. PER-DEFECT-TYPE METRICS (at threshold 0.50)
# ==============================================================================
print("=" * 80)
print("1. PER-DEFECT-TYPE METRICS AT SAVED THRESHOLD (0.50)")
print("=" * 80)

# Compute metrics per image at 0.50
for r in results:
    pred_bin = (r["prob_256"] > SAVED_THRESHOLD).astype(np.float32)
    true_bin = r["mask_256"]
    r["pred_bin_50"] = pred_bin
    
    if r["is_anomaly"] == 1:
        inter = (pred_bin * true_bin).sum()
        union = pred_bin.sum() + true_bin.sum()
        r["dice_50"] = (2.0 * inter / (union + 1e-7)) if union > 0 else 1.0
        r["iou_50"] = (inter / (union - inter + 1e-7)) if (union - inter) > 0 else 1.0
    else:
        # good
        fp = 1.0 if pred_bin.sum() > 0 else 0.0
        r["dice_50"] = np.nan
        r["iou_50"] = np.nan
        r["fp_50"] = fp

# Summary table by defect_type
defect_summary = []
for dtype in ["broken_large", "broken_small", "contamination", "good"]:
    subset = [r for r in results if r["defect_type"] == dtype]
    n_imgs = len(subset)
    if dtype != "good":
        dices = [r["dice_50"] for r in subset]
        ious = [r["iou_50"] for r in subset]
        zero_dices = sum(1 for d in dices if d == 0.0)
        defect_summary.append({
            "Defect Type": dtype,
            "Images": n_imgs,
            "Mean Dice": round(float(np.mean(dices)), 4),
            "Mean IoU": round(float(np.mean(ious)), 4),
            "Images Dice=0": zero_dices,
            "FP Rate": "N/A"
        })
    else:
        fps = [r["fp_50"] for r in subset]
        defect_summary.append({
            "Defect Type": dtype,
            "Images": n_imgs,
            "Mean Dice": "N/A",
            "Mean IoU": "N/A",
            "Images Dice=0": "N/A",
            "FP Rate": f"{np.mean(fps):.4f}"
        })

summary_df = pd.DataFrame(defect_summary)
print(summary_df.to_string(index=False))
print("\nIndividual Defective Validation Images (12 images):")
print("-" * 65)
print(f"{'Filename':<12} | {'Defect Type':<15} | {'Dice @ 0.50':<12} | {'IoU @ 0.50':<12}")
print("-" * 65)
defective_results = [r for r in results if r["is_anomaly"] == 1]
for r in defective_results:
    print(f"{r['filename']:<12} | {r['defect_type']:<15} | {r['dice_50']:<12.4f} | {r['iou_50']:<12.4f}")
print("-" * 65)
print()

# ==============================================================================
# 2. LOCATION CHECK (right place, wrong place, or too small?)
# ==============================================================================
print("=" * 80)
print("2. LOCATION CHECK FOR EACH DEFECTIVE VALIDATION IMAGE")
print("=" * 80)

location_rows = []
for r in defective_results:
    true_bin = r["mask_256"]
    pred_bin = r["pred_bin_50"]
    prob = r["prob_256"]
    
    true_area = int(true_bin.sum())
    pred_area = int(pred_bin.sum())
    ratio = pred_area / true_area if true_area > 0 else 0.0
    
    inter = (pred_bin * true_bin).sum()
    precision = inter / pred_area if pred_area > 0 else 0.0
    recall = inter / true_area if true_area > 0 else 0.0
    
    # Centroid distance
    if pred_area > 0 and true_area > 0:
        py, px = np.where(pred_bin > 0)
        ty, tx = np.where(true_bin > 0)
        c_dist = float(np.sqrt((py.mean() - ty.mean())**2 + (px.mean() - tx.mean())**2))
        c_dist_str = f"{c_dist:.1f}"
    else:
        c_dist = None
        c_dist_str = "-"
        
    max_prob_in = float(prob[true_bin > 0].max()) if (true_bin > 0).any() else 0.0
    max_prob_out = float(prob[true_bin == 0].max()) if (true_bin == 0).any() else 0.0
    
    # Verdict assignment
    if pred_area == 0:
        verdict = "empty prediction"
    elif inter == 0 or (precision == 0.0):
        verdict = "wrong place"
    elif precision >= 0.50 and ratio < 0.60:
        verdict = "right place, too small"
    elif precision >= 0.50 and ratio > 1.80:
        verdict = "right place, too large"
    elif inter > 0:
        verdict = "partly right"
    else:
        verdict = "wrong place"
        
    r["location_check"] = {
        "true_area": true_area,
        "pred_area": pred_area,
        "ratio": ratio,
        "precision": precision,
        "recall": recall,
        "centroid_dist": c_dist,
        "max_prob_in": max_prob_in,
        "max_prob_out": max_prob_out,
        "verdict": verdict
    }
    
    location_rows.append({
        "Filename": r["filename"],
        "Type": r["defect_type"],
        "True Area": true_area,
        "Pred Area": pred_area,
        "Pred/True": round(ratio, 2),
        "Precision": round(precision, 3),
        "Recall": round(recall, 3),
        "Centroid Dist": c_dist_str,
        "Max P(in)": round(max_prob_in, 3),
        "Max P(out)": round(max_prob_out, 3),
        "Verdict": verdict
    })

loc_df = pd.DataFrame(location_rows)
print(loc_df.to_string(index=False))

print("\nVerdict Summary Counts:")
verdict_counts = loc_df["Verdict"].value_counts().to_dict()
for v, count in verdict_counts.items():
    print(f"  - {v:<25}: {count}")
print()

# ==============================================================================
# 3. THRESHOLD SWEEP CHECK (0.05 to 0.95 in steps of 0.05)
# ==============================================================================
print("=" * 80)
print("3. THRESHOLD SWEEP CHECK (0.05 to 0.95, step 0.05)")
print("=" * 80)

sweep_thresholds = np.round(np.arange(0.05, 0.96, 0.05), 2)
new_sweep_rows = []

for thr in sweep_thresholds:
    dices, ious = [], []
    fp_count = 0
    pos_pixel_counts = []
    
    for r in results:
        p_bin = (r["prob_256"] > thr).astype(np.float32)
        pos_pixel_counts.append(p_bin.mean())
        if r["is_anomaly"] == 1:
            t_bin = r["mask_256"]
            inter = (p_bin * t_bin).sum()
            union = p_bin.sum() + t_bin.sum()
            d = (2.0 * inter / (union + 1e-7)) if union > 0 else 1.0
            iou = (inter / (union - inter + 1e-7)) if (union - inter) > 0 else 1.0
            dices.append(d)
            ious.append(iou)
        else:
            if p_bin.sum() > 0:
                fp_count += 1
                
    mean_dice = float(np.mean(dices))
    mean_iou = float(np.mean(ious))
    fp_rate = fp_count / 31.0
    mean_pos = float(np.mean(pos_pixel_counts))
    
    new_sweep_rows.append({
        "threshold": thr,
        "mean_dice_def": round(mean_dice, 4),
        "mean_iou_def": round(mean_iou, 4),
        "fp_rate_norm": round(fp_rate, 4),
        "mean_pos_pixel_frac": round(mean_pos, 6)
    })

new_sweep_df = pd.DataFrame(new_sweep_rows)
print(new_sweep_df.to_string(index=False))

# Read and print existing threshold_sweep.csv for comparison
orig_sweep_path = PROJECT_ROOT / "ml/outputs/bottle/unet/threshold_sweep.csv"
if orig_sweep_path.is_file():
    print("\nExisting ml/outputs/bottle/unet/threshold_sweep.csv (for comparison):")
    print("-" * 65)
    orig_df = pd.read_csv(orig_sweep_path)
    print(orig_df.to_string(index=False))

# Sweep analysis questions:
highest_dice_idx = new_sweep_df["mean_dice_def"].idxmax()
highest_dice_row = new_sweep_df.iloc[highest_dice_idx]
print(f"\n(a) Threshold with highest mean Dice: {highest_dice_row['threshold']} (Dice = {highest_dice_row['mean_dice_def']:.4f}, FP Rate = {highest_dice_row['fp_rate_norm']:.4f})")

print(f"(b) Threshold chosen by training script: 0.50.")
print("    Why: In bottle_unet.py, the training sweep used soft_dice over batch tensors, which evaluated to 0.2865 for all thresholds, and picked 0.50 as the lowest threshold with FP <= 0.05.")

# Check thresholds between 0.2 and 0.4
t_20_40 = new_sweep_df[(new_sweep_df["threshold"] >= 0.20) & (new_sweep_df["threshold"] <= 0.40)]
t_50_fp = new_sweep_df[new_sweep_df["threshold"] == 0.50]["fp_rate_norm"].values[0]
print(f"(c) Thresholds between 0.20 and 0.40:")
for _, row in t_20_40.iterrows():
    diff_fp = (row['fp_rate_norm'] - t_50_fp) * 100
    print(f"    - Thr {row['threshold']:.2f}: Dice = {row['mean_dice_def']:.4f}, FP Rate = {row['fp_rate_norm']:.4f} (+{diff_fp:.1f}% vs thr 0.50)")

print()

# ==============================================================================
# 4. SMALL-DEFECT RESOLUTION CHECK
# ==============================================================================
print("=" * 80)
print("4. SMALL-DEFECT RESOLUTION CHECK (900x900 vs 256x256)")
print("=" * 80)

resolution_rows = []
for r in defective_results:
    area_900 = int(r["orig_mask"].sum())
    area_256 = int(r["mask_256"].sum())
    disappeared = (area_256 == 0)
    under_50 = (area_256 < 50)
    flag = "DISAPPEARED" if disappeared else ("UNDER 50 PX" if under_50 else "OK")
    
    resolution_rows.append({
        "Filename": r["filename"],
        "Type": r["defect_type"],
        "Area 900x900": area_900,
        "Area 256x256": area_256,
        "Ratio (256/900)": round(area_256 / area_900, 4) if area_900 > 0 else 0,
        "Status": flag
    })

res_df = pd.DataFrame(resolution_rows)
print(res_df.to_string(index=False))

print("\nResolution Statistics by Defect Type (at 256x256):")
for dtype in ["broken_large", "broken_small", "contamination"]:
    sub = res_df[res_df["Type"] == dtype]
    mean_a = sub["Area 256x256"].mean()
    min_a = sub["Area 256x256"].min()
    print(f"  - {dtype:<15}: Mean Area = {mean_a:6.1f} px | Min Area = {min_a:4d} px")
print()

# ==============================================================================
# 5. VISUAL OUTPUTS
# ==============================================================================
print("=" * 80)
print("5. GENERATING VISUAL OUTPUTS")
print("=" * 80)

# One figure per defect type
saved_paths = []
for dtype in ["broken_large", "broken_small", "contamination"]:
    sub = [r for r in defective_results if r["defect_type"] == dtype]
    n_samples = len(sub)
    
    fig, axes = plt.subplots(n_samples, 4, figsize=(15, 3.5 * n_samples))
    if n_samples == 1:
        axes = axes[np.newaxis, :]
        
    for i, r in enumerate(sub):
        img_np = r["img_256"]
        true_m = r["mask_256"]
        prob_m = r["prob_256"]
        pred_m = r["pred_bin_50"]
        dice_val = r["dice_50"]
        
        # Overlay: image + green prediction
        overlay = img_np.copy()
        # Tint prediction green
        overlay[pred_m > 0] = overlay[pred_m > 0] * 0.4 + np.array([0, 1, 0]) * 0.6
        
        axes[i, 0].imshow(img_np)
        axes[i, 0].set_title(f"{r['filename']} ({dtype})", fontsize=10)
        axes[i, 0].axis("off")
        
        axes[i, 1].imshow(true_m, cmap="gray")
        axes[i, 1].set_title(f"True Mask ({int(true_m.sum())} px)", fontsize=10)
        axes[i, 1].axis("off")
        
        im_p = axes[i, 2].imshow(prob_m, cmap="magma", vmin=0, vmax=1)
        axes[i, 2].set_title(f"Predicted Prob (Max: {prob_m.max():.2f})", fontsize=10)
        axes[i, 2].axis("off")
        fig.colorbar(im_p, ax=axes[i, 2], fraction=0.046, pad=0.04)
        
        axes[i, 3].imshow(np.clip(overlay, 0, 1))
        # Draw true mask outline in red using matplotlib contour
        if true_m.sum() > 0:
            axes[i, 3].contour(true_m, levels=[0.5], colors=['red'], linewidths=1.5)
        axes[i, 3].set_title(f"Overlay @ 0.50 (Dice={dice_val:.3f})\nRed outline=True, Green=Pred", fontsize=10)
        axes[i, 3].axis("off")
        
    fig.suptitle(f"Diagnostic Breakdown: {dtype.upper()} (Validation Set)", fontsize=13, weight="bold")
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    out_file = OUT_DIR / f"{dtype}.png"
    plt.savefig(out_file, dpi=120, bbox_inches="tight")
    plt.close(fig)
    saved_paths.append(out_file)
    print(f"Saved: {out_file}")

# Normal images top 3 highest predicted probability
normal_sub = [r for r in results if r["is_anomaly"] == 0]
normal_sub.sort(key=lambda r: float(r["prob_256"].max()), reverse=True)
top3_normal = normal_sub[:3]

fig, axes = plt.subplots(3, 4, figsize=(15, 10.5))
for i, r in enumerate(top3_normal):
    img_np = r["img_256"]
    true_m = r["mask_256"]
    prob_m = r["prob_256"]
    pred_m = r["pred_bin_50"]
    max_p = float(prob_m.max())
    
    overlay = img_np.copy()
    overlay[pred_m > 0] = overlay[pred_m > 0] * 0.4 + np.array([0, 1, 0]) * 0.6
    
    axes[i, 0].imshow(img_np)
    axes[i, 0].set_title(f"Normal: {r['filename']}", fontsize=10)
    axes[i, 0].axis("off")
    
    axes[i, 1].imshow(true_m, cmap="gray")
    axes[i, 1].set_title("True Mask (empty)", fontsize=10)
    axes[i, 1].axis("off")
    
    im_p = axes[i, 2].imshow(prob_m, cmap="magma", vmin=0, vmax=1)
    axes[i, 2].set_title(f"Pred Prob (Max: {max_p:.4f})", fontsize=10)
    axes[i, 2].axis("off")
    fig.colorbar(im_p, ax=axes[i, 2], fraction=0.046, pad=0.04)
    
    axes[i, 3].imshow(overlay)
    axes[i, 3].set_title(f"Thresholded @ 0.50 (Pred pixels: {int(pred_m.sum())})", fontsize=10)
    axes[i, 3].axis("off")

fig.suptitle("Top 3 Normal Validation Images with Highest Predicted Probability", fontsize=13, weight="bold")
plt.tight_layout(rect=[0, 0.03, 1, 0.97])
norm_file = OUT_DIR / "normal_top3_fp.png"
plt.savefig(norm_file, dpi=120, bbox_inches="tight")
plt.close(fig)
saved_paths.append(norm_file)
print(f"Saved: {norm_file}")

# Save CSV summaries to diagnostics
loc_df.to_csv(OUT_DIR / "location_check.csv", index=False)
res_df.to_csv(OUT_DIR / "small_defect_resolution.csv", index=False)
new_sweep_df.to_csv(OUT_DIR / "threshold_sweep_rerun.csv", index=False)
summary_df.to_csv(OUT_DIR / "per_defect_metrics.csv", index=False)
print("Saved summary CSVs to diagnostics directory.")
print()
print("=" * 80)
print("DIAGNOSTIC TASK COMPLETED")
print("=" * 80)
