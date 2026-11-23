"""
ml/investigate_part_a.py
PART A: Investigate the false-positive "cliff" in v1 (read-only).
Analyzes all 31 normal validation images from manifest_bottle_val.csv.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import segmentation_models_pytorch as smp

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VAL_MANIFEST = PROJECT_ROOT / "ml/manifests/manifest_bottle_val.csv"
CKPT_PATH = PROJECT_ROOT / "ml/outputs/bottle/unet/best.ckpt"
OUT_DIR = PROJECT_ROOT / "ml/outputs/bottle/unet/diagnostics"
OUT_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = 256
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
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

# Load normal validation images
val_df = pd.read_csv(VAL_MANIFEST)
normal_df = val_df[val_df["is_anomaly"] == 0].reset_index(drop=True)

records = []
prob_maps = []
imgs_rgb = []

with torch.no_grad():
    for idx, row in normal_df.iterrows():
        path = row["image_path"]
        with Image.open(path) as img:
            img_rgb = img.convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
            img_arr = np.asarray(img_rgb, dtype=np.float32) / 255.0
            
        img_t = torch.from_numpy(img_arr.transpose(2, 0, 1)).unsqueeze(0).to(device)
        logit = model(img_t)
        prob = torch.sigmoid(logit).squeeze().cpu().numpy()
        
        prob_maps.append(prob)
        imgs_rgb.append(img_arr)
        
        # Max prob location
        max_val = float(prob.max())
        y_max, x_max = np.unravel_index(np.argmax(prob), prob.shape)
        y_max, x_max = int(y_max), int(x_max)
        
        # Distance to nearest border
        dist_border = min(x_max, y_max, (IMG_SIZE - 1) - x_max, (IMG_SIZE - 1) - y_max)
        
        records.append({
            "idx": idx + 1,
            "filename": Path(path).name,
            "max_prob": max_val,
            "x": x_max,
            "y": y_max,
            "border_dist": dist_border
        })

df_results = pd.DataFrame(records)

print("=" * 80)
print("PART A.1: MAXIMUM PREDICTED PROBABILITY PER NORMAL VALIDATION IMAGE (31 IMAGES)")
print("=" * 80)
print(df_results.to_string(index=False))

# Histogram of 31 maxima
maxima = df_results["max_prob"].values
print("\n" + "=" * 80)
print("HISTOGRAM OF THE 31 NORMAL MAXIMA:")
print("=" * 80)
bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
counts, edges = np.histogram(maxima, bins=bins)
for i in range(len(counts)):
    bar = "#" * int(counts[i] * 2)
    print(f"  [{edges[i]:.1f} - {edges[i+1]:.1f}) : {counts[i]:2d} images | {bar}")

print(f"\nSummary stats of maxima:")
print(f"  Min   : {maxima.min():.4f}")
print(f"  Median: {np.median(maxima):.4f}")
print(f"  Mean  : {maxima.mean():.4f}")
print(f"  Max   : {maxima.max():.4f}")

# Border distance statistics
dists = df_results["border_dist"].values
print(f"\nBorder distance statistics (pixels from nearest border at 256x256):")
print(f"  Min dist to border   : {dists.min()} px")
print(f"  Median dist to border: {np.median(dists):.1f} px")
print(f"  Mean dist to border  : {dists.mean():.1f} px")
print(f"  Max dist to border   : {dists.max()} px")
print(f"  Count with dist < 10 px: {sum(dists < 10)} / 31")
print(f"  Count with dist > 30 px: {sum(dists > 30)} / 31")

# Mean probability map across all 31 normal images
mean_prob_map = np.mean(prob_maps, axis=0)

# PART A.2: Figure generation
print("\n" + "=" * 80)
print("PART A.2: SAVING VISUAL ANALYSIS FIGURE")
print("=" * 80)

# Sort images by max probability descending
df_sorted = df_results.sort_values(by="max_prob", ascending=False).reset_index(drop=True)
top3_indices = [int(df_sorted.loc[i, "idx"]) - 1 for i in range(3)]

fig, axes = plt.subplots(2, 4, figsize=(18, 9))

# Row 1, Col 1: Average Normal Image (RGB)
mean_img_rgb = np.mean(imgs_rgb, axis=0)
axes[0, 0].imshow(mean_img_rgb)
axes[0, 0].set_title("Average Normal Image (RGB)", fontsize=11, weight="bold")
axes[0, 0].axis("off")

# Row 1, Col 2: Average Normal Probability Heatmap
im0 = axes[0, 1].imshow(mean_prob_map, cmap="magma", vmin=0, vmax=max(0.1, mean_prob_map.max()))
axes[0, 1].set_title(f"Normal Mean Prob Map (Max: {mean_prob_map.max():.4f})", fontsize=11, weight="bold")
axes[0, 1].axis("off")
fig.colorbar(im0, ax=axes[0, 1], fraction=0.046, pad=0.04)

# Row 1, Col 3: Coordinates distribution scatter
axes[0, 2].imshow(mean_img_rgb, alpha=0.6)
axes[0, 2].scatter(df_results["x"], df_results["y"], c=df_results["max_prob"], cmap="plasma", s=60, edgecolors="white", linewidths=1.5)
axes[0, 2].set_title("Peak Prob Locations on Bottle", fontsize=11, weight="bold")
axes[0, 2].set_xlim(0, 255)
axes[0, 2].set_ylim(255, 0)
axes[0, 2].axis("off")

# Row 1, Col 4: Histogram plot
axes[0, 3].hist(maxima, bins=np.linspace(0.1, 0.6, 11), color="royalblue", edgecolor="black", alpha=0.7)
axes[0, 3].axvline(0.50, color="red", linestyle="--", linewidth=2, label="v1 Thr (0.50)")
axes[0, 3].set_xlabel("Max Predicted Probability")
axes[0, 3].set_ylabel("Number of Normal Images")
axes[0, 3].set_title("Distribution of Max Prob (31 Normals)", fontsize=11, weight="bold")
axes[0, 3].legend()
axes[0, 3].grid(True, linestyle=":", alpha=0.6)

# Row 2: Top 3 Highest Probability Normal Images
for col_idx, img_i in enumerate(top3_indices):
    ax = axes[1, col_idx]
    ax.imshow(imgs_rgb[img_i])
    # Overlay heatmap with transparency
    p = prob_maps[img_i]
    ax.imshow(p, cmap="hot", alpha=0.5, vmin=0, vmax=0.5)
    px, py = df_results.loc[img_i, "x"], df_results.loc[img_i, "y"]
    pmax = df_results.loc[img_i, "max_prob"]
    ax.scatter([px], [py], color="cyan", s=80, marker="x", linewidths=2.5, label=f"Max ({px},{py})")
    ax.set_title(f"Rank {col_idx+1}: {df_results.loc[img_i, 'filename']}\nMax P = {pmax:.4f} @ ({px},{py})", fontsize=10)
    ax.axis("off")
    ax.legend(loc="upper right", fontsize=8)

# Row 2, Col 4: Top 1 Zoom-in on the bottle rim
top1_i = top3_indices[0]
ax_zoom = axes[1, 3]
p1 = prob_maps[top1_i]
px1, py1 = df_results.loc[top1_i, "x"], df_results.loc[top1_i, "y"]
# Crop 60x60 around peak
y1_min, y1_max = max(0, py1 - 30), min(IMG_SIZE, py1 + 30)
x1_min, x1_max = max(0, px1 - 30), min(IMG_SIZE, px1 + 30)
ax_zoom.imshow(imgs_rgb[top1_i][y1_min:y1_max, x1_min:x1_max])
ax_zoom.imshow(p1[y1_min:y1_max, x1_min:x1_max], cmap="hot", alpha=0.5, vmin=0, vmax=0.5)
ax_zoom.scatter([px1 - x1_min], [py1 - y1_min], color="cyan", s=100, marker="o", edgecolors="black", linewidths=2)
ax_zoom.set_title(f"Zoom-in @ Peak: {df_results.loc[top1_i, 'filename']}\n(Glass rim specular reflection)", fontsize=10)
ax_zoom.axis("off")

plt.tight_layout()
fig_path = OUT_DIR / "part_a_normal_analysis.png"
plt.savefig(fig_path, dpi=130, bbox_inches="tight")
plt.close(fig)
print(f"Saved analysis figure to: {fig_path}")

# Check specific coordinate cluster
print("\n" + "=" * 80)
print("LOCATION ANALYSIS OF HIGH-PROBABILITY PIXELS:")
print("=" * 80)
# Check y distribution (top rim is typically around y=10-70, center is ~128, bottom is ~200-250)
print(f"Y-coordinate distribution (vertical position on 256x256 image):")
print(f"  Min y: {df_results['y'].min()}, Max y: {df_results['y'].max()}, Median y: {df_results['y'].median():.1f}")
print(f"X-coordinate distribution (horizontal position on 256x256 image):")
print(f"  Min x: {df_results['x'].min()}, Max x: {df_results['x'].max()}, Median x: {df_results['x'].median():.1f}")

# Categorize where they sit
rim_count = sum((df_results["y"] >= 20) & (df_results["y"] <= 90) & (df_results["x"] >= 80) & (df_results["x"] <= 176))
border_count = sum(df_results["border_dist"] <= 5)
corner_count = sum(((df_results["x"] <= 15) | (df_results["x"] >= 240)) & ((df_results["y"] <= 15) | (df_results["y"] >= 240)))
body_count = len(df_results) - rim_count - border_count - corner_count

print(f"\nSpatial Classification of Maxima Locations:")
print(f"  - Bottle mouth / top rim (y in 20-90, x in 80-176) : {rim_count} / 31 ({rim_count/31*100:.1f}%)")
print(f"  - Image borders (within 5 px of edge)             : {border_count} / 31 ({border_count/31*100:.1f}%)")
print(f"  - Image corners (within 15 px of corners)         : {corner_count} / 31 ({corner_count/31*100:.1f}%)")
print(f"  - Bottle body / other surface locations           : {body_count} / 31 ({body_count/31*100:.1f}%)")

print("\nPART A ANALYSIS COMPLETE.")
