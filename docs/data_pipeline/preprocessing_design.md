# VisionInspect AI — Image & Mask Preprocessing Design

**Document ID:** VIA-PIP-PRE-001  
**Version:** 1.0.0  
**Phase:** Phase 3 — Data Preprocessing, Dataset Splitting & Augmentation Pipeline  
**Author:** Shrikrishna Chavan (Individual Contributor)  
**Branch:** `shrikrishna-chavan`  
**Status:** Approved Technical Design  

---

## 1. Overview & Objectives

In automated visual quality inspection, preprocessing must strike an uncompromising balance between computational efficiency, model compatibility, and **defect preservation fidelity**. Industrial surface defects (pinholes, micro-cracks, thin scratches) often occupy fewer than 10 pixels in high-resolution source imagery ($700\times 700$ to $1024\times 1024$).

The VisionInspect AI preprocessing subsystem (`src/preprocessing/`) standardizes visual inputs into calibrated PyTorch tensors while actively auditing against defect disappearance.

---

## 2. Image Preprocessing Pipeline

### 2.1 Sequential Transformations
The pipeline executes seven discrete, deterministic transformation stages:

```
[Raw Image / Bytes / File Path]
               |
               v
 [1. Safe Loading & Integrity Verification]
     - Verifies byte buffer decodability
     - Enforces dimension bounds: 16 <= W, H <= 4096
               |
               v
 [2. Color-Space Normalization to RGB]
     - Grayscale (1-channel 'L') replicated across 3 channels
     - RGBA (4-channel) composited against black background
     - Palette ('P') / CMYK converted to 8-bit RGB
               |
               v
 [3. Anti-Aliased Bilinear Resizing]
     - Scaled to configurable target resolution (Default: 256 x 256)
     - Uses PIL.Image.Resampling.BILINEAR with antialiasing filters
               |
               v
 [4. Tensor Conversion & Dimension Permutation]
     - Scaled from uint8 [0, 255] to float32 [0.0, 1.0]
     - Permuted from Channels-Last (H, W, 3) to Channels-First (3, H, W)
               |
               v
 [5. ImageNet Standardization]
     - Channel-wise mean subtraction and division by standard deviation:
       mu = [0.485, 0.456, 0.406],  sigma = [0.229, 0.224, 0.225]
               |
               v
 [6. Metadata & Dimension Caching]
     - Preserves original spatial resolution (orig_w, orig_h) and color mode
     - Enables exact coordinate re-mapping during post-processing and overlay generation
```

---

## 3. Mask Processing & Defect Preservation Architecture

### 3.1 Binary Label Integrity
- **Ground-Truth Format:** 1-channel Grayscale 8-bit image.
- **Strict Binary Validation:** Values must strictly belong to the set $\{0, 255\}$ (or $\{0, 1\}$). Intermediate gray values (e.g., $128$) indicate corrupted or anti-aliased ground-truth annotations and immediately trigger an explicit `ValueError`.
- **Implicit Normal Masks:** Normal training and test samples synthesize an in-memory zero tensor of shape `(1, H, W)` without requiring physical empty mask files on disk.

### 3.2 The Nearest-Neighbor Resizing Mandate
Binary segmentation masks must **never** be resized using bilinear, bicubic, or Lanczos interpolation. Resampling continuous curves over binary boundaries synthesizes invalid intermediate floating-point values ($0.1, 0.4, 0.7$).  
VisionInspect AI strictly mandates **Nearest-Neighbor Interpolation (`PIL.Image.Resampling.NEAREST`)** for all mask transformations, guaranteeing output pixel sets strictly remain $\{0.0, 1.0\}$.

---

## 4. Small Defect Disappearance Evaluation

### 4.1 The Area Downsampling Hazard
Downsampling an image from $1024\times 1024$ to $256\times 256$ represents a $16\times$ reduction in total pixel area ($4\times$ reduction per spatial axis). A fine scratch spanning 2 pixels in width can fall between sampling points during nearest-neighbor selection and completely vanish.

```
+-------------------------------------------------------------------------------+
|                       SPATIAL DOWNSAMPLING HAZARD                             |
+-------------------------------------------------------------------------------+
| Source Mask (1024 x 1024):                Resized Mask (256 x 256):           |
| Total Pixels: 1,048,576                   Total Pixels: 65,536                |
| Small Scratch Area: 4 pixels              Nearest-Neighbor Subsampling:       |
| Relative Area: 0.00038%                   Result: 0 PIXELS (DEFECT VANISHED!) |
+-------------------------------------------------------------------------------+
```

### 4.2 Automated Defect Loss Auditing
The `MaskProcessor` computes quantitative defect retention metrics for every processed mask:

1. **Pre-Resize Metrics:** Computes original defective pixel count $A_{\text{orig}}$ and surface ratio $\rho_{\text{orig}} = \frac{A_{\text{orig}}}{W_{\text{orig}} \times H_{\text{orig}}} \times 100\%$.
2. **Post-Resize Metrics:** Computes resized defective pixel count $A_{\text{resized}}$ and surface ratio $\rho_{\text{resized}} = \frac{A_{\text{resized}}}{W_{\text{target}} \times H_{\text{target}}} \times 100\%$.
3. **Disappearance Detection (`defect_disappeared`):**
   $$\text{defect\_disappeared} = (A_{\text{orig}} > 0) \land (A_{\text{resized}} == 0)$$
   When triggered, a high-priority warning is logged, alerting quality engineers that the chosen resolution eliminates the defect signal.
4. **Significant Loss Detection (`significant_defect_loss`):**
   Calculates deviation from expected scaled area:
   $$\text{Loss\%} = \max\left(0, \frac{A_{\text{orig}} \cdot \left(\frac{W_{\text{target}} \cdot H_{\text{target}}}{W_{\text{orig}} \cdot H_{\text{orig}}}\right) - A_{\text{resized}}}{A_{\text{orig}} \cdot \left(\frac{W_{\text{target}} \cdot H_{\text{target}}}{W_{\text{orig}} \cdot H_{\text{orig}}}\right)}\right) \times 100\%$$
   If $\text{Loss\%} > 50.0\%$, the mask is flagged as having suffered substantial defect attenuation.

---

## 5. Resolution Trade-Off Analysis

| Strategy | Spatial Resolution | VRAM Footprint | Inference Latency (GPU) | Small Defect Preservation | Engineering Complexity |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Fixed Resizing (Baseline)** | $256 \times 256$ | Low ($\approx 1.2\text{ GB}$) | Ultra-fast ($\approx 80\text{ ms}$) | Moderate (Risk of micro-defect loss) | Low (Single forward pass) |
| **High-Resolution Resizing** | $512 \times 512$ | Medium ($\approx 3.5\text{ GB}$) | Fast ($\approx 220\text{ ms}$) | High (Preserves $> 95\%$ of defects) | Low (Single forward pass) |
| **Tiled / Patch Inference** | Sliding $256\times 256$ patches | Low | Slow ($N_{\text{tiles}} \times 60\text{ ms}$) | Perfect ($1:1$ native sensor resolution) | High (Border blending, tiling stitch) |

**Architectural Recommendation:**  
- For Phase 3 and initial prototype benchmarking, the configurable baseline resolution is established at **$256 \times 256$**.
- Because `ImagePreprocessConfig.target_size` and `MaskPreprocessConfig.target_size` are fully decoupled and parameter-driven, scaling to $512 \times 512$ or dynamic resolutions requires zero code refactoring.
