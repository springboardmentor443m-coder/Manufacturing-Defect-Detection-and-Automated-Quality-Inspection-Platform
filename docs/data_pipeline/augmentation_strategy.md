# VisionInspect AI — Data Augmentation Strategy

**Document ID:** VIA-PIP-AUG-001  
**Version:** 1.0.0  
**Phase:** Phase 3 — Data Preprocessing, Dataset Splitting & Augmentation Pipeline  
**Author:** Shrikrishna Chavan (Individual Contributor)  
**Branch:** `shrikrishna-chavan`  
**Status:** Approved Technical Design  

---

## 1. Objectives & Physics-Based Principles

In automated manufacturing inspection, data augmentation must be **physically plausible**. Blindly applying arbitrary computer vision augmentations (e.g., vertical flips on upright bottles, extreme perspective warps) corrupts industrial priors and causes models to learn unphysical failure modes.

The VisionInspect AI augmentation engine (`src/preprocessing/augmentation.py`) operates under three core principles:
1. **Strict Spatial Synchronization:** Any geometric transformation (flip, rotation, affine shift) applied to an image must apply identically to its paired ground-truth mask.
2. **Category-Specific Physical Invariance:** Augmentations must reflect authentic mechanical tolerances and conveyor orientations for that specific product line.
3. **Strict Evaluation Isolation:** Random augmentations occur strictly on-the-fly during training. Validation and test partitions must **never** receive random augmentations.

---

## 2. Category-Specific Augmentation Rules

| Category | Product Geometry | Horizontal Flip | Vertical Flip | 90° Rotations | Color Jitter (Brightness / Contrast) | Physical Justification |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`bottle`** | Upright Rigid Translucent Object | **Enabled** ($p=0.5$) | **DISABLED** | **DISABLED** | Subtle ($\pm 10\%$) | Bottles travel strictly upright on conveyor lines. An inverted or horizontally rotated bottle violates real manufacturing physics. |
| **`metal_nut`** | Hexagonal Machined Hardware | **Enabled** ($p=0.5$) | **Enabled** ($p=0.5$) | **Enabled** ($p=0.5$) | Subtle ($\pm 10\%$) | Machined nuts can land on conveyor trays at arbitrary $360^\circ$ planar rotations and inverted faces. |
| **`tile`** | Planar Repeating Texture | **Enabled** ($p=0.5$) | **Enabled** ($p=0.5$) | **Enabled** ($p=0.5$) | Subtle ($\pm 10\%$) | Ceramic tiles exhibit full $D_4$ dihedral group symmetry (8 distinct rotation/reflection states). |

---

## 3. Synchronized Spatial Transformation Mechanics

```
                         [Random Decision Generator (RNG)]
                                    |
            +-----------------------+-----------------------+
            |                                               |
            v                                               v
   [Image Tensor: (3, H, W)]                       [Mask Tensor: (1, H, W)]
            |                                               |
            v                                               v
   [Geometric Transform]                          [Identical Geometric Transform]
   (Horizontal / Vertical Flip,                   (Nearest-Neighbor Interpolation
    90° Rotation)                                  Preserving Strict Binary Values)
            |                                               |
            v                                               v
   [Color Jitter Stage]                                    |
   (Brightness & Contrast ±10%)                            |
   (APPLIES TO IMAGE ONLY!)                                |
            |                                               |
            +-----------------------+-----------------------+
                                    |
                                    v
                 [Synchronized Pair: Image (3, H, W), Mask (1, H, W)]
```

### 3.1 Mask Binary Value Invariance
During spatial rotation or interpolation, nearest-neighbor sampling (`TF.InterpolationMode.NEAREST`) is enforced for masks. After transformation, the augmentor enforces a thresholding gate:
$$\text{Mask}_{\text{final}} = (\text{Mask}_{\text{aug}} > 0.5).\text{float32}()$$
This guarantees that masks strictly retain binary $\{0.0, 1.0\}$ representations without intermediate gray transitions.

### 3.2 Photometric Decoupling
Photometric transformations (brightness and contrast jitter) simulate variations in industrial strobe illumination, LED aging, and ambient light. Because defects are defined by their spatial presence rather than lighting conditions, photometric jitter is applied **strictly to the RGB image tensor**, leaving the mask tensor unaltered.

---

## 4. Evaluation and Test Isolation

To maintain statistical rigor:
1. **Training Split:** Passes through `SynchronizedAugmentor` when `is_train=True`.
2. **Validation Split:** Evaluated strictly on canonical preprocessed inputs (`is_train=False`). Random augmentations are bypassed.
3. **Test Split:** Evaluated strictly on canonical held-out inputs (`is_train=False`). Deterministic evaluation guarantees reproducible benchmarks.
