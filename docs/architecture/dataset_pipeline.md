# VisionInspect AI — Dataset Pipeline & Data-Splitting Architecture

**Document ID:** VIA-ARCH-DATASET-001  
**Version:** 1.0.0  
**Phase:** Phase 1 — Requirements Analysis & System Architecture Design  
**Author:** Shrikrishna Chavan (Individual Contributor)  
**Branch:** `shrikrishna-chavan`  
**Status:** Approved for Design Baseline  

---

## 1. Overview & Dataset Selection

VisionInspect AI is evaluated against the authoritative industrial benchmark: **MVTec Anomaly Detection (MVTec AD) 2019**. The dataset contains high-resolution industrial photographs depicting real-world industrial objects and textures with authentic defects (e.g. scratches, dents, structural breaks, contamination).

Initial Target Categories:
1. **`bottle`** (Rigid object with translucent material, liquid level anomalies, contamination, and structural fractures).
2. **`metal_nut`** (Precision machined hardware with thread deformation, scratches, discoloration, and foreign particles).
3. **`tile`** (Structural repeating surface texture with cracks, glue spots, gray stroke blemishes, and oil stains).

---

## 2. Licensing & Legal Compliance Warning

> [!CAUTION]
> **Dataset Licensing Restriction (CC BY-NC-SA 4.0):**  
> The MVTec AD dataset is released under the **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License**.  
> - **Permitted:** Non-commercial academic research, open-source benchmarking, and educational prototyping.  
> - **Prohibited:** Commercial deployment, incorporation into proprietary products sold for commercial gain, or factory deployment for profit without explicit commercial licensing from MVTec Software GmbH.  
> - **Corporate Requirement:** Formal legal approval is strictly required before utilizing models trained on MVTec AD within commercial production environments. Any commercial deployment must train exclusively on proprietary, enterprise-owned, or commercially licensed dataset acquisitions.

---

## 3. Directory Layout & Dataset Structure

The canonical dataset filesystem layout on disk:

```
data/
└── mvtec_ad/
    ├── bottle/
    │   ├── train/
    │   │   └── good/                     # Normal, defect-free images
    │   │       ├── 000.png
    │   │       └── ...
    │   ├── test/
    │   │   ├── good/                     # Normal test images
    │   │   ├── broken_large/             # Defect subtype 1
    │   │   ├── broken_small/             # Defect subtype 2
    │   │   └── contamination/            # Defect subtype 3
    │   └── ground_truth/
    │       ├── broken_large/
    │       │   ├── 000_mask.png          # Pixel-accurate binary mask (0=bg, 255=defect)
    │       │   └── ...
    │       ├── broken_small/
    │       └── contamination/
    ├── metal_nut/
    │   ├── train/
    │   ├── test/
    │   └── ground_truth/
    └── tile/
        ├── train/
        ├── test/
        └── ground_truth/
```

---

## 4. Dataset Manifest Schema

To ensure 100% reproducibility and prevent filesystem race conditions, all dataset access occurs via immutable, versioned JSON manifests generated prior to training.

### JSON Manifest Record Schema:
```json
{
  "dataset_name": "mvtec_ad_2019",
  "dataset_version": "1.0.0",
  "created_at": "2026-11-17T20:00:00Z",
  "random_seed": 42,
  "samples": [
    {
      "sample_id": "bottle_test_broken_large_000",
      "category": "bottle",
      "split": "test",
      "protocol": "held_out_benchmark",
      "is_anomalous": true,
      "defect_type": "broken_large",
      "image_rel_path": "bottle/test/broken_large/000.png",
      "mask_rel_path": "bottle/ground_truth/broken_large/000_mask.png",
      "image_sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
      "mask_sha256": "ca978112ca1bbdcafac231b39a23dc4da786081496a8be6b5779c13b3026aa60",
      "resolution": [1024, 1024],
      "channels": 3
    },
    {
      "sample_id": "bottle_train_good_000",
      "category": "bottle",
      "split": "train",
      "protocol": "patchcore_normal_only",
      "is_anomalous": false,
      "defect_type": "good",
      "image_rel_path": "bottle/train/good/000.png",
      "mask_rel_path": null,
      "image_sha256": "4b227777d4dd1fc61c6f884f48641d02b4d121d3fd328cb08b5531fcacdabf8a",
      "mask_sha256": null,
      "resolution": [1024, 1024],
      "channels": 3
    }
  ]
}
```

---

## 5. Splitting Strategy & Data Leakage Prevention

### 5.1 The Official MVTec AD Protocol vs Supervised Benchmark Challenge
The official MVTec AD split structure was designed specifically for **unsupervised anomaly detection** (like PatchCore):
- **Official Train Set:** Contains *strictly* normal (good) images. Zero defect images.
- **Official Test Set:** Contains normal (good) images and all defect classes with corresponding ground truth pixel masks.

Because supervised models (like U-Net) require defective samples during training, naive practitioners often train U-Net on the official test set and evaluate it on the same set, causing catastrophic data leakage.

### 5.2 Strict Split Protocol Design

VisionInspect AI defines two strictly segregated data protocols:

```
+---------------------------------------------------------------------------------+
|                       OFFICIAL MVTEC AD PARTITIONING                            |
+---------------------------------------------------------------------------------+
| Train Folder (100% Good)                   | Test Folder (Good + Defects)       |
+--------------------------------------------+------------------------------------+
|                                            |                                    |
| [Protocol A: Official PatchCore]           | [Protocol B: Supervised U-Net]     |
|                                            |                                    |
|  * Train Good -> PatchCore Training (80%)  |  * Test Partition (Strictly Split):|
|  * Val Good   -> Threshold Tuning  (20%)   |      - 50% Stratified -> U-Net Train|
|                                            |      - 50% Stratified -> HELD-OUT  |
|                                            |                          TEST SET  |
+---------------------------------------------------------------------------------+
|                      COMMON HELD-OUT TEST EVALUATION                            |
|  Both PatchCore and U-Net are evaluated on the exact same 50% HELD-OUT TEST SET |
+---------------------------------------------------------------------------------+
```

#### Protocol A: PatchCore (Unsupervised / One-Class)
- **PatchCore Training Set:** $80\%$ of the official `train/good/` images, selected via fixed random seed (`seed=42`).
- **Validation Calibration Set:** $20\%$ of the official `train/good/` images. Used strictly to calibrate image-level and pixel-level anomaly thresholds ($\tau_{\text{pass}}$ at the 99th percentile).
- **Test Evaluation:** Evaluated on the held-out test partition.

#### Protocol B: U-Net Supervised Comparative Protocol
- To benchmark U-Net fairly, the official test set (which contains defects) is split using **stratified defect-type sampling** into:
  - **U-Net Supervised Train Set (50% of defects + 50% of test good):** Paired with ground truth masks for backpropagation training.
  - **U-Net Supervised Validation Set (10% of defects):** Used for early stopping and mIoU checkpoint selection.
  - **Held-Out Test Set (40% of defects + remaining test good):** Strictly sequestered. **Neither PatchCore nor U-Net ever observes these images during training or threshold tuning.**

#### Rules for Preventing Data Leakage:
1. **Zero Overlap:** The test set is isolated prior to any model experimentation.
2. **No Threshold Tuning on Test Data:** Calibration of $\tau_{\text{binary}}$, $\tau_{\text{pass}}$, or QC rules on the test set is strictly prohibited. All thresholds are calibrated on the validation sets.
3. **Identical Test Subset for Comparison:** When comparing PatchCore and U-Net performance metrics, both models must be scored on the **exact same held-out test split**, while noting that U-Net utilized defect supervision on the complementary partition.

---

## 6. Reproducibility & Random Seed Management

To guarantee bit-exact dataset splitting across independent execution environments:
- The manifest generator script fixes random seeds across all libraries:
  ```python
  import random
  import os
  import numpy as np
  import torch

  def set_reproducible_seed(seed: int = 42) -> None:
      random.seed(seed)
      os.environ["PYTHONHASHSEED"] = str(seed)
      np.random.seed(seed)
      torch.manual_seed(seed)
      torch.cuda.manual_seed_all(seed)
      torch.backends.cudnn.deterministic = True
      torch.backends.cudnn.benchmark = False
  ```
- The split manifest JSON records the seed integer (`"random_seed": 42`).
- Dataset manifests are committed to git version control, ensuring all team members train on identical splits.

---

## 7. Dataset Integrity & Pre-computation Verification

The dataset pipeline executes an automated pre-flight integrity check prior to training:
1. **File Existence & Missing Link Check:** Verifies every path in the manifest resolves to an existing file on disk.
2. **SHA-256 Checksum Validation:** Computes cryptographic hashes for all images and masks, verifying against stored manifest values.
3. **Format & Dimension Integrity:** Validates that image files are uncorrupted 3-channel images, and that ground truth masks are 1-channel images matching the exact spatial dimensions $(H, W)$ of their corresponding source photographs.
4. **Binary Value Range:** Verifies mask pixel values contain strictly binary sets $\{0, 255\}$ (or $\{0, 1\}$).
