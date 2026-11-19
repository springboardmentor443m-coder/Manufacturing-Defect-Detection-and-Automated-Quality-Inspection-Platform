# VisionInspect AI — Dataset Structure & Inventory Specification

**Document ID:** VIA-DAT-STR-001  
**Version:** 1.0.0  
**Phase:** Phase 2 — Dataset Acquisition & Exploratory Data Analysis (EDA)  
**Author:** Shrikrishna Chavan (Individual Contributor)  
**Branch:** `shrikrishna-chavan`  
**Status:** Approved Specification  

---

## 1. Directory Hierarchy & Taxonomy

The original MVTec AD 2019 dataset enforces a standardized directory layout across all 15 industrial categories. The target categories for VisionInspect AI (`bottle`, `metal_nut`, `tile`) exhibit the following canonical organization under `data/raw/mvtec_ad/`:

```
data/raw/mvtec_ad/
├── bottle/
│   ├── train/
│   │   └── good/                     # Defect-free normal training images
│   │       ├── 000.png
│   │       └── ...
│   ├── test/
│   │   ├── good/                     # Defect-free normal evaluation images
│   │   ├── broken_large/             # Defect subtype A
│   │   ├── broken_small/             # Defect subtype B
│   │   └── contamination/            # Defect subtype C
│   └── ground_truth/
│       ├── broken_large/
│       │   ├── 000_mask.png          # Binary mask (0=normal, 255=defect)
│       │   └── ...
│       ├── broken_small/
│       └── contamination/
│
├── metal_nut/
│   ├── train/
│   │   └── good/
│   ├── test/
│   │   ├── good/
│   │   ├── bent/
│   │   ├── color/
│   │   ├── flip/
│   │   └── scratch/
│   └── ground_truth/
│       ├── bent/
│       ├── color/
│       ├── flip/
│       └── scratch/
│
└── tile/
    ├── train/
    │   └── good/
    ├── test/
    │   ├── good/
    │   ├── crack/
    │   ├── glue_strip/
    │   ├── gray_stroke/
    │   ├── oil/
    │   └── rough/
    └── ground_truth/
        ├── crack/
        ├── glue_strip/
        ├── gray_stroke/
        ├── oil/
        └── rough/
```

---

## 2. Image and Ground-Truth Mask Correspondence

### 2.1 File Naming & Matching Protocol
- **Test Defect Image:** `data/raw/mvtec_ad/{category}/test/{defect_type}/{index}.png`
- **Corresponding Ground-Truth Mask:** `data/raw/mvtec_ad/{category}/ground_truth/{defect_type}/{index}_mask.png`
- **Image-Mask Spatial Alignment:**
  - The mask spatial dimensions $(W_{\text{mask}}, H_{\text{mask}})$ must **strictly match** the test image spatial dimensions $(W_{\text{img}}, H_{\text{img}})$.
  - Pixel coordinates $(x, y)$ in the mask correspond to the exact optical location in the source RGB image.

### 2.2 Normal Images & Implicit Masks
- Normal images in `train/good/` and `test/good/` **do not have physical ground-truth mask files** in MVTec AD.
- Creating physical files filled with zeros on disk is prohibited during Phase 2 to prevent altering or duplicating the original dataset structure.
- Downstream segmentation pipelines (Phase 3+) will synthesize in-memory zero arrays on the fly when processing normal samples. In the inventory manifest, normal samples record `mask_path: null`.

### 2.3 Mask Encoding & Value Range
- **File Format:** 1-channel Grayscale 8-bit PNG (`image/png`, depth `L`).
- **Valid Pixel Values:**
  - `0`: Background (Nominal / Defect-free surface).
  - `255`: Foreground (Defective / Anomaly surface region).
- Any intermediate grayscale pixel values (e.g., anti-aliased boundaries like `128`) are flagged by the validator as `INVALID_MASK_VALUES`.

---

## 3. Preliminary Inventory Manifest Schema

To decouple data consumption from filesystem scanning, VisionInspect AI generates a structured preliminary inventory manifest (`data/reports/dataset_manifest.json` and `dataset_manifest.csv`).

### 3.1 Field Specifications

| Field Name | Data Type | Nullable | Description | Example |
| :--- | :--- | :--- | :--- | :--- |
| `image_path` | String | No | Relative POSIX path from dataset root | `"bottle/test/broken_large/000.png"` |
| `mask_path` | String | Yes | Relative POSIX path to mask, or `null` for normal | `"bottle/ground_truth/broken_large/000_mask.png"` |
| `category` | String | No | Category identifier | `"bottle"` |
| `split` | String | No | Original folder split (`"train"` or `"test"`) | `"test"` |
| `defect_type` | String | No | Subtype name (`"good"` for normal samples) | `"broken_large"` |
| `is_anomalous`| Boolean | No | `False` if normal, `True` if defective | `true` |
| `image_width` | Integer | No | Image width in pixels | `1024` |
| `image_height`| Integer | No | Image height in pixels | `1024` |
| `image_format`| String | No | Encoding format | `"PNG"` |
| `file_size_bytes` | Integer | No | Exact file size in bytes | `1245082` |
| `image_sha256`| String | No | 64-character SHA-256 cryptographic checksum | `"8f4a132bc6e7..."` |
| `validation_status` | String | No | Integrity status (`"VALID"`, `"WARNING"`, `"ERROR"`) | `"VALID"` |

> [!IMPORTANT]
> **Inventory Manifest Scope Restriction:**  
> This manifest serves purely as an inventory of physical assets on disk. **No model train/validation/test experimental splits are assigned in Phase 2.** Split generation and leakage prevention protocols belong strictly to Phase 3.
