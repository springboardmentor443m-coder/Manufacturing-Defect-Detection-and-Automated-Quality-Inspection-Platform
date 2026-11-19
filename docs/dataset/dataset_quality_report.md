# VisionInspect AI — Dataset Quality & Integrity Report

**Document ID:** VIA-DAT-QLT-001  
**Version:** 1.0.0  
**Phase:** Phase 2 — Dataset Acquisition & Exploratory Data Analysis (EDA)  
**Author:** Shrikrishna Chavan (Individual Contributor)  
**Branch:** `shrikrishna-chavan`  
**Status:** Approved Technical Report  

---

## 1. Quality Assurance Framework & Validation Rules

VisionInspect AI enforces a strict quality and integrity gate across all dataset assets. Quality rules are organized hierarchically:

```
+-------------------------------------------------------------------------------+
|                       DATASET QUALITY ASSURANCE GATE                          |
+-------------------------------------------------------------------------------+
| [1. Structural Layout]                                                        |
|   * Category Directory Existence                                              |
|   * Train ('good') and Test ('good' + defect subdirectories) Structure        |
|   * Ground Truth Subdirectory Alignment with Test Subtypes                   |
+-------------------------------------------------------------------------------+
| [2. Image Integrity & Pre-flight Decoding]                                    |
|   * Supported Container Formats: PNG, JPEG                                    |
|   * Byte Header Verification & Corrupt File Decompression Checking            |
|   * Minimum Spatial Resolution Threshold: >= 16x16 px                         |
|   * SHA-256 Duplicate Image Collision Detection                               |
+-------------------------------------------------------------------------------+
| [3. Ground-Truth Mask Integrity & Correspondence]                             |
|   * Strict 1:1 Defect Image to Mask Pairing                                   |
|   * Strict Spatial Dimension Match: (W_mask, H_mask) == (W_img, H_img)        |
|   * Strict Binary Pixel Values: Elements in {0, 255} (or {0, 1})              |
|   * Detection of Orphan / Unmatched Mask Files                                |
|   * Zero Alterations to Original Dataset Assets                               |
+-------------------------------------------------------------------------------+
```

---

## 2. Issue Severity Classification

The validation engine distinguishes non-blocking warnings from fatal errors:

### 2.1 Fatal Errors (`ERROR`) — Aborts Pipeline
| Issue Code | Trigger Condition | System Impact |
| :--- | :--- | :--- |
| `MISSING_DATASET_ROOT` | Target dataset directory does not exist | Cannot read data |
| `EMPTY_DATASET_ROOT` | No category subdirectories present | Pipeline has no inputs |
| `MISSING_CATEGORY_DIRECTORY` | Requested category folder missing | Cannot load category model |
| `MISSING_TRAIN_DIR` / `MISSING_TEST_DIR` | Core split directory missing | Corrupted benchmark layout |
| `MISSING_TRAIN_GOOD_DIR` | No `train/good` directory found | Normal-only training impossible |
| `EMPTY_IMAGE_DIRECTORY` | Directory contains zero image files | Zero training/eval tensors |
| `CORRUPTED_IMAGE` | Truncated binary or corrupt headers | PIL/OpenCV decode crash |
| `MISSING_GROUND_TRUTH_MASK` | Defective test image has no mask | Supervised loss / IoU impossible |
| `DIMENSION_MISMATCH` | Image size != Mask size | Spatial alignment failure |
| `INVALID_MASK_VALUES` | Mask pixels not in $\{0, 255\}$ | Corrupted ground-truth labels |

### 2.2 Non-Blocking Warnings (`WARNING`) — Logged & Flagged
| Issue Code | Trigger Condition | Operational Action |
| :--- | :--- | :--- |
| `UNEXPECTED_SYSTEM_FILE` | OS files (`.DS_Store`, `Thumbs.db`) | Ignored during dataset scan |
| `UNEXPECTED_FILE` | Stray `.txt` or non-image files | Flagged in audit report |
| `UNEXPECTED_TRAIN_SUBDIR` | Non-good folders inside `train/` | MVTec AD train should be normal-only |
| `UNMATCHED_GROUND_TRUTH_MASK` | Ground truth mask without test image | Orphan mask tracked in report |
| `DUPLICATE_IMAGE_CONTENT` | Identical SHA-256 hash across images | Flagged for data leakage check |

---

## 3. Automated Validation Test Suite Results

The validation engine was tested against a suite of 15 automated test cases covering nominal configurations and synthetic edge-case anomalies:

```
tests/test_dataset_validation.py::TestDatasetValidation::test_empty_dataset_directory PASSED
tests/test_dataset_validation.py::TestDatasetValidation::test_missing_category_directory PASSED
tests/test_dataset_validation.py::TestDatasetValidation::test_valid_normal_image PASSED
tests/test_dataset_validation.py::TestDatasetValidation::test_valid_defective_image_with_matching_mask PASSED
tests/test_dataset_validation.py::TestDatasetValidation::test_missing_mask PASSED
tests/test_dataset_validation.py::TestDatasetValidation::test_unmatched_mask PASSED
tests/test_dataset_validation.py::TestDatasetValidation::test_corrupted_image PASSED
tests/test_dataset_validation.py::TestDatasetValidation::test_invalid_mask_values PASSED
tests/test_dataset_validation.py::TestDatasetValidation::test_dimension_mismatch PASSED
tests/test_dataset_validation.py::TestDatasetValidation::test_unexpected_file PASSED
tests/test_dataset_validation.py::TestLicensingGate::test_licensing_gate_default_unapproved PASSED
tests/test_dataset_validation.py::TestLicensingGate::test_licensing_gate_explicit_confirmation PASSED
tests/test_dataset_validation.py::TestLicensingGate::test_licensing_gate_env_var PASSED
tests/test_dataset_validation.py::TestManifestAndEDA::test_manifest_builder PASSED
tests/test_dataset_validation.py::TestManifestAndEDA::test_eda_analyzer PASSED

============================= 15 passed in 1.85s ==============================
```

---

## 4. Known Dataset Limitations & Industrial Reality

Although MVTec AD 2019 is the accepted scientific standard for visual anomaly detection, several structural limitations must be accounted for in downstream architectural phases:

1. **Category Resolution Heterogeneity:**
   - Spatial resolutions differ across categories (`bottle`: $900\times 900$, `metal_nut`: $700\times 700$, `tile`: $840\times 840$).
   - *Mitigation:* Preprocessing pipelines in Phase 3 must standardize input tensors to canonical dimensions ($256\times 256$) via anti-aliased bilinear interpolation, while retaining original aspect dimensions for post-processing overlays.
2. **Defect Subtype Sample Scarcity:**
   - Individual defect subtypes often contain only 10–25 physical examples (e.g., `metal_nut/flip` has $\approx 19$ samples).
   - *Mitigation:* Supervised multi-class classification (ResNet18) is prone to severe overfitting under this distribution. It is retained strictly as an optional baseline. Normal-only one-class anomaly detection (PatchCore) avoids this limitation entirely.
3. **Controlled Illumination vs. Factory Floor Variance:**
   - MVTec AD images were acquired in controlled laboratory setups with fixed camera working distances and diffuse studio lighting.
   - *Mitigation:* Factory deployment will encounter vibration blur, dust accumulation, and lighting flicker. Models trained on MVTec AD must not be claimed as ready for uncalibrated factory deployment without domain adaptation.
4. **Lack of Metrological Calibration Targets:**
   - MVTec AD does not include millimeter/micron scaling factors, optical calibration grids, or camera focal length metadata.
   - *Mitigation:* Defect measurements produced by VisionInspect AI are strictly reported in pixel units ($px$) and area percentages ($\%$). Claims of physical millimeter accuracy are strictly prohibited.

---

## 5. Unresolved Issues & Phase 3 Transition Criteria

| Issue / Item | Current Status | Phase 3 Requirement |
| :--- | :--- | :--- |
| **Legal Acquisition Confirmation** | Pending explicit user/company confirmation | Must confirm before extracting actual `.tar.xz` archives |
| **Local Raw Data Extraction** | Unextracted (Gate active) | Extract to `data/raw/mvtec_ad/` upon approval |
| **Empirical Manifest Generation** | Tooling verified on synthetic data | Run `python -m src.dataset.cli build-manifest` on raw data |
| **Empirical EDA Plots** | Tooling verified on synthetic data | Run `python -m src.dataset.cli eda` on raw data |
| **Data Split Assignment** | Strictly deferred | Phase 3 will assign train/val/test splits preventing leakage |
