# VisionInspect AI — Phase 3B: Actual Dataset Validation Report

## 1. Executive Summary

- **Component:** Dataset Acquisition & Physical Validation Module
- **Target Dataset:** Original MVTec Anomaly Detection (MVTec AD) 2019
- **Categories:** `bottle`, `metal_nut`, `tile`
- **Execution Date:** 2026-11-19
- **Branch:** `shrikrishna-chavan`
- **Validation Status:** **PASSED (100% VALID)**
- **Total Images Scanned:** 974
- **Total Masks Scanned:** 240
- **Validation Errors:** 0
- **Validation Warnings:** 6 (Official root metadata `license.txt` and `readme.txt`)

Formal project owner authorization for non-commercial educational/evaluative research use under Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0) was registered. All three category archives were securely unpacked using path-traversal-safe extraction, scanned, validated, and confirmed to have zero data integrity violations.

---

## 2. Raw Archive Inventory & Verification

All three archives inside `data/raw/` were verified before extraction:

| Category | File Name | Archive Path | Size (Bytes) | Size (MB) | SHA-256 Checksum | Tar Header Integrity |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Bottle** | `bottle.tar.xz` | `data/raw/bottle.tar.xz` | 155,880,244 | ~148.66 MB | `726512129cb3b1f47d4f6cd7c7fc9170db7fc209132249c788fec06029cb5dc3` | Valid (`True`) |
| **Metal Nut** | `metal_nut.tar.xz` | `data/raw/metal_nut.tar.xz` | 165,414,484 | ~157.75 MB | `86e3c8e163ceb19146ad3ddc408e85cc34ae02e0da430e07a73bac6bbb48532a` | Valid (`True`) |
| **Tile** | `tile.tar.xz` | `data/raw/tile.tar.xz` | 351,279,732 | ~335.01 MB | `392228153b96d0db9f179d3f7c2d63e131e907154883b57309fcbf798643173c` | Valid (`True`) |
| **Total** | 3 archives | `data/raw/` | **672,574,460** | **~641.42 MB** | — | **All Valid** |

Original downloaded archive files were strictly preserved and remain intact in `data/raw/`.

---

## 3. Safe Extraction Audit

Extraction was executed via `DatasetAcquisitionManager.safe_extract_tar` targeting `data/raw/mvtec_ad/`:

```text
2026-11-19 13:40:45 [INFO] Starting safe extraction for categories: ['bottle', 'metal_nut', 'tile']
2026-11-19 13:40:51 [INFO] Extracted category 'bottle' successfully.
2026-11-19 13:40:57 [INFO] Extracted category 'metal_nut' successfully.
2026-11-19 13:41:04 [INFO] Extracted category 'tile' successfully.
```

- **Extraction Security Checks:**
  - Archive member path resolution verified against target directory root.
  - Zero absolute paths detected.
  - Zero directory traversal sequences (`..`) detected.
  - Zero symlinks, hardlinks, FIFOs, or device files detected.
- **Archive Member Counts:**
  - `bottle.tar.xz`: 369 members
  - `metal_nut.tar.xz`: 444 members
  - `tile.tar.xz`: 449 members
  - **Total Members Unpacked:** 1,262
- **Extracted Hierarchy:**
  - `data/raw/mvtec_ad/bottle/`
  - `data/raw/mvtec_ad/metal_nut/`
  - `data/raw/mvtec_ad/tile/`
  No duplicate nested directories were created.

---

## 4. Physical Dataset Validation Findings

The automated validator `DatasetValidator` scanned the entire physical directory tree:

### 4.1 Per-Category Sample & Mask Audit

| Category | Split | Normal Samples | Defective Samples | Total Images | Masks | Subtypes Verified |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Bottle** | Train | 209 | 0 | 209 | 0 | None (Normal-only) |
| | Test | 20 | 63 | 83 | 63 | `broken_large` (20), `broken_small` (22), `contamination` (21) |
| | **Subtotal**| **229** | **63** | **292** | **63** | 3 subtypes |
| **Metal Nut** | Train | 220 | 0 | 220 | 0 | None (Normal-only) |
| | Test | 22 | 93 | 115 | 93 | `bent` (25), `color` (22), `flip` (23), `scratch` (23) |
| | **Subtotal**| **242** | **93** | **335** | **93** | 4 subtypes |
| **Tile** | Train | 230 | 0 | 230 | 0 | None (Normal-only) |
| | Test | 33 | 84 | 117 | 84 | `crack` (17), `glue_strip` (18), `gray_stroke` (16), `oil` (18), `rough` (15) |
| | **Subtotal**| **263** | **84** | **347** | **84** | 5 subtypes |
| **Overall** | **Total** | **734** | **240** | **974** | **240** | **12 distinct defect subtypes** |

### 4.2 Data Integrity Check Results
- **Image Readability:** 974 / 974 images loaded successfully without decompression errors.
- **Mask Correspondence:** 240 / 240 defective test images possess exactly one matching mask (`<stem>_mask.png`).
- **Unmatched / Orphan Masks:** 0.
- **Dimension Matching:** 100% matched. Every mask matches the pixel width and height of its source image.
  - Bottle: $900 \times 900$ (Image and Mask identical)
  - Metal Nut: $700 \times 700$ (Image and Mask identical)
  - Tile: $840 \times 840$ (Image and Mask identical)
- **Mask Binary Pixel Values:** All mask arrays strictly contain binary values $\{0, 255\}$. Zero intermediate artifact values (e.g., 128) detected.
- **Corrupted / Truncated Files:** 0.
- **Duplicate Images:** 0 duplicates detected across SHA-256 hashes.

### 4.3 Validation Issues Logged
`data/reports/dataset_validation_issues.csv` recorded 6 non-blocking warnings:
- `WARNING,bottle,UNEXPECTED_FILE,bottle/license.txt`
- `WARNING,bottle,UNEXPECTED_FILE,bottle/readme.txt`
- `WARNING,metal_nut,UNEXPECTED_FILE,metal_nut/license.txt`
- `WARNING,metal_nut,UNEXPECTED_FILE,metal_nut/readme.txt`
- `WARNING,tile,UNEXPECTED_FILE,tile/license.txt`
- `WARNING,tile,UNEXPECTED_FILE,tile/readme.txt`

These text files represent original distribution documentation bundled by MVTec Software GmbH and do not interfere with the image loading pipeline.

---

## 5. Discrepancy Note: Tile Normal Count

- **Theoretical Literature Figure:** Frequently cited as 245 train normal samples in preliminary CVPR 2019 benchmark papers.
- **Empirical Measured Figure:** The official downloaded `tile.tar.xz` archive contains exactly **230** train normal samples (`000.png` through `229.png`).
- **Resolution:** VisionInspect AI strictly utilizes measured reality (230 train normal samples) in all manifests, splits, and calculations. Zero synthetic filler files were added.
