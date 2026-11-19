# VisionInspect AI — Phase 3B: Actual Dataset Exploratory Data Analysis (EDA) Report

## 1. Scope & Execution Summary

- **Dataset:** Original MVTec Anomaly Detection (MVTec AD) 2019
- **Categories Analyzed:** `bottle`, `metal_nut`, `tile`
- **Execution Date:** 2026-11-19
- **Status:** **COMPLETED ON PHYSICAL EXTRACTED DATA**
- **Inventory Records Analyzed:** 974 images, 240 defective ground-truth masks
- **Output Artifacts:**
  - Manifest CSV: `data/reports/dataset_manifest.csv` (974 rows)
  - Manifest JSON: `data/reports/dataset_manifest.json` (974 objects)
  - Statistical Summary: `data/reports/eda_statistical_summary.json`
  - Visualizations: 36 multi-panel inspection PNGs in `data/visualizations/`

All values in this report represent empirically measured figures computed directly from the extracted image and mask files.

---

## 2. Empirical Category-Level Profiles

| Attribute | Bottle | Metal Nut | Tile | Overall Total |
| :--- | :--- | :--- | :--- | :--- |
| **Physical Archetype** | Rigid Cylindrical Object | Structural Rotational Hardware | Planar Structural Texture | 3 Industrial Domains |
| **Resolution ($W \times H$)** | $900 \times 900$ | $700 \times 700$ | $840 \times 840$ | All Square ($1.00$ aspect ratio) |
| **Color Channels** | 3 (RGB) | 3 (RGB) | 3 (RGB) | 100% 3-Channel RGB |
| **Image Format** | PNG | PNG | PNG | 100% Lossless PNG |
| **Train Normal Images** | 209 | 220 | 230 | **659** |
| **Test Normal Images** | 20 | 22 | 33 | **75** |
| **Test Defective Images** | 63 | 93 | 84 | **240** |
| **Ground-Truth Masks** | 63 | 93 | 84 | **240** |
| **Total Images** | **292** | **335** | **347** | **974** |
| **Mean File Size (KB)** | 522.8 KB | 482.6 KB | 988.9 KB | ~664.8 KB average |

---

## 3. Defect Area Statistics & Physical Scale

Defect foreground pixel areas were calculated from the binary masks ($M(x, y) = 255$) relative to the image resolution ($W \times H$).

### 3.1 Defect Area Metric Distribution

| Metric | Bottle ($900\times 900 = 810\text{k px}$) | Metal Nut ($700\times 700 = 490\text{k px}$) | Tile ($840\times 840 = 705.6\text{k px}$) | Overall Combined Defect Set |
| :--- | :--- | :--- | :--- | :--- |
| **Min Defect Area (px)** | 4,663 px | 2,135 px | 5,942 px | 2,135 px |
| **Max Defect Area (px)** | 222,091 px | 238,751 px | 250,395 px | 250,395 px |
| **Mean Defect Area (px)** | 61,694 px | 70,995 px | 69,165 px | 67,918 px |
| **Median Defect Area (px)** | 46,314 px | 17,990 px | 55,957 px | 35,422 px |
| **Min Defect Area (%)** | **0.576%** | **0.436%** | **0.842%** | **0.436%** |
| **25th Percentile ($p_{25}$ %)**| 2.574% | 1.898% | 3.076% | 2.214% |
| **Median ($p_{50}$ %)** | **5.718%** | **3.671%** | **7.930%** | **5.845%** |
| **75th Percentile ($p_{75}$ %)**| 12.829% | 12.301% | 13.941% | 13.018% |
| **95th Percentile ($p_{95}$ %)**| 15.512% | 48.489% | 23.314% | 35.487% |
| **Max Defect Area (%)** | **27.419%** | **48.725%** | **35.487%** | **48.725%** |
| **Mean Defect Area (%)** | **7.617%** | **14.489%** | **9.802%** | **11.045%** |

---

## 4. Defect Subtype Breakdown & Imbalance Analysis

### 4.1 Bottle (63 Defective Samples)
- `broken_large`: 20 samples ($31.7\%$) — Major structural fractures spanning up to $27.4\%$ of surface area.
- `broken_small`: 22 samples ($34.9\%$) — Micro-cracks along bottle neck and mouth ($0.58\% - 3.2\%$ area).
- `contamination`: 21 samples ($33.3\%$) — Foreign particles and dust deposits inside the translucent body.
- *Imbalance assessment:* Well-balanced ($\sim 20-22$ samples per subtype).

### 4.2 Metal Nut (93 Defective Samples)
- `bent`: 25 samples ($26.9\%$) — Severe physical deformation altering outer polygonal contour.
- `color`: 22 samples ($23.7\%$) — Thermal discoloration and chemical oxidation.
- `flip`: 23 samples ($24.7\%$) — Inverted installation orientation exposing threaded base ($>45\%$ defect area).
- `scratch`: 23 samples ($24.7\%$) — Fine surface scratches ($0.44\% - 2.1\%$ area).
- *Imbalance assessment:* Well-balanced ($\sim 22-25$ samples per subtype).

### 4.3 Tile (84 Defective Samples)
- `crack`: 17 samples ($20.2\%$) — Thin structural fissures traversing textured ceramic.
- `glue_strip`: 18 samples ($21.4\%$) — Adhesive residue strips.
- `gray_stroke`: 16 samples ($19.0\%$) — Subtle localized pigmentation smudges.
- `oil`: 18 samples ($21.4\%$) — Liquid contamination darkening surface reflectance.
- `rough`: 15 samples ($17.9\%$) — Surface abrasive wear disrupting pattern regularity.
- *Imbalance assessment:* Moderately balanced in the actual distribution ($15-18$ samples per subtype).

---

## 5. Downsampling Preservation Audit ($256 \times 256$)

A critical risk identified in Phase 1 and Phase 2 was whether downsampling high-resolution images ($700-900\text{px}$) to the standard ML input resolution ($256 \times 256$) would cause thin scratches or micro-cracks to vanish.

Using `MaskProcessor` with strict nearest-neighbor interpolation on all 240 physical masks:
- **Defects Disappeared ($A_{resized} = 0$):** **0 / 240** ($0.00\%$)
- **Significant Defect Area Loss ($>50\%$ scaled area lost):** **0 / 240** ($0.00\%$)
- **Original Mean Defect Area Ratio:** $11.045\%$
- **Resized Mean Defect Area Ratio:** $11.044\%$
- **Conclusion:** Downsampling to $256 \times 256$ via nearest-neighbor preserves $100\%$ of defective regions without a single instance of defect erasure across the entire dataset.
