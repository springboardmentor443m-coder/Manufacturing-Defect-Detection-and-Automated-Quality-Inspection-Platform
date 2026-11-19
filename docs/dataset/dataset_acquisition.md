# VisionInspect AI — Dataset Acquisition & Licensing Gate Specification

**Document ID:** VIA-DAT-ACQ-001  
**Version:** 1.0.0  
**Phase:** Phase 2 — Dataset Acquisition & Exploratory Data Analysis (EDA)  
**Author:** Shrikrishna Chavan (Individual Contributor)  
**Branch:** `shrikrishna-chavan`  
**Status:** Approved Specification / Acquisition Gate Active  

---

## 1. Executive Summary & Dataset Identification

VisionInspect AI targets visual defect detection and anomaly localization across three representative manufacturing categories (`bottle`, `metal_nut`, `tile`) selected from the authoritative industrial benchmark:

- **Dataset Identifier:** MVTec Anomaly Detection Benchmark (MVTec AD)
- **Release Year:** 2019
- **Originating Institution:** MVTec Software GmbH, Munich, Germany
- **Official Portal:** [https://www.mvtec.com/research-teaching/datasets/mvtec-ad](https://www.mvtec.com/research-teaching/datasets/mvtec-ad)
- **Primary Scientific Publication:**
  > Bergmann, P., Fauser, M., Sattlegger, D., & Steger, C. (2019).  
  > *MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection.*  
  > In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 9592–9600.

---

## 2. Licensing Analysis & Internship Legal Gate

### 2.1 License Terms (CC BY-NC-SA 4.0)
The MVTec AD dataset is distributed under the **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-NC-SA 4.0)**. Key legal constraints:
1. **Attribution (BY):** Proper academic citation of MVTec Software GmbH and the original authors is mandatory.
2. **Non-Commercial (NC):** The dataset cannot be used for commercial advantage or monetary compensation.
3. **Share-Alike (SA):** Derivative datasets must be distributed under identical licensing terms.

### 2.2 Corporate Internship Legal Gate Policy
Because this engineering project is developed within an internship associated with a commercial corporate entity, strict risk mitigation policies apply:

> [!CAUTION]
> **Automated Dataset Download is Gated by Default:**  
> 1. Internship participation does **not** grant automatic exemption from CC BY-NC-SA 4.0 commercial use restrictions.  
> 2. The system **strictly forbids automatic scraping or unattended downloading** of MVTec AD without explicit verification of company/legal approval.  
> 3. No proprietary company data or commercial IP may be co-mingled with CC BY-NC-SA 4.0 assets.  
> 4. Any eventual commercial production deployment must train on proprietary, enterprise-acquired, or commercially licensed image data, not MVTec AD.

### 2.3 Current Legal Status
- **Current Legal Approval Status:** **PENDING EXPLICIT USER/COMPANY CONFIRMATION**.
- **Dataset Local Storage State:** Not downloaded into `data/raw/` in this repository turn.
- **Tooling State:** Non-destructive dataset inspection, validation, manifest building, and synthetic test suites have been fully implemented and verified.

---

## 3. Mandatory Acquisition Checklist

To obtain and activate local dataset files, the project administrator or intern supervisor must complete this compliance checklist:

- [ ] **Step 1:** Confirm with the internship mentor and legal team that VisionInspect AI's use of MVTec AD is purely educational, non-commercial, and restricted to academic prototype benchmarking.
- [ ] **Step 2:** Ensure no commercial client deliverables, proprietary intellectual property, or production services incorporate models trained on MVTec AD.
- [ ] **Step 3:** Confirm authorization by setting the system environment variable:
  ```bash
  # Linux / macOS:
  export VISIONINSPECT_MVTEC_LEGAL_APPROVED="TRUE"

  # Windows PowerShell:
  $env:VISIONINSPECT_MVTEC_LEGAL_APPROVED="TRUE"
  ```
- [ ] **Step 4:** Execute the gate verification check:
  ```bash
  python -m src.dataset.cli check-gate
  ```

---

## 4. Manual Dataset Acquisition & Extraction Guide

Once legal authorization is verified:

### 4.1 Download Category Archives
Download the category archive files directly from the official portal ([https://www.mvtec.com/research-teaching/datasets/mvtec-ad](https://www.mvtec.com/research-teaching/datasets/mvtec-ad)):

| Category | Filename | Compressed Archive Type | Target Path |
| :--- | :--- | :--- | :--- |
| **Bottle** | `bottle.tar.xz` | XZ-compressed TAR archive | `data/raw/bottle.tar.xz` |
| **Metal Nut** | `metal_nut.tar.xz` | XZ-compressed TAR archive | `data/raw/metal_nut.tar.xz` |
| **Tile** | `tile.tar.xz` | XZ-compressed TAR archive | `data/raw/tile.tar.xz` |

### 4.2 Safe Extraction
Extract each archive into `data/raw/mvtec_ad/` preserving original directory structures:

```bash
# Example extraction commands:
mkdir -p data/raw/mvtec_ad

# Extract bottle:
tar -xf data/raw/bottle.tar.xz -C data/raw/mvtec_ad/

# Extract metal_nut:
tar -xf data/raw/metal_nut.tar.xz -C data/raw/mvtec_ad/

# Extract tile:
tar -xf data/raw/tile.tar.xz -C data/raw/mvtec_ad/
```

### 4.3 Post-Extraction Structure Verification
Run the automated VisionInspect AI validator to confirm extraction integrity, image decodability, and mask pairings:

```bash
python -m src.dataset.cli validate --dataset-dir data/raw/mvtec_ad --categories bottle metal_nut tile
```

> [!NOTE]
> All raw dataset images, masks, and archive files are strictly excluded from Git commits via `.gitignore` (`data/raw/`, `*.tar.xz`, `*.png`).
