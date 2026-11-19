# VisionInspect AI — Scope and Limitations Document

**Document ID:** VIA-REQ-SCL-001  
**Version:** 1.0.0  
**Phase:** Phase 1 — Requirements Analysis & System Architecture Design  
**Author:** Shrikrishna Chavan (Individual Contributor)  
**Branch:** `shrikrishna-chavan`  
**Status:** Approved for Design Baseline  

---

## 1. Executive Summary

This document establishes the precise functional boundaries, design limitations, scientific constraints, and operational scope for the **VisionInspect AI** project. In automated industrial inspection, maintaining rigorous distinctions between algorithmic capability, empirical benchmark metrics, and certified factory-floor deployment is paramount to prevent false safety assurances or uncalibrated claims.

---

## 2. Project Scope Categorization

```
+-------------------------------------------------------------------------------+
|                             VISIONINSPECT AI SCOPE                             |
+-------------------------------------------------------------------------------+
| [1. Initial Mandatory Scope]                                                 |
|   * MVTec AD Baseline: bottle, metal_nut, tile                                |
|   * Normal-only PatchCore Anomaly Detection & Localization                    |
|   * Supervised U-Net Binary Segmentation Benchmark                            |
|   * Morphological Processing (Bounding Boxes, Contours, Pixel Area)          |
|   * Independent 4-Tier QC Rule Engine (PASS, REVIEW, REWORK, REJECT)          |
|   * FastAPI Backend + PostgreSQL DB + File Artifact Storage                   |
|   * React Inspection Dashboard & Analytics Overview                           |
|   * Strict Data Leakage Prevention Protocol & Manifest System                 |
+-------------------------------------------------------------------------------+
| [2. Optional Scope]                                                           |
|   * ResNet18 Binary/Multi-class Classification Baseline                       |
|   * Multi-threshold Sensitivity Presets (Strict, Standard, Tolerant)          |
|   * Batch Inspection CLI Utility                                              |
+-------------------------------------------------------------------------------+
| [3. Future Enhancements (Post-Prototype)]                                     |
|   * Hardware Triggered Industrial Camera Ingestion (GigE Vision / GenICam)    |
|   * Industrial Fieldbus & PLC Connectivity (OPC-UA, Modbus TCP, MQTT)         |
|   * Distributed Celery / Redis Asynchronous Task Queue                        |
|   * Edge Model Compilation (TensorRT, OpenVINO, ONNX Runtime)                 |
+-------------------------------------------------------------------------------+
| [4. Non-Claimable Capabilities (Out of Scope / Scientific Constraints)]        |
|   * Physical Metric Measurements (mm / microns) without Camera Calibration    |
|   * Certified Safety-Critical Quality Adjudications (ISO/ASME/FDA)            |
|   * Commercial Deployment under MVTec AD Dataset (CC BY-NC-SA 4.0)            |
+-------------------------------------------------------------------------------+
```

---

## 3. Detailed Scope Breakdown

### 3.1 Initial Mandatory Scope (Phase 1 – Phase 8)

The initial mandatory scope encompasses an end-to-end, production-architected prototype system:

1. **Target Manufacturing Categories:**
   - Three representative industrial categories from the MVTec Anomaly Detection (MVTec AD) benchmark:
     - `bottle` (translucent surface, structural cracks, contamination).
     - `metal_nut` (machined metal texture, thread damage, scratches, bent parts).
     - `tile` (periodic repeating surface pattern, cracks, glue, oil stains).

2. **Core ML Capabilities:**
   - **PatchCore Pipeline:** Normal-only training using pre-trained feature representations (WideResNet50 / ResNet50 backbone), memory bank coreset subsampling, patch-level nearest-neighbor anomaly scoring, and spatial anomaly map generation.
   - **U-Net Pipeline:** Supervised defect segmentation trained on paired defect images and pixel ground truth masks, outputting probability maps and binary masks.
   - **Comparative Benchmark:** Fair, held-out evaluation protocol on strictly disjoint test splits comparing PatchCore (unsupervised/one-class) and U-Net (fully supervised).

3. **Post-Processing & Defect Quantification:**
   - Anomaly map spatial smoothing (Gaussian filter) and binarization via calibrated category thresholds.
   - Morphological filtering (noise reduction) and connected component analysis.
   - Extraction of defect clusters, bounding boxes $[x_{\min}, y_{\min}, x_{\max}, y_{\max}]$, centroids, and pixel area metrics.
   - Rendering of composite inspection overlay artifacts (original image + heatmap + bounding boxes).

4. **Quality Control (QC) Rule Engine:**
   - Decoupled rule evaluation engine executing independently of model inference.
   - Configurable per-category thresholds for anomaly score and defect area.
   - Four-tier disposition logic: `PASS`, `REVIEW`, `REWORK`, `REJECT` with deterministic explanation strings.
   - Fail-safe routing policy: unknown or uncertain states default to `REVIEW`, never `PASS`.

5. **Backend & Persistence Architecture:**
   - RESTful API built on **FastAPI** with structured request validation and RFC 7807 error envelopes.
   - Relational persistence in **PostgreSQL** tracking categories, model versions, configurations, inspection metadata, and defect predictions.
   - Local filesystem/object storage for binary artifacts (raw images, overlays, masks).

6. **Frontend Dashboard:**
   - Single-page application built on **React** featuring:
     - Interactive inspection upload and real-time result inspection.
     - Side-by-side comparison of original image, anomaly heatmap, and bounding box overlay.
     - Historic inspection log with pagination and category/disposition filters.
     - Yield and defect distribution analytics overview.

---

### 3.2 Optional Features

The following capabilities are architected into the system design but prioritized as optional depending on Phase execution timelines and data sufficiency:

1. **ResNet18 Defect Classification:**
   - Implementing a supervised convolutional classification baseline (normal vs anomalous, or multi-class defect classification).
   - *Condition for Implementation:* Will only be enabled if sufficient labeled defect samples exist per defect subtype, avoiding invalid statistical conclusions on severely imbalanced test slices.

2. **Configurable Sensitivity Profiles:**
   - Exposing preset QC rule profiles (`Strict`, `Balanced`, `Tolerant`) allowing operators to toggle between higher recall (fewer missed defects, more false rejects) and higher precision.

3. **Batch Inspection CLI Utility:**
   - A standalone Python CLI tool to batch-submit folders of images for offline inspection and generate CSV summary reports.

---

### 3.3 Future Enhancements (Post-Prototype / Industrial Roadmap)

The following capabilities represent natural extensions for a commercial factory deployment but are intentionally deferred:

1. **Hardware Camera Integration:**
   - Direct hardware triggering via industrial GigE Vision, USB3 Vision, or GenICam protocols.
2. **Factory Automation & Fieldbus Communication:**
   - Integration with Programmable Logic Controllers (PLCs) via OPC-UA, Modbus TCP, or industrial Ethernet to trigger physical reject kickers on conveyor lines.
3. **Asynchronous Distributed Worker Queue:**
   - Offloading image processing and inference tasks from synchronous HTTP threads to Celery / Redis worker pools across distributed GPU clusters.
4. **Edge Accelerator Compilation:**
   - Exporting models to ONNX, TensorRT, or OpenVINO for deployment on industrial edge computers (e.g., NVIDIA Jetson AGX Orin, Intel industrial PCs).
5. **Active Learning & Closed-Loop Retraining:**
   - Capturing operator feedback during `REVIEW` triage to flag false positives and periodically augment the PatchCore coreset or fine-tune U-Net.

---

## 4. Fundamental Limitations & Non-Claimable Capabilities

In adherence to scientific integrity and industrial engineering ethics, the following limitations are explicitly documented and must **never** be claimed by the system in its current prototype form:

### 4.1 Physical Dimensionality Limitations (Pixels vs Millimeters)
- **Constraint:** All defect measurements produced by VisionInspect AI are expressed strictly in **image pixel units** ($px$) and **relative percentage of surface area** ($\%$).
- **Non-Claimable Capability:** The system **cannot claim physical dimensional accuracy** (e.g., "defect length is $1.2\text{ mm}$" or "scratch depth is $50\ \mu\text{m}$").
- **Reason:** The MVTec AD dataset does not contain extrinsic or intrinsic camera calibration parameters, focal length specifications, sensor pixel pitch, lens distortion coefficients, or physical optical targets (e.g., checkerboard calibration grids). Translating pixels to SI units without calibrated telecentric lenses and constant working distances is physically invalid.

### 4.2 Production Safety & Regulatory Certification
- **Constraint:** VisionInspect AI is an engineering research prototype designed to evaluate anomaly detection and computer vision architectures.
- **Non-Claimable Capability:** The system **cannot be claimed to provide certified safety-critical quality decisions** or replace regulated quality assurance standards (e.g., ISO 9001, ISO 13485 for medical devices, IATF 16949 for automotive, or FDA CFR Part 820).
- **Reason:** Regulatory production deployment requires formal Gage Repeatability and Reproducibility (Gage R&R) studies, statistical process control (SPC) validation, fail-safe hardware interlocks, and rigorous third-party auditing.

### 4.3 Semantic Anomaly Score Interpretation
- **Constraint:** Anomaly scores generated by PatchCore represent $k$-nearest-neighbor distances in an empirical feature embedding space; U-Net outputs represent uncalibrated pixel classification logits.
- **Non-Claimable Capability:** An anomaly score of $0.85$ **must not be claimed as an 85% physical probability of part failure** or a direct metric of physical structural severity.
- **Reason:** Distance in deep feature space reflects visual dissimilarity from the normal training distribution, which may be influenced by lighting variations, specular reflections, or non-critical cosmetic textures, rather than functional failure.

### 4.4 Dataset Licensing & Commercial Restriction
- **Constraint:** The MVTec Anomaly Detection dataset (MVTec AD) is licensed under the **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-NC-SA 4.0)**.
- **Non-Claimable Capability:** VisionInspect AI **cannot be deployed commercially or monetized** using models trained on MVTec AD.
- **Requirement:** Any future commercialization or production factory usage requires retraining models exclusively on proprietary, company-owned, or commercially licensed dataset acquisitions with formal legal clearance.

---

## 5. Scope Governance Matrix

| Capability / Claim | Status | Governance / Prerequisite |
| :--- | :--- | :--- |
| MVTec AD 3-Category Anomaly Detection | **In Scope (Mandatory)** | Prototype execution on `shrikrishna-chavan` branch |
| Pixel-level Anomaly Heatmap Localization | **In Scope (Mandatory)** | Validated against MVTec ground truth masks |
| Bounding Box & Pixel Area Estimation | **In Scope (Mandatory)** | Pixel coordinates only; no metric units |
| Configurable 4-Tier QC Rule Engine | **In Scope (Mandatory)** | Decoupled from inference tensor outputs |
| Full-Stack Dashboard & Persistence | **In Scope (Mandatory)** | FastAPI + PostgreSQL + React + Docker |
| Physical Millimeter Defect Sizing | **Strictly Out of Scope** | Requires calibrated optical camera setup |
| ISO / Industrial Safety Compliance | **Strictly Out of Scope** | Requires formal industrial Gage R&R audit |
| Commercial Deployment with MVTec AD | **Strictly Prohibited** | Restricted by CC BY-NC-SA 4.0 license |
