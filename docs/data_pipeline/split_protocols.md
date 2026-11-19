# VisionInspect AI — Dataset Split Protocols & Leakage Prevention

**Document ID:** VIA-PIP-SPL-001  
**Version:** 1.0.0  
**Phase:** Phase 3 — Data Preprocessing, Dataset Splitting & Augmentation Pipeline  
**Author:** Shrikrishna Chavan (Individual Contributor)  
**Branch:** `shrikrishna-chavan`  
**Status:** Approved Technical Design  

---

## 1. Overview & The Benchmark Challenge

Automated defect detection systems operate under two fundamentally different supervision paradigms:
1. **Unsupervised / One-Class Anomaly Detection (PatchCore):** Assumes zero defective training samples are available. Models the manifold of nominal manufacturing appearance using exclusively normal images.
2. **Supervised Defect Segmentation (U-Net):** Learns pixel-level defect boundaries via backpropagation using paired defective images and ground-truth masks.

To evaluate both models rigorously without invalidating benchmark integrity, VisionInspect AI implements two distinct, mathematically segregated dataset splitting protocols.

---

## 2. Protocol A: Official Normal-Only Anomaly Detection (PatchCore)

### 2.1 Protocol Design
Protocol A adheres to the official MVTec AD benchmark methodology (*Bergmann et al., CVPR 2019*):
- **Training Set (PatchCore Memory Bank):** Composed exclusively of normal, defect-free images ($80\%$ of `train/good/`).
- **Validation Calibration Set:** Composed exclusively of normal, defect-free images ($20\%$ of `train/good/`). Used strictly for empirical anomaly threshold calibration ($\tau_{\text{pass}}$ at the 99th percentile of normal scores).
- **Held-Out Test Set:** Preserves the entire official `test/` folder ($100\%$ of `test/good/` plus $100\%$ of all defective subtypes with ground-truth masks).

```
+-------------------------------------------------------------------------------+
|                       PROTOCOL A: PATCHCORE PARTITIONING                      |
+-------------------------------------------------------------------------------+
| Raw train/good/ (100% Normal)               | Raw test/ (Normal + Defects)    |
|   |                                         |   |                             |
|   +--> 80% PatchCore Train Memory Bank      |   +--> 100% HELD-OUT TEST SET   |
|   |    (EXCLUSIVELY NORMAL)                 |        (Normal + All Defect     |
|   |                                         |         Subtypes for Final      |
|   +--> 20% Normal Validation Set            |         Benchmark Evaluation)   |
|        (Threshold Calibration Only)         |                                 |
+-------------------------------------------------------------------------------+
```

### 2.2 Strict Invariance Rules for Protocol A:
1. **Zero Defective Pixels in Training:** Defective samples must **never** enter the PatchCore memory bank. The training loader (`src.dataset.torch_datasets.PatchCoreDataset`) actively asserts `is_anomalous == False` for every training sample and raises a fatal `ValueError` if violated.
2. **Zero Test Data Influence on Calibration:** Anomaly score thresholds ($\tau_{\text{pass}}, \tau_{\text{binary}}$) must be tuned exclusively on the 20% normal validation split. Evaluating or tuning on the held-out test set prior to final reporting is strictly prohibited.

---

## 3. Protocol B: Custom Supervised Segmentation Comparison (U-Net)

### 3.1 Protocol Design
Because supervised segmentation models require defective images with paired ground-truth masks during training, U-Net cannot be trained under Protocol A without seeing defects. To benchmark U-Net fairly, Protocol B defines a **custom stratified split**:
- **Train Split (60%):** Stratified blend of normal images and defective images across all defect subtypes.
- **Validation Split (20%):** Stratified blend of normal and defective images used for early stopping and mIoU checkpoint selection.
- **Held-Out Test Split (20%):** Disjoint held-out evaluation partition containing normal and defective samples.

```
+-------------------------------------------------------------------------------+
|                 PROTOCOL B: SUPERVISED U-NET CUSTOM PARTITION                 |
+-------------------------------------------------------------------------------+
| Total Inventory (Normal + All Defect Subtypes)                               |
|   |                                                                           |
|   +--> 60% Supervised Training Split (Normal + Defective with Masks)         |
|   |                                                                           |
|   +--> 20% Supervised Validation Split (Early Stopping & Checkpoint Choice)   |
|   |                                                                           |
|   +--> 20% Held-Out Test Set (Identical Disjoint Evaluation Partition)        |
+-------------------------------------------------------------------------------+
```

> [!IMPORTANT]
> **Scientific Disclosure:**  
> Protocol B represents a **custom experimental comparison protocol**, NOT the official MVTec AD benchmark. Benchmark publications evaluate unsupervised models under Protocol A. When reporting comparative findings, this distinction in supervision and data access must be explicitly documented.

---

## 4. Rare Defect Subtype Allocation

Industrial anomaly datasets frequently contain rare defect subtypes with very few physical samples (e.g., $N=1$ or $N=2$). Blind stratified splitting algorithms fail or crash when attempting to split 1 sample across 3 folds.

The `DatasetSplitter` incorporates defensive rare-subtype allocation:
1. **Single Sample ($N=1$):** Allocated to the training split with an audit warning (`Rare subtype detected: allocating to train`). This ensures the neural network observes the defect pattern during training rather than encountering an unrepresented anomaly in testing.
2. **Two Samples ($N=2$):** Allocated as 1 sample to training and 1 sample to held-out test, ensuring representation in both learning and verification.
3. **Three or More Samples ($N \ge 3$):** Partitioned according to configurable ratio targets while enforcing non-empty partitions.

---

## 5. Data Leakage Prevention Guarantees

VisionInspect AI enforces a strict mathematical guarantee of disjointness across splits:

$$\mathcal{S}_{\text{train}} \cap \mathcal{S}_{\text{val}} = \emptyset$$
$$\mathcal{S}_{\text{train}} \cap \mathcal{S}_{\text{test}} = \emptyset$$
$$\mathcal{S}_{\text{val}} \cap \mathcal{S}_{\text{test}} = \emptyset$$

### 5.1 Automated Disjointness Auditing
Every execution of `DatasetSplitter.split()` invokes `verify_no_leakage()`. If an identical relative image path appears in more than one partition, the method aborts with a fatal `DataLeakageError`.

### 5.2 Content Hash Verification
To prevent near-duplicate images (e.g., identical photographs saved under different filenames) from spanning across splits, the preliminary inventory records the SHA-256 cryptographic hash of every image buffer. The validator flags any cross-split hash collisions during pre-flight checks.
