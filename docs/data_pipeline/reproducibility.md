# VisionInspect AI — Pipeline Reproducibility & Determinism

**Document ID:** VIA-PIP-REP-001  
**Version:** 1.0.0  
**Phase:** Phase 3 — Data Preprocessing, Dataset Splitting & Augmentation Pipeline  
**Author:** Shrikrishna Chavan (Individual Contributor)  
**Branch:** `shrikrishna-chavan`  
**Status:** Approved Technical Design  

---

## 1. Reproducibility Principles & Guarantees

In manufacturing quality inspection, **non-deterministic behavior is a liability**. If an inspection pipeline produces different split partitions, training batches, or augmentation states across identical test runs, performance improvements cannot be distinguished from random noise.

VisionInspect AI enforces determinism across four pipeline pillars:

1. **Manifest-Driven Execution:** All data loaders consume versioned, immutable JSON/CSV manifests. No directory-scanning or filesystem ordering race conditions occur during training.
2. **Local Random State Isolation:** All random number generation uses dedicated, isolated instance generators (`random.Random(seed)`). Global mutation of `random.seed()` or `numpy.random.seed()` is avoided to prevent thread/process race conditions.
3. **Cryptographic Content Hashing:** All samples are tracked via SHA-256 hashes computed from raw image bytes.
4. **Isolated Evaluation Paths:** Validation and test data loaders receive zero random transformations, ensuring bit-exact metric reproducibility across multiple evaluation runs.

---

## 2. Seed Policy & Configuration Architecture

### 2.1 Default System Seeds
- **Global Pipeline Seed:** `42`
- **Split Configuration Seed:** `SplitConfig(seed=42)`
- **Augmentation Generator Seed:** `AugmentationConfig(seed=42)`
- **PyTorch DataLoader Generator:** Seeded via `torch.Generator().manual_seed(42)`

### 2.2 Replay Verification
Given identical inventory manifest inputs and identical seed integers, `DatasetSplitter` produces bit-exact identical lists of `train`, `val`, and `test` records across all platforms and operating systems. This was empirically verified by `TestPhase3DataPipeline.test_11_deterministic_dataset_splitting`.

---

## 3. Boundaries of Bitwise Determinism & Hardware Variance

While algorithmic and logical determinism is guaranteed, computer vision pipelines encounter known boundaries of bitwise floating-point reproducibility across diverse hardware environments:

### 3.1 Bilinear Interpolation & SIMD Differences
- OpenCV and Pillow implement hardware-accelerated vector instructions (AVX-512, NEON, SSE2) for bilinear image resampling.
- Minor floating-point rounding variations ($\approx \pm 10^{-6}$) can occur when resizing an image on x86_64 Intel CPUs versus ARM64 Apple Silicon versus NVIDIA CUDA tensor cores.
- *Mitigation:* VisionInspect AI uses Pillow's high-precision `Resampling.BILINEAR` with standardized filter kernels across CPU workers.

### 3.2 CUDA / CuDNN Non-Determinism
- Certain CUDA convolutional backpropagation kernels use atomic addition operations across thread blocks, introducing non-deterministic floating-point ordering.
- *Deterministic Flag Configuration:*
  ```python
  import torch
  torch.manual_seed(42)
  if torch.cuda.is_available():
      torch.cuda.manual_seed_all(42)
      torch.backends.cudnn.deterministic = True
      torch.backends.cudnn.benchmark = False
  ```
- *Performance Trade-off:* Enforcing `cudnn.deterministic = True` can reduce training throughput by $10\% - 20\%$. In development and benchmark validation, this trade-off is accepted to ensure audit fidelity.

---

## 4. Pipeline Metadata Audit Trail

Every generated split and preprocessed dataset records the following audit metadata block:

```json
{
  "pipeline_version": "1.0.0",
  "phase": "Phase 3 - Preprocessing & Splitting",
  "split_protocol": "protocol_a_patchcore",
  "seed": 42,
  "target_resolution": [256, 256],
  "normalization": {
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225]
  },
  "created_at": "2026-11-17T20:30:00Z"
}
```
This metadata guarantees that downstream model training runs in Phase 4 can trace their exact data lineage.
