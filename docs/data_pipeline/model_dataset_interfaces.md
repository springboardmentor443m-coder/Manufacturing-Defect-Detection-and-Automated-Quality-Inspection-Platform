# VisionInspect AI — Model Dataset Interfaces Specification

**Document ID:** VIA-PIP-INT-001  
**Version:** 1.0.0  
**Phase:** Phase 3 — Data Preprocessing, Dataset Splitting & Augmentation Pipeline  
**Author:** Shrikrishna Chavan (Individual Contributor)  
**Branch:** `shrikrishna-chavan`  
**Status:** Approved Technical Design  

---

## 1. Overview & Segregation Rationale

PatchCore and U-Net represent fundamentally different machine learning approaches:
- **PatchCore:** A non-parametric nearest-neighbor anomaly detection architecture. Its training stage consists of feature extraction from normal images to construct a coreset memory bank. It requires **no mask tensors** during training and **strictly forbids defective images**.
- **U-Net:** A parametric convolutional encoder-decoder trained via gradient backpropagation on pixel-wise classification loss. It **strictly requires paired ground-truth binary masks** for every sample.

Merging these into a single monolithic dataset abstraction creates high coupling, ambiguous null checks, and the risk of anomalous data leaking into the PatchCore memory bank. VisionInspect AI implements two physically separated PyTorch `Dataset` interfaces under `src.dataset.torch_datasets`.

---

## 2. Interface Specifications

### 2.1 Interface 1: `UNetDataset` (Supervised Segmentation)

#### Contract & Data Payload:
Every call to `__getitem__(index)` returns a dictionary:

```python
{
    "image": torch.Tensor,        # Shape: (3, 256, 256), dtype: torch.float32, normalized
    "mask": torch.Tensor,         # Shape: (1, 256, 256), dtype: torch.float32, strictly {0.0, 1.0}
    "category": str,              # e.g., "bottle", "tile"
    "defect_type": str,           # e.g., "broken_large", "good"
    "is_anomalous": bool,         # True for defects, False for normal
    "original_size": (int, int),  # (width, height) prior to resizing
    "image_path": str,            # Relative POSIX path to source image
    "mask_path": Optional[str],   # Relative POSIX path to mask, or None
}
```

#### Implicit Empty Mask Synthesis:
For normal images (`is_anomalous == False`), the `mask_path` is `None`. The `UNetDataset` invokes `MaskProcessor.create_empty_mask()`, which synthesizes an in-memory all-zero tensor `torch.zeros((1, H, W))` on the fly. This eliminates the need to create, store, or manage artificial zero-mask files on disk.

#### Augmentation Wiring:
When `is_train=True` and an augmentor is provided, the paired `(image, mask)` tensors pass through `SynchronizedAugmentor.transform()`. When `is_train=False` (validation or test mode), augmentation is automatically bypassed.

---

### 2.2 Interface 2: `PatchCoreDataset` (Unsupervised Anomaly Detection)

#### Contract & Data Payload:
Every call to `__getitem__(index)` returns a dictionary:

```python
{
    "image": torch.Tensor,        # Shape: (3, 256, 256), dtype: torch.float32, normalized
    "category": str,              # e.g., "bottle"
    "defect_type": str,           # e.g., "good", "broken_large" (for test evaluation)
    "is_anomalous": bool,         # True for defects, False for normal
    "original_size": (int, int),  # (width, height) prior to resizing
    "image_path": str,            # Relative POSIX path to source image
}
```

#### Strict Normal-Only Enforcement Gate:
In training mode (`is_train=True`), `PatchCoreDataset` executes an assertion check across all supplied manifest records during initialization:
```python
if self.is_train:
    anomalous = [r for r in self.records if r.is_anomalous]
    if anomalous:
        raise ValueError(
            f"Protocol Violation: PatchCore training dataset must contain exclusively normal images! "
            f"Found {len(anomalous)} anomalous samples in training split."
        )
```
If an anomalous record is inadvertently passed into the training loader, the pipeline halts immediately, preventing defective patch features from polluting the nominal coreset memory bank.

---

## 3. High-Throughput PyTorch DataLoaders

The factory function `create_dataloader` provides optimized PyTorch `DataLoader` instances:

```python
def create_dataloader(
    dataset: Dataset,
    batch_size: int = 16,
    shuffle: bool = False,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = False,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
```

### Operational Recommendations:
- **U-Net Training:** `batch_size = 8` or `16`, `shuffle = True`, `drop_last = True` (to maintain stable batch normalization statistics).
- **PatchCore Memory Bank Feature Extraction:** `batch_size = 32`, `shuffle = False`, `drop_last = False` (sequential evaluation over all normal training images).
- **Evaluation / Benchmarking:** `batch_size = 16`, `shuffle = False`.
