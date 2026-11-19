# VisionInspect AI — Component Design Specification

**Document ID:** VIA-ARCH-CMP-001  
**Version:** 1.0.0  
**Phase:** Phase 1 — Requirements Analysis & System Architecture Design  
**Author:** Shrikrishna Chavan (Individual Contributor)  
**Branch:** `shrikrishna-chavan`  
**Status:** Approved for Design Baseline  

---

## 1. Overview & Module Organization

This document details the internal design, object-oriented abstractions, component interfaces, and runtime data structures across each subsystem of VisionInspect AI. The implementation adheres strictly to Clean Architecture and SOLID design principles to ensure modularity, high testability, and clear separation of concerns.

---

## 2. Directory & Module Package Layout

The target implementation codebase is structured into cohesive packages under `src/`:

```
src/
├── api/                         # Presentation & REST Interface Layer
│   ├── routes/
│   │   ├── inspections.py       # POST /inspections, GET /inspections, GET /inspections/{id}
│   │   ├── analytics.py         # GET /analytics/overview
│   │   ├── models.py            # GET /models
│   │   └── health.py            # GET /health/live, GET /health/ready
│   ├── middleware/
│   │   ├── correlation.py       # Request ID injection & propagation
│   │   ├── timing.py            # Latency profiling middleware
│   │   └── exception_handler.py # RFC 7807 global exception envelope
│   ├── schemas/                 # Pydantic v2 Request/Response DTOs
│   │   ├── inspection.py
│   │   ├── analytics.py
│   │   ├── model.py
│   │   └── error.py
│   └── app.py                   # FastAPI application factory
│
├── core/                        # System Configurations & Utilities
│   ├── config.py                # Pydantic Settings (ENV variables)
│   ├── constants.py             # System enumerations & codes
│   ├── logging.py               # Structured JSON logger configuration
│   └── exceptions.py            # Domain-specific base exceptions
│
├── preprocessing/               # Pre-flight Validation & Transformation
│   ├── validator.py             # ImageValidator (MIME, size, corruption)
│   ├── preprocessor.py          # ImagePreprocessor (Resize, normalize, tensor)
│   └── transforms.py            # PyTorch torchvision transform pipelines
│
├── inference/                   # ML Model Inference Layer
│   ├── base.py                  # BaseModelRunner abstract base class
│   ├── dispatcher.py            # ModelDispatcher (Device allocation & registry)
│   ├── patchcore/
│   │   ├── runner.py            # PatchCoreRunner implementation
│   │   ├── feature_extractor.py # Intermediate backbone feature extraction
│   │   └── coreset.py           # Coreset memory bank representation
│   └── unet/
│       ├── runner.py            # UNetRunner implementation
│       └── model.py             # PyTorch U-Net architecture definition
│
├── postprocessing/              # Anomaly Localization & Geometry Engine
│   ├── postprocessor.py         # Gaussian smoothing, binarization, contours
│   ├── geometry.py              # Bounding boxes, centroids, area calculations
│   └── visualizer.py            # OverlayRenderer (Heatmap & bounding boxes)
│
├── qc/                          # Quality Control Rule Engine
│   ├── engine.py                # QCRuleEngine (Disposition logic)
│   ├── rules.py                 # Concrete rule evaluators
│   └── config.py                # Category QC threshold schemas
│
├── storage/                     # Persistence & Artifact Store
│   ├── database.py              # SQLAlchemy engine & session factory
│   ├── models/                  # Declarative ORM models
│   │   ├── category.py
│   │   ├── model_version.py
│   │   ├── inspection.py
│   │   └── defect.py
│   ├── repositories/            # Repository pattern implementations
│   │   ├── inspection_repo.py
│   │   ├── analytics_repo.py
│   │   └── model_repo.py
│   └── artifacts/               # Local/Object artifact storage managers
│       ├── base.py              # BaseArtifactStore interface
│       └── local.py             # LocalFileArtifactStore
│
└── telemetry/                   # Metrics & Health Probes
    ├── metrics.py               # Prometheus metrics definitions
    └── probes.py                # Database & model readiness evaluators
```

---

## 3. Core Component Designs

### 3.1 Pre-flight Validation & Preprocessing Component

#### Class: `ImageValidator`
- **Purpose:** Enforces rigorous pre-flight safety checks on incoming byte streams before memory allocation or model loading.
- **Key Methods:**
  - `validate_payload(content_type: str, file_size: int, stream: IO[bytes]) -> ValidationResult`:
    - Validates MIME type against `{"image/jpeg", "image/png"}`.
    - Asserts `file_size <= 15 * 1024 * 1024` bytes.
    - Decodes header bytes to verify format signature.
    - Uses PIL/OpenCV to verify complete buffer decodability without truncation.
    - Checks dimensions: $256 \le \text{width, height} \le 4096$.
- **Failure Behavior:** Raises `InvalidImageFormatError`, `PayloadTooLargeError`, or `CorruptedImageError`.

#### Class: `ImagePreprocessor`
- **Purpose:** Standardizes raw visual data into calibrated PyTorch tensors.
- **Key Methods:**
  - `process(image_bytes: bytes, target_size: Tuple[int, int] = (256, 256)) -> PreprocessedImageBundle`:
    - Decodes image into 3-channel RGB NumPy array ($H \times W \times 3$).
    - Performs anti-aliased bilinear resizing to `target_size`.
    - Converts pixel values to $[0.0, 1.0]$ float32.
    - Applies ImageNet mean/standard deviation normalization.
    - Returns `PreprocessedImageBundle` containing:
      - `tensor`: `torch.FloatTensor` of shape `(1, 3, 256, 256)`.
      - `original_rgb`: Original resolution NumPy array for overlay compositing.
      - `original_dims`: `(original_width, original_height)`.

---

### 3.2 Machine Learning Inference Subsystem

#### Abstract Interface: `BaseModelRunner`
```python
from abc import ABC, abstractmethod
import numpy as np
import torch
from dataclasses import dataclass

@dataclass(frozen=True)
class ModelInferenceOutput:
    raw_anomaly_score: float             # Uncalibrated image-level scalar
    raw_anomaly_map: np.ndarray          # 2D continuous array (H, W) in float32
    inference_duration_ms: float         # Execution time in milliseconds
    device_used: str                     # "cuda:0" or "cpu"

class BaseModelRunner(ABC):
    @abstractmethod
    def load(self, artifact_path: str, device: torch.device) -> None:
        """Loads weights, checkpoints, or coreset memory banks from disk."""
        pass

    @abstractmethod
    def predict(self, input_tensor: torch.Tensor) -> ModelInferenceOutput:
        """Executes forward inference in torch.inference_mode()."""
        pass
```

#### Class: `PatchCoreRunner` (implements `BaseModelRunner`)
- **Internal Mechanics:**
  - Uses a frozen `WideResNet50_2` or `ResNet50` backbone pretrained on ImageNet.
  - Hooks intermediate representations from `layer2` and `layer3`.
  - Performs spatial patch aggregation via `torch.nn.AdaptiveAvgPool2d(1)` or local neighborhood unfolding.
  - Measures minimum $L_2$ Euclidean distance to the category coreset memory bank using an efficient nearest-neighbor search (`torch.cdist` or FAISS).
  - Computes raw anomaly map via spatial interpolation to input resolution, followed by image-level score derivation via re-weighted maximum patch distance.

#### Class: `UNetRunner` (implements `BaseModelRunner`)
- **Internal Mechanics:**
  - Standard U-Net convolutional encoder-decoder architecture with contracting path, bottleneck, and expanding path with skip connections.
  - Final $1\times 1$ convolution followed by Sigmoid activation yielding a pixel-wise defect probability map $P \in [0, 1]^{H \times W}$.
  - Image anomaly score computed as the 99th percentile or top-k mean of pixel defect probabilities.

#### Class: `ModelDispatcher`
- **Purpose:** Manages model lifecycle, active instances, and hardware execution.
- **Responsibilities:**
  - Initializes device context (`torch.cuda.is_available() -> "cuda:0"` else `"cpu"`).
  - Caches loaded category runners in memory as singletons.
  - Enforces timeout deadlines via `asyncio.wait_for`.
  - Exposes `get_runner(category: str, model_name: Optional[str]) -> BaseModelRunner`.

---

### 3.3 Post-Processing & Geometry Engine

#### Class: `AnomalyPostProcessor`
- **Purpose:** Converts continuous probability/anomaly tensors into discrete geometric defect entities.
- **Key Methods:**
  - `process(raw_map: np.ndarray, threshold: float, original_dims: Tuple[int, int]) -> DefectEvidence`:
    - **Step 1 (Smoothing):** Applies Gaussian blur with kernel $\sigma=4$ to remove single-pixel artifacts.
    - **Step 2 (Resizing):** Bilinearly scales anomaly map back to `original_dims` $(W, H)$.
    - **Step 3 (Binarization):** Creates binary mask $B = (M > \tau_{\text{binary}})$.
    - **Step 4 (Morphology):** Applies morphological opening (`cv2.morphologyEx` with $5\times 5$ elliptical structuring element).
    - **Step 5 (Contour Extraction):** Invokes `cv2.findContours(B, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)`.
    - **Step 6 (Geometry Computation):** For each contour:
      - Bounding box: $[x_{\min}, y_{\min}, x_{\max}, y_{\max}]$.
      - Centroid: Moments calculation $(\frac{M_{10}}{M_{00}}, \frac{M_{01}}{M_{00}})$.
      - Pixel area: $A_i = \text{cv2.contourArea}(c)$.
    - **Step 7 (Aggregation):** Computes total defective area $A_{\text{total}}$, area ratio $\rho = \frac{A_{\text{total}}}{W \times H} \times 100\%$, and defect count.

#### Class: `OverlayRenderer`
- **Purpose:** Renders composite visual inspection artifacts.
- **Key Methods:**
  - `render_overlay(original_rgb: np.ndarray, anomaly_map: np.ndarray, bboxes: List[BoundingBox]) -> np.ndarray`:
    - Normalizes anomaly map to $[0, 255]$ uint8.
    - Applies OpenCV `COLORMAP_JET` or `COLORMAP_TURBO`.
    - Performs alpha blending with original image:
      $$I_{\text{composite}} = 0.6 \cdot I_{\text{original}} + 0.4 \cdot I_{\text{colormap}}$$
    - Draws high-contrast bounding boxes (e.g., bright yellow or red rectangles) with labeled coordinates around each detected defect component.
    - Returns high-resolution 3-channel RGB image.

---

### 3.4 Quality Control (QC) Rule Engine

#### Class: `QCRuleEngine`
- **Design Philosophy:** Pure domain logic; zero dependencies on PyTorch, OpenCV, or database drivers.
- **Input Data Contract (`QCInputPayload`):**
  - `anomaly_score: float` (Normalized continuous score)
  - `defect_area_ratio_pct: float` (Total defect percentage of surface)
  - `max_component_area_px: int` (Largest contiguous defect size)
  - `defect_count: int` (Number of detected defect clusters)
  - `confidence_score: float` (Model certainty metric)
- **Configuration Contract (`CategoryQCRules`):**
  - `tau_pass: float`
  - `tau_review: float`
  - `tau_reject: float`
  - `max_rework_area_pct: float`
  - `max_defect_count: int`
  - `min_confidence: float`
- **Adjudication Method:**
  ```python
  def adjudicate(self, evidence: QCInputPayload, rules: CategoryQCRules) -> QCDecision:
      justifications = []
      
      # Fail-safe check 1: Low model confidence
      if evidence.confidence_score < rules.min_confidence:
          justifications.append(f"Model confidence {evidence.confidence_score:.2f} is below minimum {rules.min_confidence:.2f}")
          return QCDecision(disposition=Disposition.REVIEW, justifications=justifications)
          
      # Fail-safe check 2: Ambiguous anomaly score
      if rules.tau_pass <= evidence.anomaly_score < rules.tau_review:
          justifications.append(f"Anomaly score {evidence.anomaly_score:.3f} falls in review band [{rules.tau_pass}, {rules.tau_review})")
          return QCDecision(disposition=Disposition.REVIEW, justifications=justifications)
          
      # Defect check: Critical reject
      if evidence.anomaly_score >= rules.tau_reject or evidence.defect_area_ratio_pct > rules.max_rework_area_pct or evidence.defect_count > rules.max_defect_count:
          if evidence.anomaly_score >= rules.tau_reject:
              justifications.append(f"Anomaly score {evidence.anomaly_score:.3f} exceeds reject threshold {rules.tau_reject:.3f}")
          if evidence.defect_area_ratio_pct > rules.max_rework_area_pct:
              justifications.append(f"Defect area {evidence.defect_area_ratio_pct:.2f}% exceeds rework threshold {rules.max_rework_area_pct:.2f}%")
          if evidence.defect_count > rules.max_defect_count:
              justifications.append(f"Defect count {evidence.defect_count} exceeds maximum allowed {rules.max_defect_count}")
          return QCDecision(disposition=Disposition.REJECT, justifications=justifications)
          
      # Defect check: Reworkable defect
      if evidence.anomaly_score >= rules.tau_review and evidence.defect_area_ratio_pct <= rules.max_rework_area_pct:
          justifications.append(f"Defect detected ({evidence.defect_area_ratio_pct:.2f}% area) within acceptable rework bounds")
          return QCDecision(disposition=Disposition.REWORK, justifications=justifications)
          
      # Pass condition: Clean
      if evidence.anomaly_score < rules.tau_pass and evidence.defect_count == 0:
          justifications.append("Anomaly score within nominal bounds with zero detected defects")
          return QCDecision(disposition=Disposition.PASS, justifications=justifications)
          
      # Default fallback
      justifications.append("Defect characteristics require human supervisor evaluation")
      return QCDecision(disposition=Disposition.REVIEW, justifications=justifications)
  ```

---

### 3.5 Persistence & Artifact Storage Component

#### Interface: `BaseArtifactStore`
- `save_raw_image(image_bytes: bytes, category: str, inspection_id: UUID) -> str`: Writes raw upload to `/artifacts/raw/{category}/{id}.jpg`.
- `save_overlay_image(overlay_array: np.ndarray, category: str, inspection_id: UUID) -> str`: Encodes and writes overlay PNG.
- `save_mask_image(mask_array: np.ndarray, category: str, inspection_id: UUID) -> str`: Encodes and writes 1-bit/8-bit mask PNG.
- `get_file_path(uri: str) -> Path`: Resolves URI to local absolute path.

#### Repository: `InspectionRepository`
- Manages transactional boundaries using SQLAlchemy `AsyncSession`.
- Encapsulates queries for inserting root `inspections`, child `inspection_results`, and array of `defect_predictions`.
- Implements cursor-based paginated fetching with category, date, and disposition filters.

---

### 3.6 Frontend Layer Architecture (React SPA)

#### Component Hierarchy:
```
<App>
  ├── <Header> (Navigation, active category selector, system health indicator)
  ├── <InspectionWorkspace>
  │     ├── <UploadDropzone> (Drag-and-drop, category selector, submit button)
  │     └── <InspectionDetailView>
  │           ├── <QCDispositionBanner> (PASS / REVIEW / REWORK / REJECT status + justification)
  │           ├── <MultiPanelImageViewer> (Original | Anomaly Heatmap | Bounding Box Overlay)
  │           └── <DefectMetricsPanel> (Score, Area %, Defect Count, Bounding Box coordinates)
  ├── <HistoryView>
  │     ├── <FilterToolbar> (Category, disposition, date picker)
  │     └── <InspectionTable> (Paginated table with thumbnail previews)
  └── <AnalyticsDashboard>
        ├── <MetricSummaryCards> (Total inspected, yield %, scrap %, rework %)
        ├── <DefectDistributionChart> (Pareto defect breakdown)
        └── <LatencyTrendChart> (p50/p95 execution timeline)
```
