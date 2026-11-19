# VisionInspect AI — Functional Requirements Document (FRD)

**Document ID:** VIA-REQ-FRD-001  
**Version:** 1.0.0  
**Phase:** Phase 1 — Requirements Analysis & System Architecture Design  
**Author:** Shrikrishna Chavan (Individual Contributor)  
**Branch:** `shrikrishna-chavan`  
**Status:** Approved for Design Baseline  

---

## 1. Overview & Objectives

VisionInspect AI is an automated visual quality inspection platform tailored for high-precision manufacturing environments. The system ingests high-resolution images of manufactured parts, detects anomalies and surface defects, localizes anomalies via pixel-level heatmaps and segmentation masks, quantifies defect geometric properties, applies configurable Quality Control (QC) decision rules, and generates actionable disposition recommendations (**PASS**, **REVIEW**, **REWORK**, **REJECT**).

This document formalizes the functional capabilities required for Phase 1 and subsequent delivery phases. Every functional requirement is assigned a permanent identifier (`FR-XXX-###`), detailed acceptance criteria, and a concrete verification methodology.

---

## 2. Requirements Traceability Matrix Summary

| Category | Identifier Range | Description | Verification Types |
| :--- | :--- | :--- | :--- |
| **Image Ingestion & Pre-flight** | `FR-ING-001` – `FR-ING-006` | Ingestion, format, resolution, channel, corruption checks | Unit Test, Integration Test |
| **Category & Configuration** | `FR-CFG-001` – `FR-CFG-004` | Category mapping, rule loading, dynamic parameter binding | Unit Test, Integration Test |
| **Model Inference & Execution** | `FR-INF-001` – `FR-INF-006` | Model dispatch, anomaly detection, semantic segmentation | Integration Test, Benchmark Test |
| **Defect Localization & Geometry**| `FR-LOC-001` – `FR-LOC-005` | Heatmap generation, contour extraction, bounding box, area calculation | Unit Test, Algorithmic Test |
| **QC Decision Engine** | `FR-DEC-001` – `FR-DEC-006` | Disposition logic, threshold adjudication, safety routing | Unit Test, Property-based Test |
| **Persistence & Artifact Store** | `FR-DAT-001` – `FR-DAT-005` | Result persistence, artifact storage, ACID integrity | Integration Test, E2E Test |
| **History & Query Services** | `FR-HIS-001` – `FR-HIS-004` | Filtering, pagination, sorting, audit log retrieval | Integration Test, API Test |
| **Analytics & Reporting** | `FR-ANA-001` – `FR-ANA-004` | Yield calculation, defect Pareto, latency metrics | Integration Test, API Test |
| **Model & Lifecycle Management** | `FR-MOD-001` – `FR-MOD-004` | Model registry, metadata tracking, active version gating | Integration Test, Admin Test |

---

## 3. Detailed Functional Requirements

### 3.1 Image Ingestion & Pre-flight Validation

#### FR-ING-001: Image Ingestion Interface
- **Description:** The system shall accept standard 2D digital image files uploaded via an HTTP `multipart/form-data` endpoint (`POST /inspections`).
- **Input Parameters:** Image file binary payload, `category` string identifier, optional `model_name` string, optional `model_version` string.
- **Verification Method:** Integration Test — Submit valid payloads across varying MIME types; verify HTTP 201 Created and receipt of tracking ID.

#### FR-ING-002: Format Validation
- **Description:** The system shall validate that the uploaded file conforms strictly to acceptable image container encodings: `image/jpeg` (`.jpg`, `.jpeg`) and `image/png` (`.png`). Any other media type (e.g., `.bmp`, `.tiff`, `.gif`, `.pdf`) shall be rejected immediately.
- **Verification Method:** Unit Test — Submit supported and unsupported MIME encodings; assert HTTP 415 Unsupported Media Type for invalid types.

#### FR-ING-003: Payload Size Boundary Checking
- **Description:** The system shall enforce a strict upper boundary on payload size: maximum 15 MB per single inspection image. Payloads exceeding this threshold shall be rejected prior to memory buffer expansion.
- **Verification Method:** Integration Test — Send payload of 15.1 MB; assert HTTP 413 Payload Too Large. Send payload of 14.9 MB; assert acceptance.

#### FR-ING-004: Image Decode & Corruption Checking
- **Description:** The system shall parse the binary stream using standard computer vision libraries (OpenCV / Pillow) to verify complete file header integrity and pixel buffer decodability. If the stream contains truncated segments or decompression errors, it shall be rejected.
- **Verification Method:** Unit Test — Ingest truncated and corrupted JPEG/PNG binaries; verify graceful rejection with HTTP 422 Unprocessable Content and explicit error code `ERR_CORRUPT_IMAGE`.

#### FR-ING-005: Spatial Dimension Constraints
- **Description:** The system shall enforce minimum and maximum resolution boundaries: minimum resolution of 256x256 pixels, maximum resolution of 4096x4096 pixels.
- **Verification Method:** Unit Test — Submit images with dimensions 128x128, 256x256, 1024x1024, and 5000x5000; assert boundary compliance checks.

#### FR-ING-006: Channel & Color Space Normalization
- **Description:** The system shall convert all accepted images to a canonical 3-channel RGB color space (8-bit per channel depth). Grayscale images shall be replicated across RGB channels; RGBA images shall have alpha channels stripped with transparent pixels composited against black background.
- **Verification Method:** Unit Test — Supply 1-channel grayscale, 3-channel BGR/RGB, and 4-channel RGBA inputs; assert output tensor dimensions are strictly `(3, H, W)` with range `[0.0, 1.0]`.

---

### 3.2 Product Category & Configuration

#### FR-CFG-001: Category Validation
- **Description:** The system shall validate the supplied `category` parameter against active supported manufacturing categories (initial baseline: `bottle`, `metal_nut`, `tile`).
- **Verification Method:** Integration Test — Submit supported categories and an unsupported category (`gear`); assert HTTP 422 with `ERR_UNSUPPORTED_CATEGORY`.

#### FR-CFG-002: Dynamic Model & Rule Resolution
- **Description:** The system shall dynamically resolve the active ML model artifact and active QC rule profile associated with the selected category from the system configuration repository.
- **Verification Method:** Integration Test — Ingest image under `bottle`; verify registry selects `bottle_patchcore_v1` and `bottle_qc_rules_v1`.

#### FR-CFG-003: Model Dispatch Override (Admin / Debug Mode)
- **Description:** Authorized inspection requests may optionally specify an explicit `model_name` (e.g., `unet` vs `patchcore`) and `model_version`. If omitted, the default production version mapped to the category shall be invoked.
- **Verification Method:** Integration Test — Inspect `tile` with default parameters vs `model_name="unet"`; confirm selected pipeline matches requested architecture.

#### FR-CFG-004: Configuration Fallback & Graceful Abort
- **Description:** If a category exists in the catalog but has no active model weights or QC rule configuration published, the inspection shall fail safely with a logged configuration fault.
- **Verification Method:** Integration Test — Query inspection for an unconfigured active category; verify safe abort returning HTTP 503 with `ERR_MODEL_UNAVAILABLE`.

---

### 3.3 Model Inference Execution

#### FR-INF-001: PatchCore Inference Pipeline
- **Description:** When routing to a PatchCore model, the system shall extract patch-level neighborhood embeddings using a pre-trained feature extractor, compute nearest-neighbor distances against the category-specific memory bank coreset, and output:
  1. An image-level anomaly score $S_{\text{image}} \in [0, \infty)$ (normalized to $[0, 1]$ relative to calibrated thresholds).
  2. A 2D spatial anomaly map $M_{\text{raw}} \in \mathbb{R}^{H \times W}$.
- **Verification Method:** Algorithmic & Integration Test — Execute inference against mock/sample tensor; verify output tensor shapes and score scalar bounds.

#### FR-INF-002: U-Net Segmentation Pipeline
- **Description:** When routing to a supervised U-Net model, the system shall perform forward propagation on the normalized image tensor and output:
  1. A 2D pixel-wise defect probability map $P_{\text{defect}} \in [0, 1]^{H \times W}$.
  2. An aggregate defect confidence score derived from the maximum probability mass.
- **Verification Method:** Algorithmic & Integration Test — Validate tensor dimensions match input spatial resolution $(H, W)$ after post-interpolation.

#### FR-INF-003: Model Device Dispatch & Concurrency
- **Description:** The inference engine shall dynamically select the execution hardware device (NVIDIA CUDA GPU if accessible; fallback to CPU) and execute inference inside an isolated context without gradient computation (`torch.inference_mode()`).
- **Verification Method:** Unit Test — Execute inference with CUDA mocked/enabled and disabled; verify correct tensor device placement and zero gradient graphs.

#### FR-INF-004: Inference Latency SLA Monitoring
- **Description:** The system shall capture and log fine-grained timing breakdowns: pre-processing time, model inference time, post-processing time, and QC adjudication time.
- **Verification Method:** Integration Test — Inspect response metadata; assert presence of `timing.inference_ms`, `timing.total_ms`.

#### FR-INF-005: Inference Timeout Handling
- **Description:** The system shall enforce a maximum inference execution deadline of 5,000 milliseconds. If model execution exceeds this duration, the worker process shall terminate execution and route the request to a fallback error state.
- **Verification Method:** Integration Test — Inject artificial sleep into mock model; assert timeout trigger and HTTP 504 Gateway Timeout.

---

### 3.4 Defect Localization & Morphological Processing

#### FR-LOC-001: Anomaly Map Binarization
- **Description:** The post-processing pipeline shall threshold the continuous anomaly map or segmentation probability map using the calibrated category-specific threshold $\tau_{\text{binary}}$ to produce a binary defect mask $B \in \{0, 1\}^{H \times W}$.
- **Verification Method:** Unit Test — Pass synthetic probability map with values above and below $\tau_{\text{binary}}$; verify strict boolean matrix output.

#### FR-LOC-002: Morphological Noise Filtering
- **Description:** The binary mask shall undergo morphological filtering (Opening: erosion followed by dilation with an elliptical kernel, e.g., $3 \times 3$ or $5 \times 5$) to eliminate spurious single-pixel sensor noise.
- **Verification Method:** Unit Test — Feed binary mask with isolated 1-pixel hot spots; assert removal of isolated noise while preserving contiguous regions.

#### FR-LOC-003: Connected Component & Contour Extraction
- **Description:** The system shall extract distinct connected components (defect clusters) using topological structural analysis (e.g., `cv2.findContours`). For each connected component $i$, the system shall compute:
  1. Pixel area $A_i$.
  2. Axis-aligned bounding box coordinates: $[x_{\min}, y_{\min}, x_{\max}, y_{\max}]$.
  3. Centroid coordinates $(c_x, c_y)$.
- **Verification Method:** Unit Test — Supply synthetic binary mask with known geometric shapes (circles, squares); verify extracted bounding box coordinates and pixel counts match ground truth within 0% error.

#### FR-LOC-004: Geometric Area Quantification
- **Description:** The system shall calculate:
  1. Total defective pixel area: $A_{\text{total}} = \sum A_i$.
  2. Defect area ratio: $\rho = \frac{A_{\text{total}}}{H \times W} \times 100\%$.
  3. Maximum individual component area: $A_{\max} = \max(A_i)$.
- **Verification Method:** Unit Test — Verify mathematical properties against controlled binary masks.

#### FR-LOC-005: Heatmap Visualization & Overlay Generation
- **Description:** The system shall render a human-interpretable visual inspection artifact combining the original RGB image, a Jet/Turbo colormap overlay of the anomaly map, and highlighted bounding boxes for all detected defect components.
- **Verification Method:** Unit Test — Verify overlay generator produces an 8-bit RGB image matching the original input resolution $(H, W, 3)$.

---

### 3.5 Quality Control (QC) Rule Engine

#### FR-DEC-001: Separation of Inference Evidence and Business Logic
- **Description:** The QC Rule Engine shall execute as an independent software component isolated from model inference. The engine takes objective inspection evidence (anomaly score, defect area ratio, maximum component area, defect count) and evaluates externalized rule parameters.
- **Verification Method:** Architectural Review & Unit Test — Invoke QC Rule Engine directly with mock evidence payloads; assert zero coupling to PyTorch/model weights.

#### FR-DEC-002: Configurable Decision Thresholds
- **Description:** QC decision boundaries shall be specified per category in an external configuration repository. Key parameters include:
  - $\tau_{\text{pass}}$: Anomaly score below which product is certified normal.
  - $\tau_{\text{review}}$: Boundary where anomaly score warrants manual operator review.
  - $A_{\text{rework}}$: Maximum defect area percentage eligible for manual or automated rework.
  - $A_{\text{reject}}$: Critical defect area percentage mandating outright rejection.
  - $C_{\max\_defects}$: Maximum permissible distinct defect components before mandatory rejection.
- **Verification Method:** Unit Test — Test matrix of evidence values against parameterized rule sets; verify exact disposition mapping.

#### FR-DEC-003: Four-Tier Disposition Adjudication
- **Description:** The system shall output exactly one of the four standardized dispositions:
  1. **PASS:** Anomaly score $< \tau_{\text{pass}}$ AND $A_{\text{total}} == 0$.
  2. **REVIEW:** Anomaly score falls in uncertain zone $[\tau_{\text{pass}}, \tau_{\text{review}}]$, OR model confidence is marginal, OR defect count exceeds 0 but area $< A_{\text{min\_visible}}$.
  3. **REWORK:** Anomaly score $\ge \tau_{\text{review}}$ AND $A_{\text{total}} \le A_{\text{rework}}$ AND defect count $\le C_{\max\_defects}$.
  4. **REJECT:** Anomaly score $> \tau_{\text{reject}}$ OR $A_{\text{total}} > A_{\text{rework}}$ OR defect count $> C_{\max\_defects}$.
- **Verification Method:** Property-based Unit Test — Exhaustively test boundary values; ensure 100% deterministic coverage of the state space.

#### FR-DEC-004: Decision Explanation Payload
- **Description:** Every inspection decision shall include an explicit, human-readable justification array detailing the triggered rule predicates (e.g., `["Anomaly score 0.82 exceeds reject threshold 0.75", "Total defect area 4.2% exceeds rework threshold 2.5%"]`).
- **Verification Method:** Unit Test — Assert non-empty, semantically consistent explanation strings across all dispositions.

#### FR-DEC-005: Safety Fail-Safe Routing
- **Description:** In the event of model inference uncertainty, feature corruption, or post-processing exceptions, the QC engine shall enforce a fail-safe policy: route disposition to **REVIEW** (or **REJECT** in high-criticality configurations), never to **PASS**.
- **Verification Method:** Unit Test — Inject corrupted tensors and mock null predictions; verify default disposition is strictly `REVIEW`.

---

### 3.6 Data Persistence & Artifact Storage

#### FR-DAT-001: ACID Transactional Persistence
- **Description:** All inspection metadata, execution parameters, timing breakdowns, defect bounding boxes, and final QC decisions shall be persisted to a relational PostgreSQL database within an atomic transaction.
- **Verification Method:** Integration Test — Simulate database disconnect mid-write; assert rollback prevents orphan records.

#### FR-DAT-002: Immutable Inspection Audit Record
- **Description:** Once committed, inspection records (`inspections`, `inspection_results`, `defect_predictions`) shall be immutable. Updates to historic inspection records are prohibited; subsequent manual operator re-evaluations must be logged as child override events.
- **Verification Method:** Database Constraint Test — Attempt `UPDATE` query on committed inspection record; verify database trigger/constraint refusal.

#### FR-DAT-003: File Artifact Separation
- **Description:** High-volume binary assets (raw uploaded image, rendered overlay image, binary mask PNG, raw float32 anomaly map array) shall be stored in dedicated object/file storage (`/artifacts/`), referenced in the database via URI / file paths.
- **Verification Method:** Integration Test — Ingest image; verify files exist at specified paths and database contains matching URI strings.

#### FR-DAT-004: Artifact Retention & Cleanup Policies
- **Description:** The system shall support configurable storage retention rules (e.g., retaining raw images for 90 days, retaining metadata permanently).
- **Verification Method:** Unit Test — Run retention policy evaluator script against expired simulated records.

---

### 3.7 Inspection History & Query Services

#### FR-HIS-001: Paginated Query Interface
- **Description:** The system shall provide an endpoint (`GET /inspections`) supporting cursor-based or offset-limit pagination (default page size: 20, max: 100).
- **Verification Method:** API Integration Test — Populate database with 50 mock inspections; query page 1 and page 2; assert distinct non-overlapping result sets.

#### FR-HIS-002: Multi-criteria Filtering
- **Description:** The inspection query API shall support filtering by:
  - `category` (e.g., `bottle`, `tile`)
  - `disposition` (`PASS`, `REVIEW`, `REWORK`, `REJECT`)
  - `start_date` and `end_date` (ISO-8601 UTC)
  - `model_version`
- **Verification Method:** API Integration Test — Execute targeted queries; verify returned collections match filter predicates exactly.

#### FR-HIS-003: Single Inspection Detail Retrieval
- **Description:** The system shall provide an endpoint (`GET /inspections/{id}`) returning complete inspection metadata, defect geometry arrays, model version info, and artifact URLs.
- **Verification Method:** API Integration Test — Ingest inspection; query by returned UUID; assert full fidelity matching.

#### FR-HIS-004: Visual Overlay Stream Retrieval
- **Description:** The system shall provide an endpoint (`GET /inspections/{id}/overlay`) serving the processed image overlay with appropriate HTTP caching headers (`Cache-Control: public, max-age=86400`).
- **Verification Method:** API Integration Test — Fetch overlay via HTTP; assert binary stream is valid PNG/JPEG with HTTP 200 OK.

---

### 3.8 Analytics & Yield Dashboard Services

#### FR-ANA-001: Global & Category Yield Metrics
- **Description:** The system shall provide an endpoint (`GET /analytics/overview`) computing:
  - Total inspected units.
  - Overall yield rate: $\frac{N_{\text{PASS}}}{N_{\text{total}}} \times 100\%$.
  - First-pass rejection rate: $\frac{N_{\text{REJECT}}}{N_{\text{total}}} \times 100\%$.
  - Rework rate and Review rate percentages.
- **Verification Method:** API Integration Test — Seed database with known distribution (100 total: 80 pass, 10 reject, 5 rework, 5 review); assert exact arithmetic match in API response.

#### FR-ANA-002: Time-Series Defect Trends
- **Description:** The analytics service shall compute aggregated inspection volumes and defect rates bucketed by hour, day, or week.
- **Verification Method:** API Integration Test — Verify time-bucketed aggregation queries return accurate counts across day intervals.

#### FR-ANA-003: Defect Area Distribution
- **Description:** The analytics service shall aggregate mean, median, and 95th percentile defect area percentages across categories.
- **Verification Method:** API Integration Test — Assert statistical distribution calculations against seeded inspection results.

#### FR-ANA-004: System Latency Profiling
- **Description:** The analytics service shall expose p50, p95, and p99 latency figures for image inference and end-to-end processing.
- **Verification Method:** API Integration Test — Verify response includes latency percentiles.

---

### 3.9 Model & Configuration Management

#### FR-MOD-001: Model Catalog Registration
- **Description:** The system shall provide an endpoint (`GET /models`) enumerating all registered models, their target category, architecture (`unet`, `patchcore`), version string, active status, and evaluation benchmark metrics.
- **Verification Method:** API Integration Test — Query `/models`; verify structured JSON array containing registered metadata.

#### FR-MOD-002: Version Traceability
- **Description:** Every inspection record in the database must reference the exact `model_version_id` used for inference, ensuring 100% retrospective auditability.
- **Verification Method:** Integration Test — Trigger inspection; verify foreign key linkage to `model_versions` table.

#### FR-MOD-003: Hot Reloading of QC Rules
- **Description:** The QC Rule Engine shall support reloading rule configuration parameters without requiring process restart or API downtime.
- **Verification Method:** Integration Test — Update QC threshold in configuration; verify subsequent inspection reflects updated threshold immediately.

---

## 4. Verification Traceability Matrix

| Requirement ID | Verification Method | Automated Test Suite | Success Metric |
| :--- | :--- | :--- | :--- |
| `FR-ING-001` | Integration Test | `test_api_ingestion.py` | HTTP 201 Created with valid UUID |
| `FR-ING-002` | Unit Test | `test_preflight_validation.py` | Rejects non-JPEG/PNG with HTTP 415 |
| `FR-ING-003` | Integration Test | `test_preflight_validation.py` | Rejects payloads > 15MB with HTTP 413 |
| `FR-ING-004` | Unit Test | `test_preflight_validation.py` | Rejects corrupted files with HTTP 422 |
| `FR-ING-005` | Unit Test | `test_preflight_validation.py` | Enforces 256x256 to 4096x4096 bounds |
| `FR-ING-006` | Unit Test | `test_preflight_validation.py` | Converts all inputs to canonical RGB (3, H, W) |
| `FR-CFG-001` | Integration Test | `test_category_config.py` | Rejects unlisted categories with HTTP 422 |
| `FR-CFG-002` | Integration Test | `test_category_config.py` | Dynamically maps category to model + rule |
| `FR-CFG-003` | Integration Test | `test_model_dispatch.py` | Respects authorized model override |
| `FR-CFG-004` | Integration Test | `test_model_dispatch.py` | Safely aborts if model unconfigured |
| `FR-INF-001` | Integration Test | `test_patchcore_pipeline.py` | Returns scalar anomaly score & 2D map |
| `FR-INF-002` | Integration Test | `test_unet_pipeline.py` | Returns probability map matching image resolution |
| `FR-INF-003` | Unit Test | `test_device_dispatch.py` | Executes in `inference_mode` on assigned device |
| `FR-INF-004` | Integration Test | `test_inference_telemetry.py` | Response includes latency breakdown |
| `FR-INF-005` | Integration Test | `test_inference_timeout.py` | Enforces 5000ms SLA with HTTP 504 |
| `FR-LOC-001` | Unit Test | `test_postprocessing.py` | Accurate binarization against $\tau_{\text{binary}}$ |
| `FR-LOC-002` | Unit Test | `test_postprocessing.py` | Filters isolated single-pixel noise |
| `FR-LOC-003` | Unit Test | `test_contour_extraction.py` | 100% accurate bounding box on known test patterns |
| `FR-LOC-004` | Unit Test | `test_area_quantification.py` | Accurate total and relative area calculation |
| `FR-LOC-005` | Unit Test | `test_overlay_rendering.py` | Produces valid 3-channel RGB overlay artifact |
| `FR-DEC-001` | Unit Test | `test_qc_rule_engine.py` | Zero coupling between rule engine and model weights |
| `FR-DEC-002` | Unit Test | `test_qc_rule_engine.py` | Evaluates configurable thresholds correctly |
| `FR-DEC-003` | Property Test | `test_qc_rule_engine.py` | Complete deterministic mapping to 4 dispositions |
| `FR-DEC-004` | Unit Test | `test_qc_rule_engine.py` | Every decision includes explicit explanation |
| `FR-DEC-005` | Unit Test | `test_fail_safe_routing.py` | Failures route to REVIEW/REJECT, never PASS |
| `FR-DAT-001` | Integration Test | `test_database_persistence.py`| ACID transaction prevents orphan records |
| `FR-DAT-002` | DB Constraint Test| `test_immutability.py` | Database prohibits direct UPDATE on results |
| `FR-DAT-003` | Integration Test | `test_artifact_storage.py` | Files saved to disk, URIs persisted in DB |
| `FR-DAT-004` | Unit Test | `test_retention_policy.py` | Identifies expired artifacts for pruning |
| `FR-HIS-001` | API Test | `test_history_api.py` | Clean pagination without overlaps |
| `FR-HIS-002` | API Test | `test_history_api.py` | Filters return exact subsets |
| `FR-HIS-003` | API Test | `test_history_api.py` | High-fidelity retrieval of inspection detail |
| `FR-HIS-004` | API Test | `test_overlay_api.py` | Serves binary overlay stream with HTTP 200 |
| `FR-ANA-001` | API Test | `test_analytics_api.py` | Yield and disposition math exact match |
| `FR-ANA-002` | API Test | `test_analytics_api.py` | Aggregates volume across time buckets |
| `FR-ANA-003` | API Test | `test_analytics_api.py` | Area distribution metrics match ground truth |
| `FR-ANA-004` | API Test | `test_analytics_api.py` | Exposes p50, p95, p99 latency stats |
| `FR-MOD-001` | API Test | `test_model_catalog_api.py`| Enumerates active and registered models |
| `FR-MOD-002` | Integration Test | `test_model_traceability.py` | Inspection linked to exact model version ID |
| `FR-MOD-003` | Integration Test | `test_qc_hot_reload.py` | Updates thresholds without restart |
