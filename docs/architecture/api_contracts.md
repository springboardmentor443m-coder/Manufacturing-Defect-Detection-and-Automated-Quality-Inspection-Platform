# VisionInspect AI — API Contract Specification

**Document ID:** VIA-ARCH-API-001  
**Version:** 1.0.0  
**Phase:** Phase 1 — Requirements Analysis & System Architecture Design  
**Author:** Shrikrishna Chavan (Individual Contributor)  
**Branch:** `shrikrishna-chavan`  
**Status:** Approved for Design Baseline  

---

## 1. Overview & Protocol Standards

The VisionInspect AI REST API complies with modern web standards and architectural best practices:
- **Base Protocol:** HTTP/1.1 and HTTP/2 over TLS.
- **Data Serialization:** Standard JSON (`application/json`) for structured requests and responses; `multipart/form-data` for image ingestion; `image/png` for binary visual overlay streams.
- **Error Format:** Strict adherence to **RFC 7807 Problem Details** (`application/problem+json`).
- **Idempotency & Tracing:** All responses return a correlation header: `X-Request-ID: <uuidv4>`.

---

## 2. Standardized Error Response Contract (RFC 7807)

All non-2xx HTTP responses follow the standardized envelope structure:

```json
{
  "type": "https://visioninspect.ai/errors/ERR_INVALID_IMAGE_FORMAT",
  "title": "Unsupported Media Type",
  "status": 415,
  "detail": "Supplied file has MIME type 'image/bmp'. Only 'image/jpeg' and 'image/png' are supported.",
  "instance": "/inspections",
  "error_code": "ERR_INVALID_IMAGE_FORMAT",
  "request_id": "c56d78a9-4b21-4f12-8e79-3d120aef8912",
  "timestamp": "2026-11-17T20:00:00Z"
}
```

### Common Error Codes Catalog:
| Error Code | HTTP Status | Description |
| :--- | :--- | :--- |
| `ERR_PAYLOAD_TOO_LARGE` | 413 | Uploaded image binary exceeds 15 MB limit |
| `ERR_INVALID_IMAGE_FORMAT` | 415 | File format is not JPEG or PNG |
| `ERR_CORRUPT_IMAGE` | 422 | Image binary could not be decoded or has corrupted headers |
| `ERR_INVALID_DIMENSIONS` | 422 | Image dimensions fall outside allowable 256x256 – 4096x4096 range |
| `ERR_UNSUPPORTED_CATEGORY` | 422 | Selected category is not registered or supported |
| `ERR_INSPECTION_NOT_FOUND` | 404 | Specified inspection ID does not exist |
| `ERR_MODEL_NOT_FOUND` | 404 | Requested model version or category model checkpoint is missing |
| `ERR_INFERENCE_TIMEOUT` | 504 | ML inference execution exceeded 5000 ms SLA deadline |
| `ERR_DATABASE_ERROR` | 500 | Database transaction could not be completed |
| `ERR_INTERNAL_SERVER_ERROR`| 500 | Unhandled internal exception occurred |

---

## 3. Detailed Endpoint Contracts

### 3.1 Endpoint 1: Submit Inspection
- **Path:** `/inspections`
- **Method:** `POST`
- **Purpose:** Ingests a manufactured part image, validates pre-flight integrity, executes model inference, performs morphological defect quantification, adjudicates QC rules, persists results, and returns complete inspection disposition.
- **Content-Type:** `multipart/form-data`

#### Request Parameters:
| Field | Type | In | Required | Validation Rules | Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `image` | Binary File | Form | Yes | JPEG/PNG, $\le 15\text{MB}$, $\ge 256\times 256$ | High-resolution inspection image |
| `category` | String | Form | Yes | Must match active slug (`bottle`, `metal_nut`, `tile`) | Manufacturing category |
| `model_name` | String | Form | No | One of: `patchcore`, `unet` (default: category default) | Optional model architecture override |
| `model_version` | String | Form | No | Valid SemVer string | Optional model version override |
| `client_batch_id`| String | Form | No | Max 100 characters | Optional client tracking ID |

#### Response: `201 Created`
- **Content-Type:** `application/json`
```json
{
  "inspection_id": "a81bc81b-dead-4e5d-abff-90865d1e13b1",
  "category": "bottle",
  "status": "COMPLETED",
  "created_at": "2026-11-17T20:00:00.123Z",
  "completed_at": "2026-11-17T20:00:00.412Z",
  "model": {
    "model_name": "patchcore",
    "version": "1.0.0",
    "architecture": "wideresnet50"
  },
  "result": {
    "disposition": "REWORK",
    "anomaly_score": 0.742,
    "confidence_score": 0.96,
    "total_defect_area_px": 1420,
    "defect_area_ratio_pct": 1.25,
    "defect_count": 2,
    "explanation": [
      "Anomaly score 0.742 exceeds review threshold 0.600",
      "Defect area 1.25% is within reworkable threshold 2.50%"
    ]
  },
  "defects": [
    {
      "component_index": 0,
      "bounding_box": {
        "xmin": 112,
        "ymin": 84,
        "xmax": 148,
        "ymax": 126
      },
      "centroid": {
        "x": 130.4,
        "y": 105.1
      },
      "pixel_area": 980,
      "area_ratio_pct": 0.86,
      "severity": "MODERATE"
    },
    {
      "component_index": 1,
      "bounding_box": {
        "xmin": 210,
        "ymin": 190,
        "xmax": 232,
        "ymax": 215
      },
      "centroid": {
        "x": 221.0,
        "y": 202.5
      },
      "pixel_area": 440,
      "area_ratio_pct": 0.39,
      "severity": "MINOR"
    }
  ],
  "artifacts": {
    "raw_image_url": "/artifacts/raw/bottle/a81bc81b-dead-4e5d-abff-90865d1e13b1.jpg",
    "overlay_image_url": "/inspections/a81bc81b-dead-4e5d-abff-90865d1e13b1/overlay",
    "mask_image_url": "/artifacts/masks/bottle/a81bc81b-dead-4e5d-abff-90865d1e13b1_mask.png"
  },
  "timing": {
    "validation_ms": 12.4,
    "preprocessing_ms": 18.1,
    "inference_ms": 165.2,
    "postprocess_ms": 48.6,
    "persistence_ms": 25.0,
    "total_latency_ms": 289.0
  }
}
```

---

### 3.2 Endpoint 2: Get Inspection Details
- **Path:** `/inspections/{id}`
- **Method:** `GET`
- **Purpose:** Retrieves the full inspection audit record, QC disposition, defect bounding boxes, and artifact references by unique inspection UUID.

#### Path Parameters:
- `id` (UUID, required): The unique inspection identifier.

#### Response: `200 OK`
- Same schema as `POST /inspections` response body.

#### Error Responses:
- `404 Not Found`: If inspection UUID does not exist (`ERR_INSPECTION_NOT_FOUND`).
- `422 Unprocessable Content`: If provided `id` is not a valid UUIDv4.

---

### 3.3 Endpoint 3: Get Inspection Visual Overlay
- **Path:** `/inspections/{id}/overlay`
- **Method:** `GET`
- **Purpose:** Serves the composited image artifact (original RGB image blended with colorized anomaly heatmap and highlighted bounding box annotations).

#### Path Parameters:
- `id` (UUID, required): Unique inspection UUID.

#### Response: `200 OK`
- **Content-Type:** `image/png`
- **Headers:**
  - `Content-Disposition: inline; filename="inspection_{id}_overlay.png"`
  - `Cache-Control: public, max-age=86400, immutable`
- **Body:** Binary PNG stream.

#### Error Responses:
- `404 Not Found`: If inspection record or overlay file does not exist.

---

### 3.4 Endpoint 4: Get Analytics Overview
- **Path:** `/analytics/overview`
- **Method:** `GET`
- **Purpose:** Delivers high-level aggregate production metrics, yield figures, disposition distributions, defect severity counts, and latency percentiles.

#### Query Parameters:
| Parameter | Type | Required | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `category` | String | No | null | Optional filter by category slug (`bottle`, `metal_nut`, `tile`) |
| `start_date` | String | No | 7 days ago | ISO-8601 UTC start timestamp |
| `end_date` | String | No | now | ISO-8601 UTC end timestamp |

#### Response: `200 OK`
- **Content-Type:** `application/json`
```json
{
  "time_window": {
    "start_date": "2026-11-10T00:00:00Z",
    "end_date": "2026-11-17T20:00:00Z"
  },
  "summary": {
    "total_inspections": 12500,
    "pass_count": 11375,
    "review_count": 375,
    "rework_count": 500,
    "reject_count": 250,
    "yield_rate_pct": 91.0,
    "rework_rate_pct": 4.0,
    "reject_rate_pct": 2.0,
    "review_rate_pct": 3.0
  },
  "category_breakdown": [
    {
      "category": "bottle",
      "total_inspections": 4500,
      "yield_rate_pct": 92.4,
      "defect_rate_pct": 4.6
    },
    {
      "category": "metal_nut",
      "total_inspections": 4000,
      "yield_rate_pct": 89.5,
      "defect_rate_pct": 7.2
    },
    {
      "category": "tile",
      "total_inspections": 4000,
      "yield_rate_pct": 91.0,
      "defect_rate_pct": 6.1
    }
  ],
  "defect_metrics": {
    "mean_defect_area_pct": 1.45,
    "median_defect_area_pct": 0.85,
    "p95_defect_area_pct": 4.10,
    "total_defect_clusters_found": 1840
  },
  "performance_metrics": {
    "p50_latency_ms": 245.0,
    "p95_latency_ms": 380.0,
    "p99_latency_ms": 495.0
  }
}
```

---

### 3.5 Endpoint 5: Get Registered Models Catalog
- **Path:** `/models`
- **Method:** `GET`
- **Purpose:** Enumerates all registered model architectures, versions, target categories, active production status, and benchmark evaluation metrics.

#### Query Parameters:
| Parameter | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `category` | String | No | Filter models for a specific category slug |
| `active_only` | Boolean| No (default: false) | If true, returns only currently active production models |

#### Response: `200 OK`
- **Content-Type:** `application/json`
```json
{
  "models": [
    {
      "model_version_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
      "category": "bottle",
      "model_name": "patchcore",
      "version": "1.0.0",
      "architecture": "wideresnet50",
      "is_active": true,
      "benchmark_metrics": {
        "image_auroc": 0.992,
        "pixel_auroc": 0.984,
        "pro_score": 0.941
      },
      "default_configuration": {
        "input_resolution": [256, 256],
        "tau_binary": 0.55,
        "tau_pass": 0.45,
        "tau_review": 0.60,
        "tau_reject": 0.80,
        "max_rework_area_pct": 2.5
      },
      "created_at": "2026-11-15T12:00:00Z"
    },
    {
      "model_version_id": "7ca14e22-8921-4d33-a1bb-1c882a99bfd1",
      "category": "bottle",
      "model_name": "unet",
      "version": "1.0.0",
      "architecture": "standard_unet",
      "is_active": false,
      "benchmark_metrics": {
        "pixel_iou": 0.765,
        "dice_score": 0.867
      },
      "default_configuration": {
        "input_resolution": [256, 256],
        "tau_binary": 0.50,
        "tau_pass": 0.30,
        "tau_review": 0.50,
        "tau_reject": 0.75,
        "max_rework_area_pct": 2.0
      },
      "created_at": "2026-11-16T15:30:00Z"
    }
  ]
}
```
