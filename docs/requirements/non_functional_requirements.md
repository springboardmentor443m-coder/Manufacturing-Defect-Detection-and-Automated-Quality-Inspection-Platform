# VisionInspect AI — Non-Functional Requirements Document (NFRD)

**Document ID:** VIA-REQ-NFRD-001  
**Version:** 1.0.0  
**Phase:** Phase 1 — Requirements Analysis & System Architecture Design  
**Author:** Shrikrishna Chavan (Individual Contributor)  
**Branch:** `shrikrishna-chavan`  
**Status:** Approved for Design Baseline  

---

## 1. Overview

Non-Functional Requirements (NFRs) specify the runtime attributes, system constraints, quality attributes, and engineering standards that VisionInspect AI must uphold. In industrial inspection environments, accuracy must be accompanied by low latency, zero silent failures, deterministic reproducibility, and rigorous auditability.

---

## 2. Requirements Traceability Matrix Summary

| Category | Identifier Range | Description | Verification Method |
| :--- | :--- | :--- | :--- |
| **Reliability & Availability** | `NFR-REL-001` – `NFR-REL-005` | Fault tolerance, fail-safe disposition, availability | Stress / Chaos Testing |
| **Maintainability & Modularity** | `NFR-MNT-001` – `NFR-MNT-005` | Clean architecture, type safety, modular decoupling | Static Analysis, Code Review |
| **Reproducibility & Determinism** | `NFR-REP-001` – `NFR-REP-005` | Seed management, model versioning, pipeline determinism | Regression Testing |
| **Performance & Efficiency** | `NFR-PRF-001` – `NFR-PRF-005` | End-to-end latency, inference throughput, memory bounds | Benchmark & Profiling |
| **Scalability & Concurrency** | `NFR-SCL-001` – `NFR-SCL-004` | Stateless design, connection pooling, resource isolation | Load Testing |
| **Security & Hardening** | `NFR-SEC-001` – `NFR-SEC-006` | Input sanitization, path traversal, injection defense | Security Scanning / SAST |
| **Observability & Telemetry** | `NFR-OBS-001` – `NFR-OBS-005` | Structured logging, request correlation, metrics | Log Audit, Monitoring Tests |
| **Testability & QA Automation**| `NFR-TST-001` – `NFR-TST-005` | Code coverage, mock harnesses, CI/CD automated gates | CI Pipeline Execution |
| **Error Handling & Resilience** | `NFR-ERR-001` – `NFR-ERR-004` | RFC 7807 error envelopes, graceful degradation | Failure Injection Tests |

---

## 3. Detailed Non-Functional Requirements

### 3.1 Reliability & Availability

#### NFR-REL-001: System Availability
- **Description:** The VisionInspect AI API service shall achieve an operational availability target of 99.5% uptime during standard factory operational windows (excluding planned maintenance windows).
- **Target Metric:** $\ge 99.5\%$ monthly uptime.
- **Verification Method:** Synthetic uptime monitoring and ping probes against `/health/ready` and `/health/live`.

#### NFR-REL-002: Fail-Safe Decision Posture
- **Description:** In the presence of internal exceptions, partial sensor corruption, missing model checkpoints, or out-of-range inference confidence, the system shall never default to a `PASS` recommendation. The system must degrade safely to `REVIEW` (or `REJECT` if configured for high-criticality lines).
- **Target Metric:** 0 occurrences of false PASS under system degradation or uncaught component faults.
- **Verification Method:** Chaos/Fault Injection testing — Simulate GPU out-of-memory, corrupted model weights, and database read failures; verify all outputs are strictly `REVIEW` or terminal HTTP error codes.

#### NFR-REL-003: Stateless API Reliability
- **Description:** The inspection backend shall maintain zero session affinity or in-memory request-coupling. The crash or restart of any individual API worker shall not compromise active sessions or historical data.
- **Verification Method:** Automated worker kill test — Terminate backend worker process while submitting concurrent inspection batches; confirm remaining workers continue processing without data loss.

#### NFR-REL-004: Transactional Persistence Guarantee
- **Description:** All database operations representing an inspection event (raw record, metrics, defect bounding boxes, and artifact pointers) must execute within an ACID-compliant database transaction. If any sub-operation fails, the entire transaction must roll back cleanly.
- **Verification Method:** Automated integration test injecting mid-transaction DB query errors; assert zero dangling rows in `inspections` or `defect_predictions`.

#### NFR-REL-005: MTBF & MTTR Targets
- **Description:** Mean Time Between Failures (MTBF) shall exceed 720 continuous operating hours under nominal load; Mean Time To Recovery (MTTR) for automated container restart shall not exceed 15 seconds.
- **Verification Method:** Extended soak testing under simulated manufacturing load.

---

### 3.2 Maintainability & Code Quality

#### NFR-MNT-001: Strict Layer Separation
- **Description:** The codebase shall enforce strict architectural boundaries: API route controllers, domain business logic, ML inference engines, QC rule evaluators, and database persistence layers must remain physically separate modules. No database queries or ML tensor manipulations may occur within API controller handlers.
- **Verification Method:** Static architecture linting and dependency analysis (`import-linter` or equivalent).

#### NFR-MNT-002: Static Typing & Linting Compliance
- **Description:** 100% of Python backend code shall utilize Python type hints (`typing` / Python 3.10+ annotations) validated by `mypy` in strict mode. All code shall conform to `ruff` / `flake8` and `black` formatting standards.
- **Target Metric:** 0 `mypy` typing errors; 0 linting warnings.
- **Verification Method:** Automated CI gate running `mypy --strict` and `ruff check .`.

#### NFR-MNT-003: Externalized Configuration Management
- **Description:** All environment parameters (database URLs, artifact paths, default thresholds, timeouts, device preferences) must be externalized using environment variables or structured YAML/JSON configurations managed via Pydantic Settings. No hardcoded credentials or operational constants are permitted.
- **Verification Method:** Automated secret scanner (`trufflehog` / `detect-secrets`) and code inspection.

#### NFR-MNT-004: Documentation Completeness
- **Description:** Every public class, module, and API endpoint shall include structured docstrings (Google or NumPy format) specifying parameters, return types, exceptions raised, and algorithmic complexity where applicable.
- **Verification Method:** Automated docstring coverage check (`interrogate` $\ge 90\%$).

#### NFR-MNT-005: Modular Decoupling of ML Frameworks
- **Description:** The core inspection service and QC rule engine shall interact with ML models exclusively via abstract interface contracts (`BaseModelRunner` defining `predict(image: np.ndarray) -> ModelInferenceOutput`). Swapping PatchCore with U-Net or a future ONNX model must require zero modifications to the QC engine or database layer.
- **Verification Method:** Code review and substitution unit testing (swapping model runner implementations).

---

### 3.3 Reproducibility & Determinism

#### NFR-REP-001: Deterministic Seed Management
- **Description:** All random operations across PyTorch, NumPy, Python standard library, and CUDA backends must be globally seeded (`torch.manual_seed(seed)`, `np.random.seed(seed)`, `torch.use_deterministic_algorithms(True)` where supported).
- **Target Metric:** Bit-exact reproducibility of coreset subsampling and inference results given identical inputs and hardware architecture.
- **Verification Method:** Regression test executing PatchCore coreset extraction and inference across 10 identical runs; assert identical floating-point output arrays.

#### NFR-REP-002: Immutable Model Artifact Tracking
- **Description:** Every trained model checkpoint and coreset memory bank file shall be versioned, indexed by SHA-256 cryptographic hash, and registered with complete training provenance (dataset version, hyperparameters, training timestamp).
- **Verification Method:** Registry verification script confirming each checkpoint file matches its recorded SHA-256 hash.

#### NFR-REP-003: Strict Dependency Pinning
- **Description:** All production and development dependencies shall be strictly pinned with exact versions and hash locking via `requirements.txt` or `pyproject.toml` (`poetry.lock` / `pip-tools`).
- **Verification Method:** CI installation test from frozen lockfile inside an isolated clean Docker container.

#### NFR-REP-004: Dataset Manifest Hashing
- **Description:** All training, validation, and test splits shall be frozen into structured JSON/CSV dataset manifests containing file paths, ground truth labels, and SHA-256 hashes for each image.
- **Verification Method:** Manifest verification script checking image existence and checksum consistency.

---

### 3.4 Performance & Latency

#### NFR-PRF-001: Single-Image Inference Latency SLA
- **Description:** On standard GPU hardware (e.g., NVIDIA T4 / RTX 3060 or equivalent):
  - U-Net segmentation inference: p95 latency $\le 100\text{ ms}$.
  - PatchCore anomaly detection inference: p95 latency $\le 250\text{ ms}$.
- On standard CPU hardware (8-core modern x86_64):
  - PatchCore inference: p95 latency $\le 1,200\text{ ms}$.
- **Verification Method:** Latency benchmarking script executing 100 sequential inferences under warmup conditions.

#### NFR-PRF-002: End-to-End API Response Latency SLA
- **Description:** The complete API turnaround time for `POST /inspections` (including network ingress, image validation, preprocessing, model inference, contour extraction, QC rule evaluation, database persistence, and artifact storage) shall satisfy:
  - GPU Mode: $\text{p95} \le 500\text{ ms}$.
  - CPU Mode: $\text{p95} \le 2,000\text{ ms}$.
- **Verification Method:** End-to-end performance benchmarking using load testing harness (`locust` or `k6`).

#### NFR-PRF-003: Peak Memory Footprint
- **Description:** The baseline inference worker process memory consumption shall not exceed 4.0 GB of Host RAM and 4.0 GB of VRAM per active category model instance.
- **Verification Method:** Process memory profiling during continuous 1-hour load test using `tracemalloc` and `torch.cuda.max_memory_allocated()`.

#### NFR-PRF-004: Image Preprocessing Throughput
- **Description:** Image validation, color-space conversion, resizing, and normalization for a 1024x1024 image shall execute in $\le 20\text{ ms}$ on a single CPU core.
- **Verification Method:** Micro-benchmark profiling of OpenCV preprocessing routines.

#### NFR-PRF-005: Analytics Query Latency
- **Description:** Dashboard overview queries (`GET /analytics/overview`) over a database containing $\le 100,000$ historical inspection records shall return within $250\text{ ms}$ (p95).
- **Verification Method:** Query execution analysis (`EXPLAIN ANALYZE`) on pre-seeded PostgreSQL database with 100,000 records.

---

### 3.5 Scalability & Concurrency

#### NFR-SCL-001: Horizontal API Scalability
- **Description:** The API layer shall be horizontally scalable behind a load balancer (Nginx / Traefik / AWS ALB). Adding API worker instances shall increase system throughput linearly up to the database and GPU capacity limits.
- **Verification Method:** Scalability load test comparing throughput with 1, 2, and 4 worker replicas.

#### NFR-SCL-002: Database Connection Pooling
- **Description:** Database connections shall be managed via an optimized asynchronous connection pool (e.g., SQLAlchemy / AsyncPG pool with 10–20 connections per worker). Connections shall be reused efficiently and idle connections automatically recycled.
- **Verification Method:** Concurrency test with 50 simultaneous inspection requests; assert zero connection exhaustion errors.

#### NFR-SCL-003: Asynchronous Storage Offloading
- **Description:** File artifact writes (saving raw images and overlay masks to disk) must not block the main API event loop. Writes must utilize asynchronous I/O (`aiofiles`) or offloaded thread pools.
- **Verification Method:** Event loop lag measurement during concurrent heavy file writes.

#### NFR-SCL-004: Asynchronous Task Queue Readiness
- **Description:** The system architecture must cleanly support transitioning from synchronous inspection execution to an asynchronous worker queue (e.g., Celery / Redis / RabbitMQ) without altering core domain logic.
- **Verification Method:** Architectural review of decoupled domain service interfaces.

---

### 3.6 Security & Infrastructure Hardening

#### NFR-SEC-001: Input Sanitization & Path Traversal Prevention
- **Description:** All user-supplied filenames, category identifiers, and query strings shall be strictly sanitized. Generated storage paths must utilize system-generated UUIDv4 tokens. No user-supplied strings may be concatenated directly into filesystem paths or SQL queries.
- **Verification Method:** Automated security testing submitting path traversal payloads (e.g., `../../etc/passwd`, `..\..\Windows\System32`); assert rejection.

#### NFR-SEC-002: Safe Decompression & Anti-Zip/Pixel Bomb Defense
- **Description:** Image decoders must enforce maximum decompression size limits (preventing decompression bombs where a tiny compressed file expands into gigabytes in RAM).
- **Verification Method:** Ingestion of synthetic decompression bomb; verify immediate rejection prior to memory allocation.

#### NFR-SEC-003: Injection Defense (SQL & Command)
- **Description:** All relational database queries must utilize parameterized statements via an Object-Relational Mapper (SQLAlchemy) or parameterized SQL. Shell execution (`subprocess.call` or `os.system`) is strictly prohibited in runtime paths.
- **Verification Method:** Static security analysis (`bandit -r .`) and automated SQL injection fuzzing.

#### NFR-SEC-004: Least-Privilege Execution
- **Description:** When deployed in containerized environments (Docker), all services shall execute as a dedicated non-root user (`appuser` with UID 10001).
- **Verification Method:** Container security audit (`docker exec id`); verify non-root UID/GID.

#### NFR-SEC-005: Cross-Origin Resource Sharing (CORS) Configuration
- **Description:** The API shall restrict CORS headers to authorized dashboard origins, preventing unauthorized cross-origin browser execution in production.
- **Verification Method:** Automated HTTP pre-flight `OPTIONS` testing with authorized and unauthorized `Origin` headers.

#### NFR-SEC-006: Cryptographic Checksum Integrity
- **Description:** Every uploaded image and stored model artifact shall have its SHA-256 hash calculated and stored. Artifact integrity shall be verifiable at any point in the lifecycle.
- **Verification Method:** Automated unit test comparing calculated image hashes against stored records.

---

### 3.7 Observability, Logging & Telemetry

#### NFR-OBS-001: Structured JSON Logging
- **Description:** All application logs shall be emitted to `stdout` in structured JSON format, including timestamp (ISO-8601 UTC), log level (`INFO`, `WARNING`, `ERROR`), request ID, category, worker ID, and execution context.
- **Verification Method:** Log parser verification script validating JSON schema of emitted log lines.

#### NFR-OBS-002: Distributed Request Tracing & Correlation IDs
- **Description:** Every incoming inspection request shall be assigned a unique `X-Request-ID` (UUIDv4) upon ingress. This ID must propagate through all internal function calls, database logs, and error responses.
- **Verification Method:** End-to-end request test; assert request ID appears in response headers and matching database log entries.

#### NFR-OBS-003: Standardized Health & Readiness Probes
- **Description:** The system shall expose standard Kubernetes-ready endpoints:
  - `/health/live`: Returns 200 OK if API process is responsive.
  - `/health/ready`: Returns 200 OK only if database connection is alive and registered model artifacts are loaded in memory.
- **Verification Method:** Automated health probe verification under normal, DB-disconnected, and model-unloaded states.

#### NFR-OBS-004: Granular Latency & Metric Telemetry
- **Description:** The system shall expose operational metrics (request count, inspection duration histograms, error counts, GPU memory consumption) in Prometheus-compatible format at `/metrics`.
- **Verification Method:** Scrape `/metrics` endpoint; verify metric names conform to OpenMetrics standard.

---

### 3.8 Testability & Verification Standards

#### NFR-TST-001: Test Coverage Baseline
- **Description:** The backend test suite shall achieve a minimum of 80% line coverage and 75% branch coverage across all domain, utility, and API routing modules (excluding raw PyTorch weight matrices).
- **Verification Method:** Automated CI step running `pytest --cov=src --cov-report=term-missing --cov-fail-under=80`.

#### NFR-TST-002: Deterministic Mock Test Harness
- **Description:** The test suite must be capable of executing fully without access to physical GPU hardware or live model weights by utilizing deterministic mock model runners that return synthetic probability tensors.
- **Verification Method:** Execute complete unit and integration test suites on CPU-only runner with environment variable `USE_MOCK_MODELS=true`.

#### NFR-TST-003: CI Test Execution Speed
- **Description:** The complete unit test suite shall execute in under 60 seconds on a standard GitHub Actions runner to ensure rapid developer feedback loops.
- **Verification Method:** Timing measurement of CI test execution step.

---

### 3.9 Error Handling & Resilience

#### NFR-ERR-001: RFC 7807 Standardized Problem Details
- **Description:** All HTTP error responses (4xx and 5xx) shall strictly conform to the RFC 7807 specification (`application/problem+json`) containing: `type`, `title`, `status`, `detail`, `instance`, and custom extensions (`error_code`, `request_id`, `timestamp`).
- **Verification Method:** API schema validation testing asserting RFC 7807 schema compliance on all error routes.

#### NFR-ERR-002: Zero Unhandled Internal Server Errors
- **Description:** All unexpected exceptions shall be intercepted by a global FastAPI exception handler. Uncaught exceptions must be logged with complete stack traces internally while returning a sanitized HTTP 500 error envelope without exposing internal file paths or credentials to the caller.
- **Verification Method:** Exception injection test; assert sanitized error payload and logged stack trace.

---

## 4. Verification Matrix

| Requirement ID | Verification Type | Target Standard | Automated Gate |
| :--- | :--- | :--- | :--- |
| `NFR-REL-001` | Availability Probe | $\ge 99.5\%$ uptime | Synthetic ping |
| `NFR-REL-002` | Chaos Injection | 0 False PASS on failure | `test_fail_safe_chaos.py` |
| `NFR-REL-004` | DB Rollback Test | 0 dangling records | `test_transaction_rollback.py` |
| `NFR-MNT-002` | Static Analysis | 0 typing/linting errors | `mypy --strict`, `ruff` |
| `NFR-MNT-003` | Secret Scanning | 0 hardcoded secrets | `trufflehog` |
| `NFR-MNT-005` | Interface Contract | Modular model swappability | `test_model_abstraction.py` |
| `NFR-REP-001` | Regression Determinism| Bit-exact tensor output | `test_deterministic_reproducibility.py` |
| `NFR-REP-002` | Hash Verification | 100% SHA-256 match | `verify_artifacts_sha256.py` |
| `NFR-PRF-001` | Latency Benchmarking | U-Net $\le 100\text{ms}$, PatchCore $\le 250\text{ms}$ | `benchmark_inference.py` |
| `NFR-PRF-002` | E2E Latency SLA | GPU p95 $\le 500\text{ms}$ | `locust` load test |
| `NFR-SCL-002` | Connection Pool Test | 0 connection exhaustion at 50 req | `test_db_concurrency.py` |
| `NFR-SEC-001` | Path Traversal Test | 100% rejection of `../` payloads | `test_security_traversal.py` |
| `NFR-SEC-004` | Docker Security Audit| Non-root execution (UID 10001) | Dockerfile inspect |
| `NFR-OBS-001` | JSON Log Schema | 100% valid JSON log lines | Log validator |
| `NFR-OBS-002` | Trace ID Propagation | Request ID present in DB & header | `test_traceability.py` |
| `NFR-TST-001` | Coverage Gate | $\ge 80\%$ line coverage | `pytest-cov` |
| `NFR-ERR-001` | Schema Validation | RFC 7807 compliant error format | `test_error_envelopes.py` |
