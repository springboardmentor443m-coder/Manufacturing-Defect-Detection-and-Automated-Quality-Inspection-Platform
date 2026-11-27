# VisionInspect AI — Architecture

```
            ┌───────────────────────────── React (Vite + Tailwind) ─────────────────────────────┐
            │ Login · Dashboard · Upload · Dataset & Camera · Batches/Image viewer · Users      │
            └───────────────────────────────────────┬────────────────────────────────────────────┘
                                                    │  REST /api  (JWT bearer token)
┌───────────────────────────────────────────────────▼──────────────────────────────────────────────┐
│ FastAPI backend                                                                                  │
│                                                                                                  │
│  1 User Management      auth.py, users.py, core/deps.py (role checks)                 [M1 ✅]    │
│  2 Image Acquisition    batches.py (upload), dataset.py (MVTec import, camera sim)    [M1 ✅]    │
│                         services/image_validation.py, services/ingest.py                         │
│  3 Image Processing     preprocessing, noise removal, enhancement, features           [M2]       │
│  4 Defect Detection     anomaly detection / YOLO localisation                         [M2]       │
│  5 Defect Classification + Severity scoring (size 30 · location 25 · type 25 · conf 20) [M3]     │
│  6 Quality Control      pass/fail decision, inspection reports                        [M3]       │
│  7 Analytics Dashboard  dashboard.py (M1: acquisition stats; M3: defect trends)       [M1→M3]    │
└───────────────┬───────────────────────────────────────────────┬──────────────────────────────────┘
                │ SQLAlchemy                                    │ files
     ┌──────────▼──────────┐                         ┌──────────▼───────────┐     ┌──────────────────┐
     │ PostgreSQL (SQLite  │                         │ storage/images/<id>/ │◄────│ MVTec AD dataset │
     │ in local dev)       │                         │ storage/masks/<id>/  │     │ data/mvtec_...   │
     └─────────────────────┘                         └──────────────────────┘     └──────────────────┘
```

## Request flow: one image entering the system

1. The image arrives in one of three ways: a user upload, a simulated camera frame, or a dataset import.
2. `services/ingest.py` computes the SHA-256 hash. If the same file is already stored, it is skipped.
3. `services/image_validation.py` checks the file type, size, whether it decodes, its resolution and whether it is blank (all **reject** conditions), then sharpness and brightness (**warning** conditions). It also records the quality metrics.
4. The file is saved under `storage/images/<batch_id>/`. An `InspectionImage` row is written with `inspection_status = pending`, or `skipped` if the image was rejected.
5. In Milestone 2, the defect detection worker will pick up the `pending` images.

## Database schema (Milestone 1)

| Table | Key columns |
|---|---|
| `users` | id, email (unique), full_name, hashed_password (bcrypt), role (`admin` / `quality_engineer` / `supervisor`), is_active, created_at |
| `batches` | id, name, product_category, source (`upload` / `camera` / `dataset`), status (`open` / `processing` / `completed`), created_by_id → users |
| `inspection_images` | id, batch_id → batches, original_filename, stored_path, product_category, source, sha256, width, height, channels, file_size, blur_score, brightness, contrast, validation_status, validation_messages (JSON), ground_truth_label, ground_truth_mask_path, inspection_status, uploaded_by_id → users |

Planned for later milestones: `defect_predictions` (bounding box or mask, defect type, confidence, model version), `severity_assessments` (the four sub-scores, total and level) and `quality_decisions` (pass/fail and the action taken). MongoDB is listed in the brief. It is reserved for raw model outputs and inspection logs in Milestone 2.

## Role permissions

| Action | Admin | Quality Engineer | Factory Supervisor |
|---|:-:|:-:|:-:|
| Create users, change roles, disable accounts | ✅ | ❌ | ❌ |
| Upload images, import the dataset, run the camera simulation, delete batches | ✅ | ✅ | ❌ |
| View the dashboard, batches and images | ✅ | ✅ | ✅ |
