# Milestone 1 (Weeks 1 & 2): Project Initialization, Design & Core Setup

**Project:** VisionInspect AI, a platform for manufacturing defect detection and quality inspection
**Branch:** `milestone-1`

---

## 1. Requirements from the brief and where each one is done

| # | Milestone 1 requirement | Status | Where |
|---|---|:-:|---|
| 1 | Define quality inspection workflows and project objectives | ✅ | [docs/architecture.md](docs/architecture.md) (request flow), this file |
| 2 | Design the system architecture and database schema | ✅ | [docs/architecture.md](docs/architecture.md), `backend/app/models.py` |
| 3 | Create UI wireframes and plan the workflows | ✅ | Built as working React pages (section 4) |
| 4 | Set up the frontend and backend environments | ✅ | `backend/` (FastAPI), `frontend/` (React + Vite + Tailwind), `docker-compose.yml` |
| 5 | Implement authentication and role-based access | ✅ | `backend/app/core/security.py`, `core/deps.py`, `routers/auth.py`, `routers/users.py` |
| 6 | Load the MVTec AD dataset | ✅ | `backend/app/services/mvtec.py`, `routers/dataset.py`, [docs/dataset.md](docs/dataset.md) |
| 7 | Build the image upload workflow and inspection dashboard | ✅ | `routers/batches.py`, `routers/dashboard.py`, `frontend/src/pages/*` |

### Evaluation criteria for Milestone 1 (Week 2)

| Criterion | Status |
|---|:-:|
| Project initialization and architecture setup completed | ✅ |
| Authentication and image upload workflows implemented | ✅ |
| Inspection dashboard functional | ✅ |
| Dataset integration and preprocessing completed | ✅ The full MVTec AD dataset is loaded (all 5,354 images), and every image is validated and quality-checked. The full preprocessing pipeline (noise removal, enhancement, feature extraction) is planned for Milestone 2, as the brief schedules it. |

---

## 2. Modules from the brief covered so far

### Module 1: User Management ✅
- Three roles: **Admin**, **Quality Engineer** and **Factory Supervisor**. The brief asks for quality engineer and factory supervisor accounts; Admin is added so someone can manage those accounts.
- Login with email and password. It returns a **JWT** that is valid for 8 hours.
- Passwords are hashed with **bcrypt**.
- **Role-based access control** is defined in one place, `core/deps.py`:
  - `can_manage_users`: Admin only
  - `can_inspect`: Admin and Quality Engineer (upload, import, camera, delete)
  - `can_view`: all roles (dashboard, batches, images)
- An admin account is created automatically on first start so the platform can be bootstrapped.
- Admins can create users, change their roles, and disable or re-enable accounts. Disabled users can't log in.
- Admins can't demote or disable their own account, so the platform can't be locked out by mistake.
- Email addresses are matched without regard to letter case, and duplicate emails are rejected.

### Module 2: Image Acquisition ✅
- **Product image upload:** drag and drop several files (up to 100 per batch, 15 MB each) and tag them with a product category and a batch name.
- **Camera integration simulation:** a virtual production-line camera captures N frames of a product. The user sets the share of defective parts, and each frame gets a small lighting change and sensor noise, the way a real camera would.
- **Batch image processing:** every upload, import or capture creates a **batch**. Each batch has a status (`open`, `processing` or `completed`) and counts of valid, warning and rejected images.
- **Image validation** (`services/image_validation.py`):
  - **Reject:** unsupported file type, file too large, corrupt or undecodable file, resolution below 64 px, or a blank image with no content
  - **Warning:** blurry (low sharpness), very dark, or overexposed
  - **Metrics stored for every image:** width, height, channels, file size, sharpness, brightness and contrast
- **Duplicate detection:** each file is identified by its SHA-256 hash, so the same image is never stored twice.
- Rejected images are kept for the audit trail but marked `skipped`, so they won't be sent to the model.

### MVTec AD dataset integration ✅
- Reads the official MVTec AD folder layout: `category/train/good`, `category/test/<defect>` and `category/ground_truth/<defect>/*_mask.png`.
- Summarizes each category: the number of good training images, good test images and defective test images, plus a count for each defect type.
- Imports images into the platform with their **ground-truth label** (`good` or the defect type) and **defect mask**. These are needed to evaluate the model in Milestone 2.
- `scripts/make_sample_dataset.py` generates a small **synthetic dataset in the MVTec layout**: a metal disc part with scratch, crack and hole defects, all with masks. You can use it until the 4.9 GB real dataset is downloaded, and the tests use it too.

### Module 7: Inspection dashboard (Milestone 1 part) ✅
- KPI cards: images captured, number of batches, images waiting for inspection, and images rejected at validation.
- Charts: images captured per day over the last 14 days, a validation breakdown (valid, warning, rejected) and images per product category.
- A ground-truth label breakdown for images that came from the dataset.
- A table of recent batches with links to each batch.

---

## 3. REST API (Swagger docs at `/docs`)

| Method | Endpoint | Role | Purpose |
|---|---|---|---|
| POST | `/api/auth/login` | public | Log in with email and password and get a JWT |
| GET | `/api/auth/me` | any | Get the current user's details |
| GET | `/api/users` | admin | List users |
| POST | `/api/users` | admin | Create a user |
| PATCH | `/api/users/{id}` | admin | Change a user's role, name or password, or enable/disable them |
| POST | `/api/batches/upload` | admin, engineer | Upload images as a new batch (multipart form) |
| GET | `/api/batches` | any | List batches, with optional `category` and `source` filters |
| GET | `/api/batches/{id}` | any | Get a batch with all its images and their metrics |
| PATCH | `/api/batches/{id}/status` | admin, engineer | Change a batch's status |
| DELETE | `/api/batches/{id}` | admin, engineer | Delete a batch and its files |
| GET | `/api/images/{id}/file` | any | Get the stored image |
| GET | `/api/images/{id}/mask` | any | Get the ground-truth defect mask (dataset images only) |
| GET | `/api/dataset` | any | Summarize the MVTec AD categories found on disk |
| POST | `/api/dataset/import` | admin, engineer | Import dataset images as a labelled batch |
| POST | `/api/camera/capture` | admin, engineer | Run the camera simulation |
| GET | `/api/dashboard/stats` | any | Get the dashboard numbers |
| GET | `/api/health` | public | Health check |

---

## 4. Frontend pages (React 18 + Vite + Tailwind CSS + Recharts)

| Page | Visible to | What it does |
|---|---|---|
| **Login** | everyone | Sign in with email and password |
| **Dashboard** | all roles | KPI cards, charts and recent batches |
| **Upload Images** | admin, engineer | Drag-and-drop batch upload with an accepted/rejected/duplicate summary |
| **Dataset & Camera** | admin, engineer | MVTec category table, dataset import form and camera simulation form |
| **Inspection Batches** | all roles | Batch list filtered by source or category |
| **Batch detail** | all roles | Image grid with validation filter chips. Clicking an image shows its metrics, any warnings, and the ground-truth mask next to it |
| **Users** | admin | Create users, change roles inline, and enable or disable accounts |

The navigation and routes hide pages a role can't use. The backend also enforces the same rules on every request, so hiding pages isn't the only protection.

---

## 4b. UI redesign and landing page

The frontend was rebuilt as a dark inspection console in the style of industrial machine-vision software. It uses Inter and JetBrains Mono, a cyan accent colour, and the same pass/warn/fail colours throughout.

| Area | What's included |
|---|---|
| **Landing page** (`/`) | An animated hero: a part goes under a scanning beam, a crack is detected and boxed, and its severity score is worked out with the brief's weights (85 / 90 / 95 / 92 gives **88, Critical**). Below that: the four-step workflow, the 7 modules with their milestone status, the severity levels, the MVTec AD dataset, and the dataset licence credit. |
| **Login** (`/login`) | A split-screen page with the inspection animation, a show/hide password button, and a redirect back to the page you were trying to open. |
| **App shell** | A collapsible sidebar with icons, grouped into Monitor, Acquire and Admin. Also: breadcrumbs, a mobile menu, and a **Ctrl+K command palette** that jumps to any page or batch. |
| **Overview** | Counters that animate up, a sparkline, an image-intake chart, a validation donut, and a category chart you can click to filter batches. Plus defect-label bars, a recent-activity feed and a getting-started screen for an empty database. |
| **Batch view** | A grid with filters by validation result and defect label, plus a dense layout option. Images open in a **full-screen viewer**: ← → to move between images, + and − to zoom, M to toggle the **ground-truth defect-mask overlay** (drawn in red, with an opacity slider), and meters for sharpness, brightness and contrast. |
| **Upload** | A drag-and-drop area with thumbnail previews, a button to remove each file, quick picks for categories, and a summary of accepted, rejected and duplicate files. |
| **MVTec dataset** | A card for each category showing a **real good part next to a real defective part**. Clicking a card opens a panel with that category's defect types, a choice of train or test split, and a slider for how many images to import. |
| **Line camera** | A live-feed view with a scan effect. Captured frames appear one at a time in a filmstrip, and known defective parts are highlighted. |
| **Users & roles** | Counts for each role, add-user and search forms, a role picker on each row, and an on/off switch for each account. |
| **Throughout** | Pop-up notifications, loading placeholders, empty-state screens, images that load only when scrolled into view, and support for reduced-motion settings. |

New backend endpoints that support the UI:
- `GET /api/images/{id}/thumb` returns a 320 px JPEG and caches it on disk, so grids don't download 900 px PNGs.
- `GET /api/dataset/{category}/preview?kind=good|defect` returns a sample image for each category card.

There are now **21 tests**, including one for thumbnails. All pass.

---

## 5. Database schema

The tables are `users`, `batches` and `inspection_images`. [docs/architecture.md](docs/architecture.md) lists every column.
- In local development the app uses **SQLite**, so it runs with no setup. Docker Compose runs **PostgreSQL 16**.
- Tables planned for Milestones 2 and 3: `defect_predictions`, `severity_assessments` and `quality_decisions`.

---

## 6. DevOps and tooling

- `docker-compose.yml`: PostgreSQL, backend and frontend, with persistent volumes for the database and stored images
- `backend/Dockerfile` and `frontend/Dockerfile`
- `backend/.env.example` lists every configuration setting
- `.gitignore` excludes the virtualenv, `node_modules`, the local database, uploaded images and the dataset

## 7. Automated tests (`backend/tests/`)

There are 21 pytest tests:
- **Auth and roles:** wrong password, missing token, admin-only user management, duplicate email, disabled user can't log in, admin can't demote themselves
- **Validation:** valid image, wrong file type, corrupt file, blank image, tiny image, blur warning
- **Upload:** a mixed batch of 2 valid images and 1 corrupt file, duplicate re-upload rejected, supervisor can view but not upload
- **Dataset:** category summary, import keeps labels and masks, re-importing the same images is rejected
- **Camera:** captures the requested number of frames, unknown category returns 404
- **Dashboard:** the stats add up

**Result: 21 / 21 passed** (`cd backend && pytest`).

---

## 7b. Verification on the real dataset ✅

### MVTec AD loaded
The full official dataset is downloaded to `data/mvtec_anomaly_detection/` (5.0 GB, git-ignored). The download passed its integrity check.

| Category | Train (good) | Test (good) | Test (defective) | Defect types |
|---|--:|--:|--:|---|
| bottle | 209 | 20 | 63 | broken_large, broken_small, contamination |
| cable | 224 | 58 | 92 | bent_wire, cable_swap, combined, cut_inner_insulation, cut_outer_insulation, missing_cable, missing_wire, poke_insulation |
| capsule | 219 | 23 | 109 | crack, faulty_imprint, poke, scratch, squeeze |
| carpet | 280 | 28 | 89 | color, cut, hole, metal_contamination, thread |
| grid | 264 | 21 | 57 | bent, broken, glue, metal_contamination, thread |
| hazelnut | 391 | 40 | 70 | crack, cut, hole, print |
| leather | 245 | 32 | 92 | color, cut, fold, glue, poke |
| metal_nut | 220 | 22 | 93 | bent, color, flip, scratch |
| pill | 267 | 26 | 141 | color, combined, contamination, crack, faulty_imprint, pill_type, scratch |
| screw | 320 | 41 | 119 | manipulated_front, scratch_head, scratch_neck, thread_side, thread_top |
| tile | 230 | 33 | 84 | crack, glue_strip, gray_stroke, oil, rough |
| toothbrush | 60 | 12 | 30 | defective |
| transistor | 213 | 60 | 40 | bent_lead, cut_lead, damaged_case, misplaced |
| wood | 247 | 19 | 60 | color, combined, hole, liquid, scratch |
| zipper | 240 | 32 | 119 | broken_teeth, combined, fabric_border, fabric_interior, rough, split_teeth, squeezed_teeth |
| **Total** | **3,629** | **467** | **1,258** | **73 defect types, with 1,258 ground-truth masks** |

The total of 5,354 images is the same as the official MVTec AD count.

### End-to-end check of the running app
The backend and frontend were both running, and every request went through the frontend's `/api` proxy, the same path the browser uses:

| Check | Result |
|---|:-:|
| Admin login, then admin creates a Quality Engineer and a Supervisor | ✅ |
| Dataset page lists all 15 categories with correct counts | ✅ |
| Import the full `bottle` test split (83 images, 900×900) in about 8 s, with labels `good`, `broken_large`, `broken_small` and `contamination` kept | ✅ |
| Ground-truth defect mask served for defective images | ✅ |
| Camera simulation: 12 `screw` frames at a 40% defect rate | ✅ |
| Upload of one real image plus one corrupt file: 1 accepted, 1 rejected | ✅ |
| Supervisor can view images and the dashboard, but uploading is blocked (403) | ✅ |
| Engineer is blocked from user management (403) | ✅ |
| Dashboard stats add up correctly by category, source and validation status | ✅ |
| Frontend production build (`npm run build`) succeeds | ✅ |

### Fix found during verification: blur threshold recalibrated
On real MVTec photos, the first blur threshold flagged **90% of images as "may be blurry"**, which made the warning useless. I measured sharpness across all 15 categories:
- Sharpness is now measured at a fixed 512 px size, so the result doesn't depend on image resolution.
- On that scale, sharp MVTec images score between 11 and 1,100. A defocus blur of 1 px already brings a bottle image down to about 10, and 2 px brings it to about 3.
- The threshold is now **8**. Result: **all 5,354 real images pass as valid**, and the unit test with a genuinely blurred image still gets a warning.

---

## 8. File inventory

```
.gitignore
README.md
MILESTONE_1.md
docker-compose.yml
docs/architecture.md
docs/dataset.md
backend/requirements.txt
backend/Dockerfile
backend/.dockerignore
backend/.env.example
backend/app/main.py                    app setup, CORS, table creation, admin seeding
backend/app/db.py                      SQLAlchemy engine/session
backend/app/models.py                  User, Batch, InspectionImage + enums
backend/app/schemas.py                 Pydantic request/response models
backend/app/core/config.py             settings (env-driven)
backend/app/core/security.py           bcrypt + JWT
backend/app/core/deps.py               current-user + role guards
backend/app/routers/auth.py
backend/app/routers/users.py
backend/app/routers/batches.py
backend/app/routers/dataset.py
backend/app/routers/dashboard.py
backend/app/services/image_validation.py
backend/app/services/ingest.py
backend/app/services/mvtec.py
backend/scripts/make_sample_dataset.py
backend/tests/conftest.py
backend/tests/test_auth.py
backend/tests/test_inspection.py
frontend/package.json, vite.config.js, tailwind.config.js, postcss.config.js, index.html, Dockerfile
frontend/src/main.jsx, App.jsx, api.js, auth.jsx, index.css
frontend/src/components/ui.jsx
frontend/src/pages/Login.jsx, Dashboard.jsx, Upload.jsx, Batches.jsx, BatchDetail.jsx, Dataset.jsx, Users.jsx
```

---

## 9. Next: Milestone 2 (Weeks 3 & 4)

- Image preprocessing pipeline: resizing and normalization, noise removal, contrast enhancement and feature extraction
- An image quality analysis report
- Training a defect detection model on MVTec AD: an anomaly-detection model trained on the good images (PatchCore or an autoencoder), plus YOLO to locate defects
- An inference worker that processes `pending` images and stores results in `defect_predictions`
- Inspection monitoring dashboards that compare predictions with the ground-truth labels
