# VisionInspect AI

An AI-powered platform for manufacturing defect detection and quality inspection, built for the Infosys Springboard internship.

| Milestone | Weeks | Scope | Status |
|---|---|---|---|
| 1 | 1–2 | Architecture, auth and roles, image acquisition, MVTec AD integration, inspection dashboard | ✅ see [MILESTONE_1.md](MILESTONE_1.md) |
| 2 | 3–4 | Image preprocessing, defect detection models | ⏳ |
| 3 | 5–6 | Defect classification, severity scoring, analytics | ⏳ |
| 4 | 7–8 | Testing, Docker/cloud deployment, docs | ⏳ |

## Run locally

**Backend** (Python 3.11 or newer):

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate          # Windows  (source .venv/bin/activate on macOS/Linux)
pip install -r requirements.txt
python scripts/make_sample_dataset.py   # optional demo data until MVTec AD is downloaded
uvicorn app.main:app --reload
```

The API runs at http://localhost:8000, with interactive docs at http://localhost:8000/docs.
Default admin: `admin@visioninspect.ai` / `Admin@123`. Change these with `FIRST_ADMIN_EMAIL` and `FIRST_ADMIN_PASSWORD` in `backend/.env`.

**Frontend** (Node 18 or newer):

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173. Requests to `/api` are forwarded to the backend.

**Tests:**

```bash
cd backend
pytest
```

**Docker (PostgreSQL + backend + frontend):**

```bash
docker compose up --build
```

## Project layout

```
backend/
  app/
    core/        config, JWT + bcrypt security, role-based access dependencies
    routers/     auth, users, batches (upload/images), dataset (MVTec + camera), dashboard
    services/    image_validation, ingest, mvtec
    models.py    SQLAlchemy schema
    schemas.py   Pydantic request/response models
  scripts/make_sample_dataset.py
  tests/
frontend/src/
  pages/         Login, Dashboard, Upload, Batches, BatchDetail, Dataset, Users
  components/    shared UI (badges, stat cards, authenticated image loader)
docs/            architecture.md, dataset.md
```
