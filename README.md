# VisionInspect AI 🔍⚙️

AI-powered manufacturing defect detection and quality inspection platform.

---

## Architecture & Technology Stack

- **Backend**: Python 3.10+, FastAPI (Application Factory), SQLAlchemy 2.0, Alembic, PostgreSQL, JWT Authentication (bcrypt + PyJWT), Role-Based Access Control (RBAC).
- **Frontend**: Next.js 14 (App Router), React 18, TypeScript, Tailwind CSS, Lucide React icons.
- **ML & Data (Upcoming Phases)**: PyTorch, OpenCV, Ultralytics YOLO, Anomalib, MVTec AD, NEU-DET.
- **DevOps**: Docker, Docker Compose.

---

## Repository Structure

```
visioninspect/
├── backend/                      # FastAPI Application
│   ├── alembic/                  # Database migration scripts & env
│   │   └── versions/             # Migration version files
│   ├── app/
│   │   ├── api/                  # API routes & dependency injection
│   │   │   ├── deps.py           # get_db, get_current_user, require_roles
│   │   │   └── v1/
│   │   │       ├── api.py        # Centralized v1 router
│   │   │       └── endpoints/    # auth, health, inspections, products, users
│   │   ├── core/                 # Config (BaseSettings) & security (bcrypt, JWT)
│   │   ├── db/                   # Database session & Base class
│   │   ├── models/               # SQLAlchemy models (User, Product, Inspection, Defect, Report)
│   │   ├── schemas/              # Pydantic schemas
│   │   └── main.py               # App factory (create_app), CORS, OpenAPI docs
│   ├── scripts/
│   │   └── seed.py               # Database seeder (Admin, Supervisor, Engineer, Sample Data)
│   ├── alembic.ini               # Alembic configuration
│   ├── requirements.txt          # Python dependencies
│   └── .env.example              # Backend environment template
├── frontend/                     # Next.js 14 App Router
│   ├── src/
│   │   ├── app/
│   │   │   ├── (dashboard)/      # Protected dashboard routes (Role-aware)
│   │   │   │   ├── dashboard/    # Overview & scoring matrix
│   │   │   │   ├── inspections/  # Inspection queue & registry (QE, Supervisor, Admin)
│   │   │   │   ├── products/     # Product catalog (Supervisor, Admin)
│   │   │   │   ├── users/        # User management (ADMIN ONLY)
│   │   │   │   └── reports/      # Quality audit placeholder (Supervisor, Admin)
│   │   │   ├── login/            # Industrial login with demo quick-fill
│   │   │   ├── register/         # Inspector registration
│   │   │   ├── globals.css       # Tailwind CSS styles
│   │   │   └── layout.tsx        # Root layout with AuthProvider
│   │   ├── components/           # Sidebar, Navbar, ProtectedRoute
│   │   ├── context/              # AuthContext (JWT session, roles, persistent state)
│   │   ├── lib/                  # API client fetch wrapper
│   │   └── types/                # TypeScript interfaces
│   ├── package.json              # Frontend npm dependencies
│   ├── tailwind.config.js        # Industrial theme color tokens
│   └── tsconfig.json             # TypeScript configuration
├── ml/                           # ML pipelines, models, and notebooks
│   └── README.md
├── data/                         # Datasets (MVTec AD, NEU-DET, git-ignored)
│   └── README.md
├── docker-compose.yml            # PostgreSQL 15 database container
├── .env.example                  # Root environment template
├── .gitignore                    # Comprehensive ignores (venv, node_modules, data, weights)
└── README.md
```

---

## Roles & Access Control Matrix

| Role | Dashboard | Inspections | Products | Reports | User Management (Admin Only) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`quality_engineer`** | ✅ | ✅ | ❌ | ❌ | ❌ |
| **`supervisor`** | ✅ | ✅ | ✅ | ✅ | ❌ |
| **`admin`** | ✅ | ✅ | ✅ | ✅ | ✅ |

---

## Defect Severity Scoring & Decision Pipeline

$$\text{Severity (0-100)} = \text{Size} \times 0.30 + \text{Location} \times 0.25 + \text{DefectType} \times 0.25 + \text{Confidence} \times 0.20$$

- **Critical (80–100)**: **Reject** (Component discarded / line stopped)
- **High (60–79)**: **Rework** (Component routed to buffing / refinishing)
- **Medium (40–59)**: **Manual Review** (Operator visual check; also triggered if confidence < 70%)
- **Low (0–39)**: **Pass** (Cleared for line progression)

---

## Quickstart Guide

### 1. Start the Database (PostgreSQL)

Ensure Docker Desktop is running, then execute:

```powershell
docker compose up -d
```

Verify that the `visioninspect_postgres` container is running and healthy:
```powershell
docker compose ps
```

---

### 2. Set Up & Start Backend

Navigate into `backend`:
```powershell
cd backend
```

Create and activate a virtual environment:
```powershell
# Windows PowerShell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

Install requirements:
```powershell
pip install -r requirements.txt
```

Run Alembic database migrations:
```powershell
alembic upgrade head
```

Seed initial accounts and sample data:
```powershell
python scripts/seed.py
```

Start the FastAPI application server:
```powershell
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

- Swagger UI interactive documentation: [http://localhost:8000/api/v1/docs](http://localhost:8000/api/v1/docs)
- Health Check: [http://localhost:8000/api/v1/health](http://localhost:8000/api/v1/health)

---

### 3. Set Up & Start Frontend

Open a new terminal tab and navigate into `frontend`:
```powershell
cd frontend
npm install
npm run dev
```

Open your browser to: [http://localhost:3000](http://localhost:3000)

---

## Seeded Default Credentials

| Role | Email | Password | Allowed Access |
| :--- | :--- | :--- | :--- |
| **Admin** | `admin@visioninspect.ai` | `Admin@123` | Full access + User Management |
| **Supervisor** | `supervisor@visioninspect.ai` | `Super@123` | Inspections, Products, Reports |
| **Quality Engineer** | `qe@visioninspect.ai` | `Engineer@123` | Dashboard & Inspections only |

*(Quick-login buttons are also provided directly on the Login page for rapid one-click role switching.)*

---

## Phase 1 Verification Checklist

- [x] **PostgreSQL Service**: Database running with persistent volume and healthcheck on port `5432`.
- [x] **Database Schema**: Tables `users`, `products`, `inspections`, `defects`, `reports` provisioned with foreign keys, timestamps, indexes, and constraints.
- [x] **Alembic Migrations**: Initial migration `0001_initial_schema` runs cleanly via `alembic upgrade head`.
- [x] **Database Seeding**: `python scripts/seed.py` creates default users and sample dataset entries.
- [x] **Authentication Endpoints**:
  - `POST /api/v1/auth/register` (Registers new inspector account)
  - `POST /api/v1/auth/login` (Returns JWT access token with user claims)
  - `GET /api/v1/auth/me` (Returns authenticated user profile)
- [x] **Role Dependency**: `require_roles(...)` successfully restricts endpoints (e.g. `GET /api/v1/users` returns 403 Forbidden for non-admins).
- [x] **Frontend Navigation**: Sidebar dynamically renders links according to the logged-in role.
- [x] **Role-Based Guards**:
  - `admin` sees and can access `/users` (User Management).
  - `supervisor` and `quality_engineer` do not see `/users` in the sidebar and are blocked with an "Access Restricted" alert if navigated to directly.
