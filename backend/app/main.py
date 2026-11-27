from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import select

from app.core.config import get_settings
from app.core.security import hash_password
from app.db import Base, SessionLocal, engine
from app.models import Role, User
from app.routers import auth, batches, dashboard, dataset, users

settings = get_settings()


def init_db() -> None:
    Base.metadata.create_all(bind=engine)
    settings.storage_dir.mkdir(parents=True, exist_ok=True)
    with SessionLocal() as db:
        if db.scalar(select(User.id).limit(1)) is None:
            db.add(
                User(
                    email=settings.first_admin_email.lower(),
                    full_name="Admin",
                    role=Role.admin,
                    hashed_password=hash_password(settings.first_admin_password),
                )
            )
            db.commit()


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    yield


app = FastAPI(title=settings.app_name, version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

for r in (auth.router, users.router, batches.router, dataset.router, dashboard.router):
    app.include_router(r, prefix=settings.api_prefix)


@app.get("/api/health", tags=["health"])
def health():
    return {"status": "ok", "app": settings.app_name}
