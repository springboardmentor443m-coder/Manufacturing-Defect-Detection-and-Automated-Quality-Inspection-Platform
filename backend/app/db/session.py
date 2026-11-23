import logging
import socket
from typing import Generator
from urllib.parse import urlparse
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from app.core.config import settings
from app.db.base_class import Base

logger = logging.getLogger("uvicorn.error")


def is_postgres_reachable(url: str, timeout: float = 0.8) -> bool:
    """Quick socket probe to check if the PostgreSQL server is accepting connections."""
    try:
        parsed = urlparse(url)
        host = parsed.hostname or "127.0.0.1"
        port = parsed.port or 5432
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except Exception:
        return False


# Determine active database URL
active_url = settings.DATABASE_URL
connect_args = {}

if "postgresql" in active_url:
    if not is_postgres_reachable(active_url):
        fallback_url = "sqlite:///./visioninspect.db"
        logger.warning(
            f"⚠️ PostgreSQL on port 5432 is not reachable. Automatically falling back to local SQLite: {fallback_url}"
        )
        active_url = fallback_url
        connect_args["check_same_thread"] = False
elif "sqlite" in active_url:
    connect_args["check_same_thread"] = False

engine = create_engine(
    active_url,
    pool_pre_ping=True,
    connect_args=connect_args,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db_if_needed():
    """Ensure tables and seed users exist if using SQLite fallback."""
    if "sqlite" in active_url:
        import app.models  # noqa: F401 - ensure all models are registered
        Base.metadata.create_all(bind=engine)
        # Check if users need seeding
        db = SessionLocal()
        try:
            from app.models.user import User
            if db.query(User).count() == 0:
                logger.info("🌱 Seeding default users and products into SQLite...")
                from scripts.seed import seed_database
                seed_database()
        except Exception as e:
            logger.warning(f"Could not auto-seed SQLite: {e}")
        finally:
            db.close()


init_db_if_needed()


def get_db() -> Generator[Session, None, None]:
    """Dependency that provides an active database session and closes it afterwards."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
