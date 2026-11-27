from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=BACKEND_DIR / ".env", extra="ignore")

    app_name: str = "VisionInspect AI"
    api_prefix: str = "/api"

    # SQLite by default so the app runs without Docker; docker-compose switches to PostgreSQL.
    database_url: str = f"sqlite:///{(BACKEND_DIR / 'visioninspect.db').as_posix()}"

    jwt_secret: str = "change-me-in-production"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60 * 8

    # Seeded on first start so someone can log in and create the other accounts.
    first_admin_email: str = "admin@visioninspect.ai"
    first_admin_password: str = "Admin@123"

    storage_dir: Path = BACKEND_DIR / "storage"
    # Root of the extracted MVTec AD dataset (folder that contains bottle/, cable/, ...).
    mvtec_root: Path = BACKEND_DIR.parent / "data" / "mvtec_anomaly_detection"

    max_upload_mb: int = 15
    min_image_side: int = 64
    # Laplacian variance (measured at 512px) below this → blurry warning.
    # Calibrated on MVTec AD: sharp test images score 11–1100 (1st percentile ≈ 21),
    # while a 1px Gaussian defocus already drops a bottle image to ≈ 10 and 2px to ≈ 3.
    blur_threshold: float = 8.0

    cors_origins: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]


@lru_cache
def get_settings() -> Settings:
    return Settings()
