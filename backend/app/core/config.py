from typing import List, Union
from pydantic import AnyHttpUrl, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    ENVIRONMENT: str = "development"
    DEBUG: bool = True
    PROJECT_NAME: str = "VisionInspect AI"
    API_V1_STR: str = "/api/v1"

    # Security
    SECRET_KEY: str = "dev_secret_key_change_in_production_min_32_characters_12345"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24  # 24 hours

    # Database
    POSTGRES_SERVER: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_USER: str = "vision_admin"
    POSTGRES_PASSWORD: str = "vision_pass123"
    POSTGRES_DB: str = "visioninspect_db"
    DATABASE_URL: str = (
        "postgresql+psycopg2://vision_admin:vision_pass123@localhost:5432/visioninspect_db"
    )

    # CORS
    CORS_ORIGINS: Union[List[str], str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3001",
    ]

    @field_validator("CORS_ORIGINS", mode="before")
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> List[str]:
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",") if i.strip()]
        elif isinstance(v, list):
            return v
        return ["http://localhost:3000", "http://127.0.0.1:3000"]

    # Initial Admin Seed
    FIRST_SUPERUSER_EMAIL: str = "admin@visioninspect.ai"
    FIRST_SUPERUSER_PASSWORD: str = "Admin@123"
    FIRST_SUPERUSER_NAME: str = "System Administrator"

    # Storage & Uploads
    UPLOAD_DIR: str = "uploads"
    MAX_UPLOAD_SIZE_MB: int = 20
    ALLOWED_IMAGE_EXTENSIONS: List[str] = [
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tiff",
        ".tif",
        ".webp",
    ]
    ALLOWED_IMAGE_MIME_TYPES: List[str] = [
        "image/jpeg",
        "image/png",
        "image/bmp",
        "image/tiff",
        "image/webp",
    ]

    @property
    def max_upload_size_bytes(self) -> int:
        return self.MAX_UPLOAD_SIZE_MB * 1024 * 1024

    model_config = SettingsConfigDict(
        env_file=(".env", "../.env"),
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )


settings = Settings()
