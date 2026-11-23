import os
import shutil
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from app.core.config import settings


class BaseStorageService(ABC):
    """Abstract base class for image storage providers (Local disk, S3, Azure Blob, etc.)."""

    @abstractmethod
    def save(
        self,
        file_bytes: bytes,
        original_filename: str,
        content_type: str,
        subfolder: Optional[str] = None,
    ) -> str:
        """
        Save file bytes to storage.
        Returns the persistent identifier / relative path of the stored file.
        """
        pass

    @abstractmethod
    def get(self, file_path: str) -> bytes:
        """Retrieve file bytes from storage."""
        pass

    @abstractmethod
    def delete(self, file_path: str) -> bool:
        """Delete file from storage."""
        pass

    @abstractmethod
    def get_url(self, file_path: str) -> str:
        """Get the accessible URL / path for frontend consumption."""
        pass


class LocalStorageService(BaseStorageService):
    """
    Local filesystem storage provider.
    Saves files into configured UPLOAD_DIR with date-based hierarchy and unique UUIDs.
    """

    def __init__(self, base_dir: Optional[str] = None):
        self.base_dir = Path(base_dir or settings.UPLOAD_DIR).resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _normalize_path(self, file_path: str) -> Path:
        """Resolve path safely to prevent directory traversal attacks."""
        clean_path = file_path.lstrip("/").replace("\\", "/")
        # Remove any leading 'uploads/' prefix if passed
        if clean_path.startswith("uploads/"):
            clean_path = clean_path[len("uploads/") :]
        full_path = (self.base_dir / clean_path).resolve()
        if not str(full_path).startswith(str(self.base_dir)):
            raise ValueError("Invalid file path: directory traversal detected.")
        return full_path

    def save(
        self,
        file_bytes: bytes,
        original_filename: str,
        content_type: str,
        subfolder: Optional[str] = None,
    ) -> str:
        # Determine file extension
        _, ext = os.path.splitext(original_filename)
        ext = ext.lower() if ext else ".jpg"

        # Unique filename using UUID4
        unique_name = f"{uuid.uuid4().hex}{ext}"

        # Subdirectory structure: YYYY/MM/DD or custom subfolder
        if subfolder:
            rel_dir = Path(subfolder)
        else:
            now = datetime.now(timezone.utc)
            rel_dir = Path(f"{now.year:04d}/{now.month:02d}/{now.day:02d}")

        target_dir = self.base_dir / rel_dir
        target_dir.mkdir(parents=True, exist_ok=True)

        target_file = target_dir / unique_name
        with open(target_file, "wb") as f:
            f.write(file_bytes)

        # Return standardized relative path using forward slashes
        relative_path = (rel_dir / unique_name).as_posix()
        return f"uploads/{relative_path}"

    def get(self, file_path: str) -> bytes:
        full_path = self._normalize_path(file_path)
        if not full_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        with open(full_path, "rb") as f:
            return f.read()

    def delete(self, file_path: str) -> bool:
        try:
            full_path = self._normalize_path(file_path)
            if full_path.exists() and full_path.is_file():
                full_path.unlink()
                return True
            return False
        except Exception:
            return False

    def get_url(self, file_path: str) -> str:
        clean = file_path.lstrip("/").replace("\\", "/")
        if not clean.startswith("uploads/"):
            clean = f"uploads/{clean}"
        return f"/{clean}"


class S3StorageService(BaseStorageService):
    """
    AWS S3 storage adapter.
    Can be seamlessly enabled via configuration without changing business logic.
    """

    def __init__(self, bucket_name: str = "", region: str = "us-east-1"):
        self.bucket_name = bucket_name
        self.region = region

    def save(
        self,
        file_bytes: bytes,
        original_filename: str,
        content_type: str,
        subfolder: Optional[str] = None,
    ) -> str:
        # Placeholder for boto3 s3_client.put_object
        raise NotImplementedError("S3 storage requires AWS credentials and boto3 configuration.")

    def get(self, file_path: str) -> bytes:
        raise NotImplementedError("S3 storage requires AWS credentials and boto3 configuration.")

    def delete(self, file_path: str) -> bool:
        raise NotImplementedError("S3 storage requires AWS credentials and boto3 configuration.")

    def get_url(self, file_path: str) -> str:
        return f"https://{self.bucket_name}.s3.{self.region}.amazonaws.com/{file_path}"


# Singleton instance
_storage_instance: Optional[BaseStorageService] = None


def get_storage_service() -> BaseStorageService:
    """Dependency provider for storage service abstraction."""
    global _storage_instance
    if _storage_instance is None:
        _storage_instance = LocalStorageService()
    return _storage_instance
