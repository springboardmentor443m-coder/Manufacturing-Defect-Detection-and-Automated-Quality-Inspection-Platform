"""Turns raw image bytes into stored, validated InspectionImage rows.

Every way an image enters the system (upload, simulated camera, dataset import)
goes through `ingest_image`, so validation and storage behave the same everywhere.
"""

import hashlib
import uuid
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.models import (
    Batch,
    ImageSource,
    InspectionImage,
    InspectionStatus,
    User,
    ValidationStatus,
)
from app.services.image_validation import validate_image


def is_duplicate(db: Session, sha256: str) -> bool:
    return db.scalar(select(InspectionImage.id).where(InspectionImage.sha256 == sha256)) is not None


def ingest_image(
    db: Session,
    *,
    batch: Batch,
    data: bytes,
    filename: str,
    source: ImageSource,
    user: User,
    ground_truth_label: str | None = None,
    ground_truth_mask_path: str | None = None,
) -> InspectionImage | None:
    """Validate and store one image. Returns None if the exact file was already ingested."""
    sha256 = hashlib.sha256(data).hexdigest()
    if is_duplicate(db, sha256):
        return None

    result = validate_image(data, filename)

    settings = get_settings()
    rel_path = Path("images") / str(batch.id) / f"{uuid.uuid4().hex}{Path(filename).suffix.lower()}"
    abs_path = settings.storage_dir / rel_path
    abs_path.parent.mkdir(parents=True, exist_ok=True)
    abs_path.write_bytes(data)

    image = InspectionImage(
        batch=batch,
        original_filename=Path(filename).name,
        stored_path=rel_path.as_posix(),
        product_category=batch.product_category,
        source=source,
        sha256=sha256,
        width=result.width,
        height=result.height,
        channels=result.channels,
        file_size=len(data),
        blur_score=result.blur_score,
        brightness=result.brightness,
        contrast=result.contrast,
        validation_status=result.status,
        validation_messages=result.messages,
        ground_truth_label=ground_truth_label,
        ground_truth_mask_path=ground_truth_mask_path,
        inspection_status=(
            InspectionStatus.skipped
            if result.status == ValidationStatus.rejected
            else InspectionStatus.pending
        ),
        uploaded_by_id=user.id,
    )
    db.add(image)
    return image
