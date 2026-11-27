"""Relational schema for VisionInspect AI.

Milestone 1 covers users, inspection batches and the images inside them.
Defect predictions (Milestone 2) and severity/quality decisions (Milestone 3)
hang off InspectionImage and are added in later migrations.
"""

import enum
from datetime import datetime, timezone

from sqlalchemy import JSON, DateTime, Enum, Float, ForeignKey, Integer, String, Boolean
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db import Base


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Role(str, enum.Enum):
    admin = "admin"
    quality_engineer = "quality_engineer"
    supervisor = "supervisor"


class ImageSource(str, enum.Enum):
    upload = "upload"
    camera = "camera"
    dataset = "dataset"


class ValidationStatus(str, enum.Enum):
    valid = "valid"
    warning = "warning"
    rejected = "rejected"


class InspectionStatus(str, enum.Enum):
    pending = "pending"  # waiting for the defect detection model (Milestone 2)
    inspected = "inspected"
    skipped = "skipped"  # rejected at validation, never sent to the model


class BatchStatus(str, enum.Enum):
    open = "open"
    processing = "processing"
    completed = "completed"


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    full_name: Mapped[str] = mapped_column(String(255))
    hashed_password: Mapped[str] = mapped_column(String(255))
    role: Mapped[Role] = mapped_column(Enum(Role), default=Role.quality_engineer)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Batch(Base):
    __tablename__ = "batches"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(255))
    product_category: Mapped[str] = mapped_column(String(100), index=True)
    source: Mapped[ImageSource] = mapped_column(Enum(ImageSource))
    status: Mapped[BatchStatus] = mapped_column(Enum(BatchStatus), default=BatchStatus.open)
    created_by_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    created_by: Mapped[User] = relationship()
    images: Mapped[list["InspectionImage"]] = relationship(
        back_populates="batch", cascade="all, delete-orphan"
    )


class InspectionImage(Base):
    __tablename__ = "inspection_images"

    id: Mapped[int] = mapped_column(primary_key=True)
    batch_id: Mapped[int] = mapped_column(ForeignKey("batches.id"), index=True)
    original_filename: Mapped[str] = mapped_column(String(255))
    stored_path: Mapped[str] = mapped_column(String(500))
    product_category: Mapped[str] = mapped_column(String(100), index=True)
    source: Mapped[ImageSource] = mapped_column(Enum(ImageSource))
    sha256: Mapped[str] = mapped_column(String(64), index=True)

    width: Mapped[int] = mapped_column(Integer)
    height: Mapped[int] = mapped_column(Integer)
    channels: Mapped[int] = mapped_column(Integer)
    file_size: Mapped[int] = mapped_column(Integer)

    # Image quality metrics computed at validation time
    blur_score: Mapped[float] = mapped_column(Float)
    brightness: Mapped[float] = mapped_column(Float)
    contrast: Mapped[float] = mapped_column(Float)
    validation_status: Mapped[ValidationStatus] = mapped_column(Enum(ValidationStatus))
    validation_messages: Mapped[list] = mapped_column(JSON, default=list)

    # Known label when the image comes from MVTec AD: "good" or a defect type like "crack"
    ground_truth_label: Mapped[str | None] = mapped_column(String(100), nullable=True)
    ground_truth_mask_path: Mapped[str | None] = mapped_column(String(500), nullable=True)

    inspection_status: Mapped[InspectionStatus] = mapped_column(
        Enum(InspectionStatus), default=InspectionStatus.pending
    )
    uploaded_by_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    batch: Mapped[Batch] = relationship(back_populates="images")
