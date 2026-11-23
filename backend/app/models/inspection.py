import enum
from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, JSON, String
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.db.base_class import Base


class InspectionStatus(str, enum.Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class InspectionDecision(str, enum.Enum):
    PASS = "pass"
    FAIL = "fail"
    REWORK = "rework"
    MANUAL_REVIEW = "manual_review"


class Inspection(Base):
    __tablename__ = "inspections"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    product_id = Column(Integer, ForeignKey("products.id", ondelete="CASCADE"), nullable=False, index=True)
    image_path = Column(String(1024), nullable=False)
    original_filename = Column(String(255), nullable=True)
    file_size_bytes = Column(Integer, nullable=True)
    image_width = Column(Integer, nullable=True)
    image_height = Column(Integer, nullable=True)
    mime_type = Column(String(50), nullable=True)
    extra_metadata = Column(JSON, nullable=True)
    
    # Preprocessing & Quality Analysis (Phase 3)
    processed_image_path = Column(String(1024), nullable=True)
    quality_score = Column(Float, nullable=True)
    quality_verdict = Column(String(50), nullable=True, index=True)
    quality_report = Column(JSON, nullable=True)
    preprocessing_config = Column(JSON, nullable=True)

    status = Column(
        String(50),
        nullable=False,
        default=InspectionStatus.PENDING.value,
        index=True,
    )
    decision = Column(String(50), nullable=True, index=True)
    overall_severity = Column(Float, default=0.0, nullable=False)
    created_by = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    # Relationships
    product = relationship("Product", back_populates="inspections")
    creator = relationship("User", back_populates="inspections")
    defects = relationship("Defect", back_populates="inspection", cascade="all, delete-orphan")
    reports = relationship("Report", back_populates="inspection", cascade="all, delete-orphan")
