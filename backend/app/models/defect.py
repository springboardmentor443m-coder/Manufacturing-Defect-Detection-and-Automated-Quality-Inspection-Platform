from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, JSON, String
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.db.base_class import Base


class Defect(Base):
    __tablename__ = "defects"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    inspection_id = Column(
        Integer, ForeignKey("inspections.id", ondelete="CASCADE"), nullable=False, index=True
    )
    defect_type = Column(String(100), nullable=False, index=True)
    confidence = Column(Float, nullable=False)
    bounding_box = Column(JSON, nullable=True)  # Format: {"xmin": int, "ymin": int, "xmax": int, "ymax": int}
    segmentation_mask_path = Column(String(1024), nullable=True)

    # Granular scoring components for explainability
    # Severity score = Size*0.30 + Location*0.25 + DefectType*0.25 + Confidence*0.20
    size_score = Column(Float, default=0.0, nullable=False)
    location_score = Column(Float, default=0.0, nullable=False)
    defect_type_score = Column(Float, default=0.0, nullable=False)
    confidence_score = Column(Float, default=0.0, nullable=False)
    severity_score = Column(Float, default=0.0, nullable=False, index=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    # Relationships
    inspection = relationship("Inspection", back_populates="defects")
