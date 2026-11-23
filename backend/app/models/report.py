from sqlalchemy import Column, DateTime, ForeignKey, Integer, JSON, Text
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.db.base_class import Base


class Report(Base):
    __tablename__ = "reports"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    inspection_id = Column(
        Integer, ForeignKey("inspections.id", ondelete="CASCADE"), nullable=False, index=True
    )
    summary = Column(Text, nullable=True)
    notes = Column(Text, nullable=True)
    generated_by = Column(
        Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True
    )
    report_data = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    # Relationships
    inspection = relationship("Inspection", back_populates="reports")
    generator = relationship("User", back_populates="reports")
