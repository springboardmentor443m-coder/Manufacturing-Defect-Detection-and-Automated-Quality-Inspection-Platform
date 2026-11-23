from datetime import datetime
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class DefectBase(BaseModel):
    defect_type: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    bounding_box: Optional[Dict[str, Any]] = None
    segmentation_mask_path: Optional[str] = None
    size_score: Optional[float] = 0.0
    location_score: Optional[float] = 0.0
    defect_type_score: Optional[float] = 0.0
    confidence_score: Optional[float] = 0.0
    severity_score: Optional[float] = 0.0


class DefectCreate(DefectBase):
    inspection_id: int


class DefectOut(DefectBase):
    id: int
    inspection_id: int
    created_at: datetime

    class Config:
        from_attributes = True
