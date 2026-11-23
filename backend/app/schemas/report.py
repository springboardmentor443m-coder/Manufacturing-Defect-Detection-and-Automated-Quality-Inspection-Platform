from datetime import datetime
from typing import Any, Dict, Optional
from pydantic import BaseModel


class ReportBase(BaseModel):
    summary: Optional[str] = None
    notes: Optional[str] = None
    report_data: Optional[Dict[str, Any]] = None


class ReportCreate(ReportBase):
    inspection_id: int


class ReportOut(ReportBase):
    id: int
    inspection_id: int
    generated_by: Optional[int] = None
    created_at: datetime

    class Config:
        from_attributes = True
