from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from app.schemas.defect import DefectOut
from app.schemas.report import ReportOut


class PreprocessingConfigSchema(BaseModel):
    """Configurable options for image preprocessing pipeline."""
    do_validate: bool = True
    do_resize: bool = True
    target_width: int = 512
    target_height: int = 512
    keep_aspect_ratio: bool = False
    do_denoise: bool = True
    denoise_method: str = "gaussian"  # "gaussian", "median", "bilateral"
    denoise_kernel_size: int = 3
    do_clahe: bool = True
    clahe_clip_limit: float = 2.0
    clahe_grid_size: int = 8
    do_edge_detection: bool = False
    edge_low_threshold: int = 50
    edge_high_threshold: int = 150
    do_roi_extraction: bool = False
    roi_mode: str = "center"  # "center", "bbox", "auto_foreground"
    roi_margin_percent: float = 0.1
    roi_custom_box: Optional[List[int]] = None


class InspectionBase(BaseModel):
    product_id: int
    image_path: str
    original_filename: Optional[str] = None
    file_size_bytes: Optional[int] = None
    image_width: Optional[int] = None
    image_height: Optional[int] = None
    mime_type: Optional[str] = None
    extra_metadata: Optional[Dict[str, Any]] = None
    status: Optional[str] = "pending"
    decision: Optional[str] = None
    overall_severity: Optional[float] = Field(default=0.0, ge=0.0, le=100.0)
    
    # Phase 3 Fields
    processed_image_path: Optional[str] = None
    quality_score: Optional[float] = None
    quality_verdict: Optional[str] = None
    quality_report: Optional[Dict[str, Any]] = None
    preprocessing_config: Optional[Dict[str, Any]] = None


class InspectionCreate(BaseModel):
    product_id: int
    image_path: str
    original_filename: Optional[str] = None
    file_size_bytes: Optional[int] = None
    image_width: Optional[int] = None
    image_height: Optional[int] = None
    mime_type: Optional[str] = None
    extra_metadata: Optional[Dict[str, Any]] = None


class InspectionUpdate(BaseModel):
    status: Optional[str] = None
    decision: Optional[str] = None
    overall_severity: Optional[float] = Field(default=None, ge=0.0, le=100.0)
    extra_metadata: Optional[Dict[str, Any]] = None
    processed_image_path: Optional[str] = None
    quality_score: Optional[float] = None
    quality_verdict: Optional[str] = None
    quality_report: Optional[Dict[str, Any]] = None
    preprocessing_config: Optional[Dict[str, Any]] = None


class InspectionOut(InspectionBase):
    id: int
    created_by: int
    created_at: datetime
    updated_at: datetime
    image_url: Optional[str] = None
    processed_image_url: Optional[str] = None
    defects: List[DefectOut] = []
    reports: List[ReportOut] = []

    class Config:
        from_attributes = True


class InspectionListResponse(BaseModel):
    items: List[InspectionOut]
    total: int
    page: int
    page_size: int
    total_pages: int


class UploadErrorItem(BaseModel):
    filename: str
    error: str


class BatchUploadResponse(BaseModel):
    total: int
    successful: int
    failed: int
    inspections: List[InspectionOut]
    errors: List[UploadErrorItem]


class QualityReportResponse(BaseModel):
    inspection_id: int
    verdict: str  # "pass", "warn", "fail"
    overall_score: float  # 0 - 100
    reasons: List[str]
    resolution: Dict[str, Any]
    blur_score: float
    brightness: float
    contrast: float
    noise_estimate: float
    metrics_breakdown: Dict[str, Any]
    features: Dict[str, Any]
    original_image_url: str
    processed_image_url: Optional[str] = None


class PreprocessResponse(BaseModel):
    inspection_id: int
    message: str
    processed_image_url: str
    processed_image_path: str
    quality_verdict: str
    quality_score: float
    pipeline_metadata: Dict[str, Any]
    quality_report: Dict[str, Any]
