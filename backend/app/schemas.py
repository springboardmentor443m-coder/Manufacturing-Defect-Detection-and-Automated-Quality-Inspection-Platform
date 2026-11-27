from datetime import datetime

from pydantic import BaseModel, ConfigDict, EmailStr, Field

from app.models import BatchStatus, ImageSource, InspectionStatus, Role, ValidationStatus


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserBase(BaseModel):
    email: EmailStr
    full_name: str = Field(min_length=1, max_length=255)
    role: Role = Role.quality_engineer


class UserCreate(UserBase):
    password: str = Field(min_length=8, max_length=72)


class UserUpdate(BaseModel):
    full_name: str | None = None
    role: Role | None = None
    is_active: bool | None = None
    password: str | None = Field(default=None, min_length=8, max_length=72)


class UserOut(UserBase):
    model_config = ConfigDict(from_attributes=True)
    id: int
    is_active: bool
    created_at: datetime


class ImageOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    batch_id: int
    original_filename: str
    product_category: str
    source: ImageSource
    width: int
    height: int
    channels: int
    file_size: int
    blur_score: float
    brightness: float
    contrast: float
    validation_status: ValidationStatus
    validation_messages: list[str]
    ground_truth_label: str | None
    inspection_status: InspectionStatus
    created_at: datetime


class BatchOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    name: str
    product_category: str
    source: ImageSource
    status: BatchStatus
    created_at: datetime
    created_by_name: str
    image_count: int
    valid_count: int
    warning_count: int
    rejected_count: int


class BatchDetail(BatchOut):
    images: list[ImageOut]


class UploadResult(BaseModel):
    batch: BatchOut
    accepted: int
    rejected: int
    duplicates: list[str]


class CameraCaptureRequest(BaseModel):
    category: str
    count: int = Field(default=5, ge=1, le=50)
    defect_ratio: float = Field(default=0.3, ge=0, le=1)
    station: str = "Line-1 / Cam-A"


class DatasetCategory(BaseModel):
    name: str
    train_good: int
    test_good: int
    test_defective: int
    defect_types: dict[str, int]


class DatasetSummary(BaseModel):
    root: str
    available: bool
    categories: list[DatasetCategory]


class DatasetImportRequest(BaseModel):
    category: str
    split: str = Field(default="test", pattern="^(train|test)$")
    limit: int = Field(default=50, ge=1, le=500)


class DashboardStats(BaseModel):
    total_images: int
    total_batches: int
    pending_inspection: int
    by_validation: dict[str, int]
    by_category: dict[str, int]
    by_source: dict[str, int]
    by_ground_truth: dict[str, int]
    uploads_per_day: list[dict]
    recent_batches: list[BatchOut]
