from typing import Optional

from pydantic import BaseModel, EmailStr


class UserCreate(BaseModel):
    full_name: str
    email: EmailStr
    password: str
    role: str = "qa_engineer"


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: dict


class InspectionCreate(BaseModel):
    file_name: str
    file_path: str


class InspectionResult(BaseModel):
    id: int
    user_id: int
    file_name: str
    file_path: str
    status: str
    defect_type: Optional[str] = None
    confidence: Optional[float] = None
    severity: Optional[str] = None
    notes: Optional[str] = None

    class Config:
        from_attributes = True
