from datetime import datetime
from typing import Optional
from pydantic import BaseModel, EmailStr, Field


class UserBase(BaseModel):
    email: EmailStr
    full_name: Optional[str] = None
    role: Optional[str] = Field(default="quality_engineer")
    is_active: Optional[bool] = True


class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(min_length=6, description="Password must be at least 6 characters")
    full_name: Optional[str] = None
    role: Optional[str] = Field(default="quality_engineer")


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class UserUpdate(BaseModel):
    email: Optional[EmailStr] = None
    full_name: Optional[str] = None
    password: Optional[str] = None
    is_active: Optional[bool] = None


class UserRoleUpdate(BaseModel):
    role: str = Field(..., pattern="^(quality_engineer|supervisor|admin)$")


class UserOut(BaseModel):
    id: int
    email: EmailStr
    full_name: Optional[str] = None
    role: str
    is_active: bool
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True
