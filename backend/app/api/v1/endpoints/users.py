from typing import Any, List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.deps import get_db, require_roles
from app.core.security import get_password_hash
from app.models.user import User, UserRole
from app.schemas.user import UserCreate, UserOut, UserRoleUpdate

router = APIRouter()


@router.get("", response_model=List[UserOut])
def list_users(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
    current_admin: User = Depends(require_roles("admin")),
) -> Any:
    """List all registered users. Restricted to Admin only."""
    users = db.query(User).offset(skip).limit(limit).all()
    return users


@router.post("", response_model=UserOut, status_code=status.HTTP_201_CREATED)
def create_user_by_admin(
    user_in: UserCreate,
    db: Session = Depends(get_db),
    current_admin: User = Depends(require_roles("admin")),
) -> Any:
    """Create a new user with any role. Restricted to Admin only."""
    existing_user = db.query(User).filter(User.email == user_in.email).first()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="A user with this email already exists.",
        )

    role = user_in.role or UserRole.QUALITY_ENGINEER.value
    valid_roles = [r.value for r in UserRole]
    if role not in valid_roles:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid role. Allowed roles: {', '.join(valid_roles)}",
        )

    user = User(
        email=user_in.email.lower().strip(),
        hashed_password=get_password_hash(user_in.password),
        full_name=user_in.full_name,
        role=role,
        is_active=True,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@router.patch("/{user_id}/role", response_model=UserOut)
def update_user_role(
    user_id: int,
    role_in: UserRoleUpdate,
    db: Session = Depends(get_db),
    current_admin: User = Depends(require_roles("admin")),
) -> Any:
    """Change a user's role. Restricted to Admin only."""
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )

    # Protect against self-demotion of the last active admin
    if user.id == current_admin.id and role_in.role != "admin":
        admin_count = (
            db.query(User)
            .filter(User.role == "admin", User.is_active.is_(True))
            .count()
        )
        if admin_count <= 1:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot demote the sole active system administrator.",
            )

    user.role = role_in.role
    db.commit()
    db.refresh(user)
    return user


@router.delete("/{user_id}", response_model=UserOut)
def deactivate_user(
    user_id: int,
    db: Session = Depends(get_db),
    current_admin: User = Depends(require_roles("admin")),
) -> Any:
    """Deactivate a user account. Restricted to Admin only."""
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )
    if user.id == current_admin.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot deactivate your own administrator account.",
        )

    user.is_active = False
    db.commit()
    db.refresh(user)
    return user
