from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.orm import Session, selectinload

from app.core.deps import can_view
from app.db import get_db
from app.models import Batch, InspectionImage, InspectionStatus, User
from app.routers.batches import batch_out
from app.schemas import DashboardStats

router = APIRouter(prefix="/dashboard", tags=["dashboard"])


def _group_count(db: Session, column) -> dict[str, int]:
    rows = db.execute(select(column, func.count()).group_by(column)).all()
    return {(getattr(k, "value", k) if k is not None else "unlabelled"): n for k, n in rows}


@router.get("/stats", response_model=DashboardStats)
def stats(db: Session = Depends(get_db), _: User = Depends(can_view)):
    since = datetime.now(timezone.utc) - timedelta(days=13)
    recent = db.scalars(select(InspectionImage.created_at).where(InspectionImage.created_at >= since)).all()
    per_day = {(since + timedelta(days=i)).date().isoformat(): 0 for i in range(14)}
    for ts in recent:
        key = ts.date().isoformat()
        if key in per_day:
            per_day[key] += 1

    recent_batches = db.scalars(
        select(Batch)
        .options(selectinload(Batch.images), selectinload(Batch.created_by))
        .order_by(Batch.created_at.desc())
        .limit(8)
    ).all()

    return DashboardStats(
        total_images=db.scalar(select(func.count(InspectionImage.id))) or 0,
        total_batches=db.scalar(select(func.count(Batch.id))) or 0,
        pending_inspection=db.scalar(
            select(func.count(InspectionImage.id)).where(
                InspectionImage.inspection_status == InspectionStatus.pending
            )
        )
        or 0,
        by_validation=_group_count(db, InspectionImage.validation_status),
        by_category=_group_count(db, InspectionImage.product_category),
        by_source=_group_count(db, InspectionImage.source),
        by_ground_truth=_group_count(db, InspectionImage.ground_truth_label),
        uploads_per_day=[{"date": d, "count": c} for d, c in per_day.items()],
        recent_batches=[batch_out(b) for b in recent_batches],
    )
