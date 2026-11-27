from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app.core.config import get_settings
from app.core.deps import can_inspect, can_view
from app.db import get_db
from app.models import Batch, BatchStatus, ImageSource, InspectionImage, User, ValidationStatus
from app.schemas import BatchDetail, BatchOut, ImageOut, UploadResult
from app.services.ingest import ingest_image
from app.services.thumbnails import thumbnail_for

router = APIRouter(tags=["inspection"])

MAX_FILES_PER_UPLOAD = 100


def batch_out(batch: Batch) -> BatchOut:
    counts = {s: 0 for s in ValidationStatus}
    for img in batch.images:
        counts[img.validation_status] += 1
    return BatchOut(
        id=batch.id,
        name=batch.name,
        product_category=batch.product_category,
        source=batch.source,
        status=batch.status,
        created_at=batch.created_at,
        created_by_name=batch.created_by.full_name,
        image_count=len(batch.images),
        valid_count=counts[ValidationStatus.valid],
        warning_count=counts[ValidationStatus.warning],
        rejected_count=counts[ValidationStatus.rejected],
    )


def _load_batch(db: Session, batch_id: int) -> Batch:
    batch = db.scalar(
        select(Batch)
        .where(Batch.id == batch_id)
        .options(selectinload(Batch.images), selectinload(Batch.created_by))
    )
    if batch is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Batch not found")
    return batch


@router.post("/batches/upload", response_model=UploadResult, status_code=status.HTTP_201_CREATED)
async def upload_batch(
    product_category: str = Form(..., min_length=1, max_length=100),
    batch_name: str = Form("", max_length=255),
    files: list[UploadFile] = File(...),
    db: Session = Depends(get_db),
    user: User = Depends(can_inspect),
):
    """Upload one or more product images as a new inspection batch."""
    if len(files) > MAX_FILES_PER_UPLOAD:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"At most {MAX_FILES_PER_UPLOAD} files per upload")

    category = product_category.strip().lower()
    batch = Batch(
        name=batch_name.strip() or f"{category} upload",
        product_category=category,
        source=ImageSource.upload,
        created_by=user,
    )
    db.add(batch)
    db.flush()  # assigns batch.id for the storage folder

    duplicates: list[str] = []
    for f in files:
        data = await f.read()
        image = ingest_image(
            db, batch=batch, data=data, filename=f.filename or "image", source=ImageSource.upload, user=user
        )
        if image is None:
            duplicates.append(f.filename or "image")

    if not batch.images:
        db.rollback()
        raise HTTPException(status.HTTP_409_CONFLICT, "All files were already uploaded before")

    db.commit()
    batch = _load_batch(db, batch.id)
    rejected = sum(1 for i in batch.images if i.validation_status == ValidationStatus.rejected)
    return UploadResult(
        batch=batch_out(batch),
        accepted=len(batch.images) - rejected,
        rejected=rejected,
        duplicates=duplicates,
    )


@router.get("/batches", response_model=list[BatchOut])
def list_batches(
    category: str | None = None,
    source: ImageSource | None = None,
    limit: int = 50,
    db: Session = Depends(get_db),
    _: User = Depends(can_view),
):
    q = select(Batch).options(selectinload(Batch.images), selectinload(Batch.created_by))
    if category:
        q = q.where(Batch.product_category == category.lower())
    if source:
        q = q.where(Batch.source == source)
    batches = db.scalars(q.order_by(Batch.created_at.desc()).limit(min(limit, 200))).all()
    return [batch_out(b) for b in batches]


@router.get("/batches/{batch_id}", response_model=BatchDetail)
def get_batch(batch_id: int, db: Session = Depends(get_db), _: User = Depends(can_view)):
    batch = _load_batch(db, batch_id)
    return BatchDetail(
        **batch_out(batch).model_dump(),
        images=[ImageOut.model_validate(i) for i in sorted(batch.images, key=lambda i: i.id)],
    )


@router.delete("/batches/{batch_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_batch(batch_id: int, db: Session = Depends(get_db), _: User = Depends(can_inspect)):
    batch = _load_batch(db, batch_id)
    storage = get_settings().storage_dir
    paths = [storage / i.stored_path for i in batch.images]
    db.delete(batch)
    db.commit()
    for p in paths:
        p.unlink(missing_ok=True)


@router.patch("/batches/{batch_id}/status", response_model=BatchOut)
def set_batch_status(
    batch_id: int, new_status: BatchStatus, db: Session = Depends(get_db), _: User = Depends(can_inspect)
):
    batch = _load_batch(db, batch_id)
    batch.status = new_status
    db.commit()
    return batch_out(batch)


@router.get("/images/{image_id}/file")
def image_file(image_id: int, db: Session = Depends(get_db), _: User = Depends(can_view)):
    image = db.get(InspectionImage, image_id)
    if image is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Image not found")
    path = get_settings().storage_dir / image.stored_path
    if not path.exists():
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Image file missing from storage")
    return FileResponse(path)


@router.get("/images/{image_id}/thumb")
def image_thumb(image_id: int, db: Session = Depends(get_db), _: User = Depends(can_view)):
    """320px JPEG preview for grids (cached on disk)."""
    image = db.get(InspectionImage, image_id)
    if image is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Image not found")
    path = get_settings().storage_dir / image.stored_path
    try:
        return FileResponse(thumbnail_for(path), media_type="image/jpeg")
    except (FileNotFoundError, ValueError):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Image cannot be previewed")


@router.get("/images/{image_id}/mask")
def image_mask(image_id: int, db: Session = Depends(get_db), _: User = Depends(can_view)):
    """Ground-truth defect mask (only for images imported from MVTec AD)."""
    image = db.get(InspectionImage, image_id)
    if image is None or not image.ground_truth_mask_path:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No ground-truth mask for this image")
    path = get_settings().storage_dir / image.ground_truth_mask_path
    if not path.exists():
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Mask file missing from storage")
    return FileResponse(path)
