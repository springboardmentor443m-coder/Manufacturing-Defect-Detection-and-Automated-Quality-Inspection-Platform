"""MVTec AD dataset integration and simulated camera capture."""

import random
import shutil
import uuid

import cv2
import numpy as np
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.deps import can_inspect, can_view
from app.db import get_db
from app.models import Batch, BatchStatus, ImageSource, User
from app.routers.batches import _load_batch, batch_out
from app.schemas import (
    BatchOut,
    CameraCaptureRequest,
    DatasetCategory,
    DatasetImportRequest,
    DatasetSummary,
)
from app.services import mvtec
from app.services.ingest import ingest_image
from app.services.thumbnails import thumbnail_for

router = APIRouter(tags=["dataset"])


def _require_category(category: str) -> None:
    if category not in mvtec.list_categories():
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            f"Category '{category}' not found under {mvtec.dataset_root()}",
        )


def _copy_mask(sample: mvtec.DatasetSample, batch_id: int) -> str | None:
    if sample.mask_path is None:
        return None
    rel = f"masks/{batch_id}/{uuid.uuid4().hex}.png"
    dest = get_settings().storage_dir / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(sample.mask_path, dest)
    return rel


@router.get("/dataset", response_model=DatasetSummary)
def dataset_summary(_: User = Depends(can_view)):
    root = mvtec.dataset_root()
    categories = [DatasetCategory(**mvtec.summarize_category(c)) for c in mvtec.list_categories()]
    return DatasetSummary(root=str(root), available=bool(categories), categories=categories)


@router.get("/dataset/{category}/preview")
def dataset_preview(category: str, kind: str = "good", _: User = Depends(can_view)):
    """Thumbnail of one sample from a category: `good` or `defect`."""
    _require_category(category)
    samples = mvtec.list_samples(category, "test")
    pool = [s for s in samples if (s.label == "good") == (kind == "good")] or samples
    return FileResponse(thumbnail_for(pool[0].path), media_type="image/jpeg")


@router.post("/dataset/import", response_model=BatchOut, status_code=status.HTTP_201_CREATED)
def import_dataset(body: DatasetImportRequest, db: Session = Depends(get_db), user: User = Depends(can_inspect)):
    """Load MVTec AD images (with their ground-truth labels) into the platform as a batch."""
    _require_category(body.category)
    samples = mvtec.list_samples(body.category, body.split)
    # Spread the limit across labels so a test import includes every defect type.
    random.Random(42).shuffle(samples)
    samples = sorted(samples[: body.limit], key=lambda s: (s.label, s.path.name))

    batch = Batch(
        name=f"MVTec AD · {body.category} · {body.split}",
        product_category=body.category,
        source=ImageSource.dataset,
        created_by=user,
    )
    db.add(batch)
    db.flush()

    for s in samples:
        ingest_image(
            db,
            batch=batch,
            data=s.path.read_bytes(),
            filename=s.path.name,
            source=ImageSource.dataset,
            user=user,
            ground_truth_label=s.label,
            ground_truth_mask_path=_copy_mask(s, batch.id),
        )

    if not batch.images:
        db.rollback()
        raise HTTPException(status.HTTP_409_CONFLICT, "These images were already imported")
    db.commit()
    return batch_out(_load_batch(db, batch.id))


def _simulate_frame(path, rng: random.Random) -> bytes:
    """Re-shoot a dataset image as a camera would: slight lighting change and sensor noise."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    gain = rng.uniform(0.92, 1.08)
    noise = np.random.default_rng(rng.randrange(2**32)).normal(0, 2.0, img.shape)
    frame = np.clip(img.astype(np.float32) * gain + noise, 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".png", frame)
    if not ok:
        raise RuntimeError("Failed to encode camera frame")
    return buf.tobytes()


@router.post("/camera/capture", response_model=BatchOut, status_code=status.HTTP_201_CREATED)
def camera_capture(body: CameraCaptureRequest, db: Session = Depends(get_db), user: User = Depends(can_inspect)):
    """Simulate a production-line camera capturing `count` parts of a product category."""
    _require_category(body.category)
    rng = random.Random()
    frames = mvtec.pick_camera_frames(body.category, body.count, body.defect_ratio, rng)

    batch = Batch(
        name=f"{body.station} · {body.category}",
        product_category=body.category,
        source=ImageSource.camera,
        status=BatchStatus.processing,
        created_by=user,
    )
    db.add(batch)
    db.flush()

    for i, s in enumerate(frames, 1):
        ingest_image(
            db,
            batch=batch,
            data=_simulate_frame(s.path, rng),
            filename=f"frame_{i:03d}.png",
            source=ImageSource.camera,
            user=user,
            ground_truth_label=s.label,
            ground_truth_mask_path=_copy_mask(s, batch.id),
        )
    db.commit()
    return batch_out(_load_batch(db, batch.id))
