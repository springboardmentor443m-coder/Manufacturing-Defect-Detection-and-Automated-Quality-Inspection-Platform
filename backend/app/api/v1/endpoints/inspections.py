import math
from pathlib import Path
from typing import Any, List, Optional
import cv2
import numpy as np
from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    UploadFile,
    status,
)
from sqlalchemy import func
from sqlalchemy.orm import Session, joinedload

from app.api.deps import get_current_active_user, get_db, require_roles
from app.models.inspection import Inspection, InspectionStatus
from app.models.product import Product
from app.models.user import User
from app.processing import (
    PreprocessingConfig,
    assess_image_quality,
    extract_image_features,
    run_preprocessing_pipeline,
)
from app.schemas.inspection import (
    BatchUploadResponse,
    InspectionCreate,
    InspectionListResponse,
    InspectionOut,
    InspectionUpdate,
    PreprocessingConfigSchema,
    PreprocessResponse,
    QualityReportResponse,
    UploadErrorItem,
)
from app.services.image_validator import ImageValidationError, validate_and_read_image
from app.services.storage import BaseStorageService, get_storage_service

router = APIRouter()


def _enrich_inspection_out(
    inspection: Inspection, storage_service: BaseStorageService
) -> InspectionOut:
    """Helper to convert Inspection ORM model to InspectionOut schema with accessible image URLs."""
    out = InspectionOut.model_validate(inspection)
    out.image_url = storage_service.get_url(inspection.image_path)
    if inspection.processed_image_path:
        out.processed_image_url = storage_service.get_url(inspection.processed_image_path)
    return out


def _analyze_image_quality_safe(file_bytes: bytes) -> tuple[Optional[float], Optional[str], Optional[dict]]:
    """Helper to compute quality and features on image bytes safely."""
    try:
        nparr = np.frombuffer(file_bytes, np.uint8)
        cv_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if cv_img is not None:
            quality_res = assess_image_quality(cv_img)
            features_res = extract_image_features(cv_img)
            quality_res["features"] = features_res
            return quality_res["overall_score"], quality_res["verdict"], quality_res
    except Exception:
        pass
    return None, None, None


@router.post(
    "/upload",
    response_model=InspectionOut,
    status_code=status.HTTP_201_CREATED,
    summary="Upload single inspection image",
)
async def upload_inspection_image(
    file: UploadFile = File(..., description="Inspection image file (JPG, PNG, BMP, TIFF, WebP)"),
    product_id: int = Form(..., description="Target manufactured product ID"),
    db: Session = Depends(get_db),
    storage: BaseStorageService = Depends(get_storage_service),
    current_user: User = Depends(
        require_roles("quality_engineer", "supervisor", "admin")
    ),
) -> Any:
    """
    Validate, store, and create an inspection record from a single uploaded image.
    Validates file format, magic bytes, dimensions, and corruption with Pillow.
    Automatically evaluates optical quality metrics upon ingest.
    """
    # 1. Verify target product exists
    product = db.query(Product).filter(Product.id == product_id).first()
    if not product:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Product with ID {product_id} not found in catalog.",
        )

    # 2. Validate image content & dimensions
    file_bytes, meta = await validate_and_read_image(file)

    # 3. Store file through storage abstraction
    saved_path = storage.save(
        file_bytes=file_bytes,
        original_filename=meta.original_filename,
        content_type=meta.mime_type,
        subfolder=f"products/{product.sku}",
    )

    # 4. Assess optical quality & extract features
    q_score, q_verdict, q_report = _analyze_image_quality_safe(file_bytes)

    # 5. Create inspection database record with metadata
    inspection = Inspection(
        product_id=product.id,
        image_path=saved_path,
        original_filename=meta.original_filename,
        file_size_bytes=meta.file_size_bytes,
        image_width=meta.width,
        image_height=meta.height,
        mime_type=meta.mime_type,
        extra_metadata={
            "format": meta.format,
            "mode": meta.mode,
            "product_sku": product.sku,
            "product_name": product.name,
        },
        quality_score=q_score,
        quality_verdict=q_verdict,
        quality_report=q_report,
        status=InspectionStatus.PENDING.value,
        decision=None,
        overall_severity=0.0,
        created_by=current_user.id,
    )

    db.add(inspection)
    db.commit()
    db.refresh(inspection)

    return _enrich_inspection_out(inspection, storage)


@router.post(
    "/upload/batch",
    response_model=BatchUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload multiple inspection images in batch",
)
async def upload_inspection_images_batch(
    files: List[UploadFile] = File(..., description="List of image files to upload"),
    product_id: int = Form(..., description="Target manufactured product ID"),
    db: Session = Depends(get_db),
    storage: BaseStorageService = Depends(get_storage_service),
    current_user: User = Depends(
        require_roles("quality_engineer", "supervisor", "admin")
    ),
) -> Any:
    """
    Process multiple image uploads in a single batch request.
    Validates each image independently; invalid files produce detailed error reports without failing the entire batch.
    """
    product = db.query(Product).filter(Product.id == product_id).first()
    if not product:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Product with ID {product_id} not found.",
        )

    successful_inspections: List[InspectionOut] = []
    error_items: List[UploadErrorItem] = []

    for file in files:
        filename = file.filename or "unknown.jpg"
        try:
            file_bytes, meta = await validate_and_read_image(file)
            saved_path = storage.save(
                file_bytes=file_bytes,
                original_filename=meta.original_filename,
                content_type=meta.mime_type,
                subfolder=f"products/{product.sku}",
            )

            q_score, q_verdict, q_report = _analyze_image_quality_safe(file_bytes)

            inspection = Inspection(
                product_id=product.id,
                image_path=saved_path,
                original_filename=meta.original_filename,
                file_size_bytes=meta.file_size_bytes,
                image_width=meta.width,
                image_height=meta.height,
                mime_type=meta.mime_type,
                extra_metadata={
                    "format": meta.format,
                    "mode": meta.mode,
                    "product_sku": product.sku,
                    "product_name": product.name,
                },
                quality_score=q_score,
                quality_verdict=q_verdict,
                quality_report=q_report,
                status=InspectionStatus.PENDING.value,
                decision=None,
                overall_severity=0.0,
                created_by=current_user.id,
            )
            db.add(inspection)
            db.commit()
            db.refresh(inspection)

            successful_inspections.append(_enrich_inspection_out(inspection, storage))

        except ImageValidationError as ive:
            error_items.append(UploadErrorItem(filename=filename, error=ive.detail))
        except Exception as ex:
            error_items.append(UploadErrorItem(filename=filename, error=str(ex)))

    return BatchUploadResponse(
        total=len(files),
        successful=len(successful_inspections),
        failed=len(error_items),
        inspections=successful_inspections,
        errors=error_items,
    )


@router.post(
    "/{inspection_id}/preprocess",
    response_model=PreprocessResponse,
    summary="Execute image preprocessing pipeline on an inspection",
)
def preprocess_inspection_image(
    inspection_id: int,
    config_in: Optional[PreprocessingConfigSchema] = None,
    db: Session = Depends(get_db),
    storage: BaseStorageService = Depends(get_storage_service),
    current_user: User = Depends(
        require_roles("quality_engineer", "supervisor", "admin")
    ),
) -> Any:
    """
    Run the configurable OpenCV preprocessing pipeline on the inspection image.
    The original image remains untouched; the processed image is saved separately to storage.
    Updates inspection quality report and metadata.
    """
    inspection = (
        db.query(Inspection)
        .options(joinedload(Inspection.product))
        .filter(Inspection.id == inspection_id)
        .first()
    )
    if not inspection:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Inspection record not found.",
        )

    # 1. Load original image bytes from storage
    try:
        raw_bytes = storage.get(inspection.image_path)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to read image from storage: {str(e)}",
        )

    nparr = np.frombuffer(raw_bytes, np.uint8)
    cv_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if cv_img is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Could not decode image for preprocessing.",
        )

    # 2. Build pipeline config
    pipeline_config = PreprocessingConfig(**(config_in.model_dump() if config_in else {}))

    # 3. Run preprocessing pipeline
    processed_img, pipeline_metadata = run_preprocessing_pipeline(cv_img, pipeline_config)

    # 4. Save processed image separately (original is never modified)
    sku = inspection.product.sku if inspection.product else "default"
    success, encoded_png = cv2.imencode(".png", processed_img)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to encode processed image.")

    orig_stem = Path(inspection.original_filename or f"insp_{inspection.id}").stem
    processed_path = storage.save(
        file_bytes=encoded_png.tobytes(),
        original_filename=f"proc_{orig_stem}.png",
        content_type="image/png",
        subfolder=f"products/{sku}/processed",
    )

    # 5. Evaluate quality & extract features on the processed result
    quality_res = assess_image_quality(processed_img)
    features_res = extract_image_features(processed_img)
    quality_res["features"] = features_res

    # 6. Update database record
    inspection.processed_image_path = processed_path
    inspection.quality_score = quality_res["overall_score"]
    inspection.quality_verdict = quality_res["verdict"]
    inspection.quality_report = quality_res
    inspection.preprocessing_config = pipeline_metadata["config"]
    db.commit()
    db.refresh(inspection)

    return PreprocessResponse(
        inspection_id=inspection.id,
        message="Preprocessing pipeline executed successfully.",
        processed_image_url=storage.get_url(processed_path),
        processed_image_path=processed_path,
        quality_verdict=quality_res["verdict"],
        quality_score=quality_res["overall_score"],
        pipeline_metadata=pipeline_metadata,
        quality_report=quality_res,
    )


@router.get(
    "/{inspection_id}/quality-report",
    response_model=QualityReportResponse,
    summary="Get optical image quality analysis and verdict for an inspection",
)
def get_inspection_quality_report(
    inspection_id: int,
    db: Session = Depends(get_db),
    storage: BaseStorageService = Depends(get_storage_service),
    current_user: User = Depends(
        require_roles("quality_engineer", "supervisor", "admin")
    ),
) -> Any:
    """Retrieve full optical quality analysis report, metrics breakdown, and feature descriptors."""
    inspection = db.query(Inspection).filter(Inspection.id == inspection_id).first()
    if not inspection:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Inspection record not found.",
        )

    report = inspection.quality_report
    if not report or "metrics_breakdown" not in report:
        # Compute dynamically if missing
        try:
            target_path = inspection.processed_image_path or inspection.image_path
            raw_bytes = storage.get(target_path)
            nparr = np.frombuffer(raw_bytes, np.uint8)
            cv_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if cv_img is not None:
                report = assess_image_quality(cv_img)
                features_res = extract_image_features(cv_img)
                report["features"] = features_res
                inspection.quality_score = report["overall_score"]
                inspection.quality_verdict = report["verdict"]
                inspection.quality_report = report
                db.commit()
                db.refresh(inspection)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to generate quality report: {e}")

    original_url = storage.get_url(inspection.image_path)
    processed_url = storage.get_url(inspection.processed_image_path) if inspection.processed_image_path else None

    return QualityReportResponse(
        inspection_id=inspection.id,
        verdict=report.get("verdict", "pass"),
        overall_score=report.get("overall_score", 80.0),
        reasons=report.get("reasons", []),
        resolution=report.get("resolution", {
            "width": inspection.image_width or 0,
            "height": inspection.image_height or 0,
            "megapixels": round(((inspection.image_width or 0) * (inspection.image_height or 0)) / 1_000_000.0, 3),
            "aspect_ratio": round(float(inspection.image_width or 1) / float(inspection.image_height or 1), 3),
        }),
        blur_score=report.get("blur_score", 0.0),
        brightness=report.get("brightness", 0.0),
        contrast=report.get("contrast", 0.0),
        noise_estimate=report.get("noise_estimate", 0.0),
        metrics_breakdown=report.get("metrics_breakdown", {}),
        features=report.get("features", {}),
        original_image_url=original_url,
        processed_image_url=processed_url,
    )


@router.get("", response_model=InspectionListResponse, summary="List inspections with pagination and filters")
def list_inspections(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(15, ge=1, le=100, description="Items per page"),
    status_filter: Optional[str] = Query(None, alias="status"),
    decision_filter: Optional[str] = Query(None, alias="decision"),
    quality_filter: Optional[str] = Query(None, alias="quality_verdict"),
    product_id: Optional[int] = Query(None, description="Filter by product ID"),
    search: Optional[str] = Query(None, description="Filter by filename or image path"),
    db: Session = Depends(get_db),
    storage: BaseStorageService = Depends(get_storage_service),
    current_user: User = Depends(
        require_roles("quality_engineer", "supervisor", "admin")
    ),
) -> Any:
    """Retrieve paginated inspection records with filters."""
    query = (
        db.query(Inspection)
        .options(joinedload(Inspection.defects), joinedload(Inspection.reports))
    )

    if status_filter:
        query = query.filter(Inspection.status == status_filter)
    if decision_filter:
        query = query.filter(Inspection.decision == decision_filter)
    if quality_filter:
        query = query.filter(Inspection.quality_verdict == quality_filter)
    if product_id:
        query = query.filter(Inspection.product_id == product_id)
    if search:
        search_pattern = f"%{search.strip()}%"
        query = query.filter(
            (Inspection.original_filename.ilike(search_pattern))
            | (Inspection.image_path.ilike(search_pattern))
        )

    total = query.count()
    offset = (page - 1) * page_size
    inspections = query.order_by(Inspection.created_at.desc()).offset(offset).limit(page_size).all()
    total_pages = math.ceil(total / page_size) if total > 0 else 1

    items = [_enrich_inspection_out(insp, storage) for insp in inspections]

    return InspectionListResponse(
        items=items,
        total=total,
        page=page,
        page_size=page_size,
        total_pages=total_pages,
    )


@router.get("/{inspection_id}", response_model=InspectionOut, summary="Get inspection details")
def get_inspection(
    inspection_id: int,
    db: Session = Depends(get_db),
    storage: BaseStorageService = Depends(get_storage_service),
    current_user: User = Depends(
        require_roles("quality_engineer", "supervisor", "admin")
    ),
) -> Any:
    """Retrieve full details of an inspection including defect analysis and image URL."""
    inspection = (
        db.query(Inspection)
        .options(joinedload(Inspection.defects), joinedload(Inspection.reports))
        .filter(Inspection.id == inspection_id)
        .first()
    )
    if not inspection:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Inspection record not found.",
        )
    return _enrich_inspection_out(inspection, storage)


@router.delete(
    "/{inspection_id}",
    status_code=status.HTTP_200_OK,
    summary="Delete an inspection and its stored image",
)
def delete_inspection(
    inspection_id: int,
    db: Session = Depends(get_db),
    storage: BaseStorageService = Depends(get_storage_service),
    current_user: User = Depends(
        require_roles("supervisor", "admin", "quality_engineer")
    ),
) -> Any:
    """
    Delete an inspection record and remove its original and processed image files from storage.
    Quality engineers can delete their own queued inspections; Supervisors and Admins can delete any.
    """
    inspection = db.query(Inspection).filter(Inspection.id == inspection_id).first()
    if not inspection:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Inspection record not found.",
        )

    # Non-admins and non-supervisors can only delete inspections they created
    if current_user.role == "quality_engineer" and inspection.created_by != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Quality Engineers can only delete inspections they created.",
        )

    # Delete physical image files (both original and processed if present)
    if inspection.image_path:
        storage.delete(inspection.image_path)
    if inspection.processed_image_path:
        storage.delete(inspection.processed_image_path)

    db.delete(inspection)
    db.commit()

    return {
        "message": f"Inspection #{inspection_id} successfully deleted.",
        "id": inspection_id,
    }
