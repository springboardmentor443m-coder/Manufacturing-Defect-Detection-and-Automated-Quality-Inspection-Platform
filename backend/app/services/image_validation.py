"""Image validation + basic quality metrics (Image Acquisition module).

Rejected  → the image cannot be inspected (not an image, too small, blank).
Warning   → it can be inspected but the result may be unreliable (blurry, too dark/bright).
"""

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from app.core.config import get_settings
from app.models import ValidationStatus

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
SHARPNESS_SIDE = 512  # longest side the sharpness metric is measured at


@dataclass
class ValidationResult:
    status: ValidationStatus
    messages: list[str] = field(default_factory=list)
    width: int = 0
    height: int = 0
    channels: int = 0
    blur_score: float = 0.0
    brightness: float = 0.0
    contrast: float = 0.0


def validate_image(data: bytes, filename: str) -> ValidationResult:
    settings = get_settings()
    errors: list[str] = []
    warnings: list[str] = []

    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        errors.append(f"Unsupported file type '{ext or 'none'}'")
    if len(data) > settings.max_upload_mb * 1024 * 1024:
        errors.append(f"File larger than {settings.max_upload_mb} MB")
    if errors:
        return ValidationResult(ValidationStatus.rejected, errors)

    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        return ValidationResult(ValidationStatus.rejected, ["File is corrupt or not a readable image"])

    height, width = img.shape[:2]
    channels = 1 if img.ndim == 2 else img.shape[2]
    if img.dtype != np.uint8:  # e.g. 16-bit TIFF
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    gray = img if channels == 1 else cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY)

    # Measure sharpness at a fixed scale so the score does not depend on resolution.
    scale = SHARPNESS_SIDE / max(height, width)
    probe = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else gray
    blur_score = float(cv2.Laplacian(probe, cv2.CV_64F).var())
    brightness = float(gray.mean())
    contrast = float(gray.std())

    if min(width, height) < settings.min_image_side:
        errors.append(f"Resolution {width}x{height} below minimum {settings.min_image_side}px")
    if contrast < 2.0:
        errors.append("Image is blank (no visible content)")
    if blur_score < settings.blur_threshold:
        warnings.append(f"Image may be blurry (sharpness {blur_score:.1f})")
    if brightness < 30:
        warnings.append("Image is very dark")
    elif brightness > 225:
        warnings.append("Image is overexposed")

    if errors:
        status = ValidationStatus.rejected
    elif warnings:
        status = ValidationStatus.warning
    else:
        status = ValidationStatus.valid

    return ValidationResult(
        status=status,
        messages=errors + warnings,
        width=width,
        height=height,
        channels=channels,
        blur_score=round(blur_score, 2),
        brightness=round(brightness, 2),
        contrast=round(contrast, 2),
    )
