import io
import os
from dataclasses import dataclass
from typing import Optional, Tuple
from PIL import Image, UnidentifiedImageError

from fastapi import HTTPException, UploadFile, status
from app.core.config import settings


@dataclass
class ValidatedImageMetadata:
    width: int
    height: int
    format: str
    mime_type: str
    file_size_bytes: int
    mode: str
    original_filename: str


class ImageValidationError(HTTPException):
    def __init__(self, detail: str):
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=detail,
        )


async def validate_and_read_image(
    file: UploadFile,
    max_size_bytes: Optional[int] = None,
) -> Tuple[bytes, ValidatedImageMetadata]:
    """
    Validates uploaded image file:
    1. Checks original filename and extension.
    2. Validates maximum file size.
    3. Reads raw bytes and verifies actual image integrity using Pillow (verifies headers + decodes).
    4. Rejects corrupt or spoofed files.
    5. Returns (raw_bytes, metadata).
    """
    filename = file.filename or "unknown.jpg"
    _, ext = os.path.splitext(filename)
    ext = ext.lower()

    # 1. Extension check
    if ext not in settings.ALLOWED_IMAGE_EXTENSIONS:
        raise ImageValidationError(
            f"Unsupported file extension '{ext}'. "
            f"Allowed extensions: {', '.join(settings.ALLOWED_IMAGE_EXTENSIONS)}"
        )

    # 2. Read bytes and check size
    max_bytes = max_size_bytes or settings.max_upload_size_bytes
    content = await file.read()
    file_size = len(content)

    if file_size == 0:
        raise ImageValidationError("Uploaded image file is empty (0 bytes).")

    if file_size > max_bytes:
        max_mb = max_bytes / (1024 * 1024)
        curr_mb = file_size / (1024 * 1024)
        raise ImageValidationError(
            f"File size exceeds maximum limit of {max_mb:.1f} MB (Received: {curr_mb:.2f} MB)."
        )

    # 3. Pillow validation & integrity check
    try:
        # First pass: verify integrity
        with Image.open(io.BytesIO(content)) as img:
            img.verify()

        # Second pass: re-open to safely extract properties (verify() invalidates image object)
        with Image.open(io.BytesIO(content)) as img:
            width, height = img.size
            img_format = (img.format or "").upper()
            mode = img.mode

            if width <= 0 or height <= 0:
                raise ImageValidationError(
                    f"Invalid image dimensions: {width}x{height}."
                )

            # Map format to canonical MIME type
            format_to_mime = {
                "JPEG": "image/jpeg",
                "PNG": "image/png",
                "BMP": "image/bmp",
                "TIFF": "image/tiff",
                "WEBP": "image/webp",
            }
            mime_type = format_to_mime.get(img_format, file.content_type or "image/jpeg")

            # Validate detected format against allowed list
            if mime_type not in settings.ALLOWED_IMAGE_MIME_TYPES:
                raise ImageValidationError(
                    f"Image format '{img_format}' is not an authorized type. "
                    f"Allowed types: JPEG, PNG, BMP, TIFF, WebP."
                )

            metadata = ValidatedImageMetadata(
                width=width,
                height=height,
                format=img_format,
                mime_type=mime_type,
                file_size_bytes=file_size,
                mode=mode,
                original_filename=filename,
            )

    except UnidentifiedImageError:
        raise ImageValidationError(
            f"The file '{filename}' could not be decoded as a valid image. "
            f"The file may be corrupt or not a recognized image format."
        )
    except Exception as e:
        if isinstance(e, ImageValidationError):
            raise e
        raise ImageValidationError(
            f"Corrupt or unreadable image file '{filename}': {str(e)}"
        )

    return content, metadata
