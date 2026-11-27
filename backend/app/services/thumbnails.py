"""Small cached JPEG previews so image grids don't download full-resolution files."""

import hashlib
from pathlib import Path

import cv2

from app.core.config import get_settings

THUMB_SIDE = 320


def thumbnail_for(source: Path, side: int = THUMB_SIDE) -> Path:
    """Return the path of a cached thumbnail for `source`, creating it if needed."""
    stat = source.stat()
    key = hashlib.sha1(f"{source.resolve()}|{stat.st_mtime_ns}|{side}".encode()).hexdigest()
    dest = get_settings().storage_dir / "thumbs" / f"{key}.jpg"
    if dest.exists():
        return dest

    img = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Cannot read image {source}")
    scale = side / max(img.shape[:2])
    if scale < 1:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    dest.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dest), img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return dest
