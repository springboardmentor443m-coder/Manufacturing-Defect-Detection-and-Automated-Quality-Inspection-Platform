"""Dataset Licensing and Acquisition Manager for VisionInspect AI.

Enforces the licensing gate for the MVTec Anomaly Detection (MVTec AD) dataset
under Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0),
and manages safe, verified local archive extraction once authorized.
"""

from __future__ import annotations

import hashlib
import logging
import os
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DatasetMetadata:
    """Canonical metadata for the MVTec Anomaly Detection benchmark."""

    name: str = "MVTec Anomaly Detection Dataset (MVTec AD)"
    version: str = "2019"
    official_url: str = "https://www.mvtec.com/research-teaching/datasets/mvtec-ad"
    license_type: str = "CC BY-NC-SA 4.0 (Creative Commons Non-Commercial Share-Alike)"
    commercial_use_allowed: bool = False
    citation: str = (
        "Bergmann, P., Fauser, M., Sattlegger, D., & Steger, C. (2019). "
        "MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection. "
        "In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)."
    )
    initial_categories: tuple[str, ...] = ("bottle", "metal_nut", "tile")


class LicensingGate:
    """Enforces compliance checks before dataset acquisition or downloading."""

    ENV_APPROVAL_VAR = "VISIONINSPECT_MVTEC_LEGAL_APPROVED"

    @classmethod
    def is_approved(cls, explicit_confirmation: Optional[str] = None) -> bool:
        """Determines whether company/legal authorization has been explicitly confirmed.

        Args:
            explicit_confirmation: Optional string passed directly from CLI or user prompt.

        Returns:
            True if approval is unambiguously confirmed, False otherwise.
        """
        if explicit_confirmation and explicit_confirmation.strip().upper() in ("YES", "TRUE", "CONFIRMED"):
            return True

        env_val = os.getenv(cls.ENV_APPROVAL_VAR, "").strip().upper()
        return env_val in ("YES", "TRUE", "1", "CONFIRMED")

    @classmethod
    def generate_acquisition_checklist(cls) -> str:
        """Generates a structured checklist for securing dataset acquisition clearance."""
        meta = DatasetMetadata()
        return f"""# MVTec AD Dataset Acquisition & Legal Compliance Checklist

## 1. Dataset Information
- **Dataset:** {meta.name}
- **Version:** {meta.version}
- **Official Source:** {meta.official_url}
- **License:** {meta.license_type}
- **Commercial Usage:** Strictly Prohibited without separate commercial license from MVTec Software GmbH.

## 2. Mandatory Approval Prerequisite
Because this project is associated with an internship at a company, automatic downloading
or commercial deployment is gated to protect against copyright and licensing violations.

Before acquiring or utilizing the dataset, verify:
- [ ] 1. Formal confirmation from company/internship supervisor that the intended usage is non-commercial and educational/research-only.
- [ ] 2. Legal clearance confirming no proprietary company data or commercial IP is co-mingled with CC BY-NC-SA 4.0 licensed assets.
- [ ] 3. Acknowledgment that models trained on MVTec AD cannot be packaged into commercial product deliverables without explicit commercial licensing.
- [ ] 4. Setting the environment variable to confirm authorization:
       ```bash
       # On Linux/macOS:
       export {cls.ENV_APPROVAL_VAR}=TRUE
       # On Windows PowerShell:
       $env:{cls.ENV_APPROVAL_VAR}="TRUE"
       ```

## 3. Manual Acquisition Procedure (Once Approved)
1. Download category archives directly from the official portal:
   {meta.official_url}
   Target files:
   - `bottle.tar.xz`
   - `metal_nut.tar.xz`
   - `tile.tar.xz`
2. Place archives in `data/raw/` or extract them directly to:
   `data/raw/mvtec_ad/bottle/`
   `data/raw/mvtec_ad/metal_nut/`
   `data/raw/mvtec_ad/tile/`
3. Verify directory layout using the VisionInspect AI dataset validator:
   ```bash
   python -m src.dataset.cli validate --dataset-dir data/raw/mvtec_ad
   ```
"""


class DatasetAcquisitionManager:
    """Manages archive integrity verification and safe extraction."""

    def __init__(self, raw_data_dir: Path) -> None:
        self.raw_data_dir = raw_data_dir

    @staticmethod
    def calculate_sha256(file_path: Path, chunk_size: int = 1024 * 1024) -> str:
        """Calculates the streaming SHA-256 hash of a file."""
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(chunk_size):
                hasher.update(chunk)
        return hasher.hexdigest()

    def verify_archive(self, archive_path: Path, expected_sha256: Optional[str] = None) -> bool:
        """Verifies archive existence and optional cryptographic checksum."""
        if not archive_path.is_file():
            logger.error("Archive not found at %s", archive_path)
            return False

        if expected_sha256:
            actual_sha = self.calculate_sha256(archive_path)
            if actual_sha.lower() != expected_sha256.lower():
                logger.error(
                    "SHA-256 mismatch for %s: expected %s, got %s",
                    archive_path,
                    expected_sha256,
                    actual_sha,
                )
                return False
            logger.info("Checksum verified for %s", archive_path.name)
        return True

    def safe_extract_tar(self, archive_path: Path, target_dir: Path) -> Path:
        """Safely extracts a tar archive preventing directory traversal (Zip Slip)."""
        target_dir.mkdir(parents=True, exist_ok=True)
        resolved_target = target_dir.resolve()

        with tarfile.open(archive_path, "r:*") as tar:
            for member in tar.getmembers():
                dest_path = (resolved_target / member.name).resolve()
                if not str(dest_path).startswith(str(resolved_target)):
                    raise ValueError(f"Security error: path traversal attempt detected in {member.name}")
            tar.extractall(path=resolved_target)

        logger.info("Extracted %s into %s", archive_path.name, resolved_target)
        return resolved_target

    def safe_extract_zip(self, archive_path: Path, target_dir: Path) -> Path:
        """Safely extracts a zip archive preventing directory traversal (Zip Slip)."""
        target_dir.mkdir(parents=True, exist_ok=True)
        resolved_target = target_dir.resolve()

        with zipfile.ZipFile(archive_path, "r") as zf:
            for member in zf.infolist():
                dest_path = (resolved_target / member.filename).resolve()
                if not str(dest_path).startswith(str(resolved_target)):
                    raise ValueError(f"Security error: path traversal attempt detected in {member.filename}")
            zf.extractall(path=resolved_target)

        logger.info("Extracted %s into %s", archive_path.name, resolved_target)
        return resolved_target
