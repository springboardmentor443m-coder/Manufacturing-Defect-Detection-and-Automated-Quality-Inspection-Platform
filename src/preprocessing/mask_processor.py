"""Ground-truth mask processing and defect preservation engine for VisionInspect AI.

Performs binary mask validation, nearest-neighbor resizing, defect area quantification,
small defect disappearance detection, and implicit empty mask synthesis for normal images.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MaskPreprocessConfig:
    """Configuration parameters for mask processing."""

    target_size: Tuple[int, int] = (256, 256)  # (width, height)
    defect_loss_threshold_pct: float = 50.0   # Percentage loss considered significant


@dataclass
class PreprocessedMask:
    """Encapsulates processed binary mask tensor and defect retention metrics."""

    tensor: torch.Tensor                     # Shape: (1, H, W), float32, strictly {0.0, 1.0}
    original_size: Tuple[int, int]           # (width, height)
    target_size: Tuple[int, int]             # (width, height)
    original_foreground_pixels: int
    resized_foreground_pixels: int
    original_defect_ratio_pct: float
    resized_defect_ratio_pct: float
    defect_disappeared: bool                 # True if defect vanished after resizing
    significant_defect_loss: bool            # True if area dropped by > threshold
    is_empty_normal: bool                    # True for normal/good samples


class MaskProcessor:
    """Validates, resizes, and monitors defect preservation in binary segmentation masks."""

    def __init__(self, config: Optional[MaskPreprocessConfig] = None) -> None:
        self.config = config or MaskPreprocessConfig()

    def create_empty_mask(
        self,
        target_size: Optional[Tuple[int, int]] = None,
        original_size: Optional[Tuple[int, int]] = None,
    ) -> PreprocessedMask:
        """Synthesizes an empty mask tensor of zeros for normal/good samples."""
        t_size = target_size or self.config.target_size
        o_size = original_size or t_size
        tensor = torch.zeros((1, t_size[1], t_size[0]), dtype=torch.float32)

        return PreprocessedMask(
            tensor=tensor,
            original_size=o_size,
            target_size=t_size,
            original_foreground_pixels=0,
            resized_foreground_pixels=0,
            original_defect_ratio_pct=0.0,
            resized_defect_ratio_pct=0.0,
            defect_disappeared=False,
            significant_defect_loss=False,
            is_empty_normal=True,
        )

    @staticmethod
    def validate_mask_values(mask_arr: np.ndarray) -> None:
        """Validates that mask contains strictly binary values."""
        unique_vals = set(np.unique(mask_arr).tolist())
        valid_subsets = [{0}, {255}, {0, 255}, {1}, {0, 1}]
        if not any(unique_vals.issubset(valid) for valid in valid_subsets):
            raise ValueError(
                f"Invalid non-binary mask pixel values detected: {sorted(list(unique_vals))[:10]}. "
                "Masks must strictly contain binary labels (0 and 255 or 0 and 1)."
            )

    @staticmethod
    def resize_nearest(mask: Image.Image, target_size: Tuple[int, int]) -> Image.Image:
        """Resizes mask using strict nearest-neighbor interpolation to preserve binary values."""
        return mask.resize(target_size, resample=Image.Resampling.NEAREST)

    def process_mask(
        self,
        mask_input: Optional[Union[Path, str, Image.Image, np.ndarray]],
        target_size: Optional[Tuple[int, int]] = None,
        expected_image_size: Optional[Tuple[int, int]] = None,
    ) -> PreprocessedMask:
        """Validates, resizes, and quantifies defect retention for a mask."""
        t_size = target_size or self.config.target_size

        if mask_input is None:
            return self.create_empty_mask(target_size=t_size, original_size=expected_image_size)

        # Load mask
        if isinstance(mask_input, (str, Path)):
            path = Path(mask_input)
            if not path.is_file():
                raise FileNotFoundError(f"Mask file does not exist: {path}")
            try:
                mask_img = Image.open(path).convert("L")
                mask_img.load()
            except Exception as e:
                raise ValueError(f"Corrupted or unreadable mask file: {path} ({e})") from e
        elif isinstance(mask_input, Image.Image):
            mask_img = mask_input.convert("L")
        elif isinstance(mask_input, np.ndarray):
            mask_img = Image.fromarray(mask_input.astype(np.uint8), mode="L")
        else:
            raise TypeError(f"Unsupported mask input type: {type(mask_input)}")

        orig_w, orig_h = mask_img.size

        # Verify dimension alignment with image
        if expected_image_size is not None:
            exp_w, exp_h = expected_image_size
            if (orig_w, orig_h) != (exp_w, exp_h):
                raise ValueError(
                    f"Dimension mismatch between image ({exp_w}x{exp_h}) and mask ({orig_w}x{orig_h})"
                )

        # Validate binary values
        orig_arr = np.array(mask_img)
        self.validate_mask_values(orig_arr)

        # Compute original defect area
        orig_foreground = int(np.count_nonzero(orig_arr > 0))
        orig_total_pixels = orig_w * orig_h
        orig_ratio_pct = (orig_foreground / orig_total_pixels) * 100.0 if orig_total_pixels > 0 else 0.0

        # Resize via nearest-neighbor
        resized_mask_img = self.resize_nearest(mask_img, target_size=t_size)
        resized_arr = np.array(resized_mask_img)

        # Compute resized defect area
        resized_foreground = int(np.count_nonzero(resized_arr > 0))
        resized_total_pixels = t_size[0] * t_size[1]
        resized_ratio_pct = (resized_foreground / resized_total_pixels) * 100.0 if resized_total_pixels > 0 else 0.0

        # Defect preservation checks
        defect_disappeared = (orig_foreground > 0) and (resized_foreground == 0)
        significant_loss = False
        if orig_foreground > 0 and not defect_disappeared:
            expected_scaled_fg = orig_foreground * (resized_total_pixels / orig_total_pixels)
            if expected_scaled_fg > 0:
                loss_pct = max(0.0, (expected_scaled_fg - resized_foreground) / expected_scaled_fg * 100.0)
                if loss_pct > self.config.defect_loss_threshold_pct:
                    significant_loss = True

        if defect_disappeared:
            logger.warning(
                "CRITICAL: Defect region completely disappeared after resizing from %dx%d to %dx%d! "
                "Original defect pixels: %d -> Resized: 0",
                orig_w, orig_h, t_size[0], t_size[1], orig_foreground
            )

        # Convert to tensor: strictly binary float32 in {0.0, 1.0}
        binary_tensor = torch.from_numpy((resized_arr > 0).astype(np.float32)).unsqueeze(0)

        return PreprocessedMask(
            tensor=binary_tensor,
            original_size=(orig_w, orig_h),
            target_size=t_size,
            original_foreground_pixels=orig_foreground,
            resized_foreground_pixels=resized_foreground,
            original_defect_ratio_pct=orig_ratio_pct,
            resized_defect_ratio_pct=resized_ratio_pct,
            defect_disappeared=defect_disappeared,
            significant_defect_loss=significant_loss,
            is_empty_normal=False,
        )
