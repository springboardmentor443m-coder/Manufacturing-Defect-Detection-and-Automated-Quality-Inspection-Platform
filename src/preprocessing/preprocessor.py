"""Image preprocessing configuration and processing pipeline for VisionInspect AI.

Provides safe loading, validation, conversion, resizing, and normalization of images
into standardized PyTorch tensors while preserving original dimensions and metadata.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ImagePreprocessConfig:
    """Configuration parameters for image standardization."""

    target_size: Tuple[int, int] = (256, 256)  # (width, height)
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)  # ImageNet mean
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)   # ImageNet std
    min_dim: int = 16
    max_dim: int = 4096


@dataclass
class PreprocessedImage:
    """Container for preprocessed image tensor, original metadata, and dimensions."""

    tensor: torch.Tensor                     # Shape: (3, H, W), float32
    original_size: Tuple[int, int]           # (width, height) in original pixels
    original_mode: str                       # e.g., "RGB", "L", "RGBA"
    target_size: Tuple[int, int]             # (width, height)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ImagePreprocessor:
    """Encapsulates image loading, color-space conversion, resizing, and normalization."""

    def __init__(self, config: Optional[ImagePreprocessConfig] = None) -> None:
        self.config = config or ImagePreprocessConfig()
        self._mean_tensor = torch.tensor(self.config.mean, dtype=torch.float32).view(3, 1, 1)
        self._std_tensor = torch.tensor(self.config.std, dtype=torch.float32).view(3, 1, 1)

    def load_and_validate(self, image_input: Union[Path, str, bytes, Image.Image]) -> Image.Image:
        """Safely loads an image and validates dimensions and decodability."""
        if isinstance(image_input, Image.Image):
            img = image_input
        elif isinstance(image_input, (str, Path)):
            path = Path(image_input)
            if not path.is_file():
                raise FileNotFoundError(f"Image file does not exist: {path}")
            try:
                img = Image.open(path)
                img.load()
            except Exception as e:
                raise ValueError(f"Corrupted or unreadable image file: {path} ({e})") from e
        elif isinstance(image_input, bytes):
            try:
                img = Image.open(io.BytesIO(image_input))
                img.load()
            except Exception as e:
                raise ValueError(f"Corrupted or unreadable image byte stream ({e})") from e
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

        w, h = img.size
        if w < self.config.min_dim or h < self.config.min_dim:
            raise ValueError(
                f"Image dimensions ({w}x{h}) are smaller than minimum allowed ({self.config.min_dim}x{self.config.min_dim})"
            )
        if w > self.config.max_dim or h > self.config.max_dim:
            raise ValueError(
                f"Image dimensions ({w}x{h}) exceed maximum allowed ({self.config.max_dim}x{self.config.max_dim})"
            )

        return img

    @staticmethod
    def convert_to_rgb(image: Image.Image) -> Image.Image:
        """Converts arbitrary color-space formats (L, RGBA, P, CMYK) to standard 3-channel RGB."""
        if image.mode == "RGB":
            return image
        if image.mode == "RGBA":
            # Composite against a neutral black background to preserve dark contrast
            bg = Image.new("RGB", image.size, (0, 0, 0))
            bg.paste(image, mask=image.split()[3])
            return bg
        return image.convert("RGB")

    def resize(self, image: Image.Image, target_size: Optional[Tuple[int, int]] = None) -> Image.Image:
        """Applies anti-aliased bilinear interpolation to scale image to target resolution."""
        size = target_size or self.config.target_size
        return image.resize(size, resample=Image.Resampling.BILINEAR)

    def normalize_to_tensor(self, image: Image.Image) -> torch.Tensor:
        """Converts PIL RGB image to normalized float32 tensor with ImageNet parameters."""
        arr = np.array(image, dtype=np.float32) / 255.0  # Range: [0.0, 1.0], shape: (H, W, 3)
        tensor = torch.from_numpy(arr).permute(2, 0, 1)   # Shape: (3, H, W)
        normalized = (tensor - self._mean_tensor) / self._std_tensor
        return normalized

    def preprocess(
        self,
        image_input: Union[Path, str, bytes, Image.Image],
        target_size: Optional[Tuple[int, int]] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> PreprocessedImage:
        """Executes full preprocessing pipeline and retains original spatial metadata."""
        raw_img = self.load_and_validate(image_input)
        orig_size = raw_img.size
        orig_mode = raw_img.mode

        rgb_img = self.convert_to_rgb(raw_img)
        resized_img = self.resize(rgb_img, target_size=target_size)
        tensor = self.normalize_to_tensor(resized_img)

        metadata = dict(extra_metadata or {})
        if isinstance(image_input, (str, Path)):
            metadata["source_path"] = str(image_input).replace("\\", "/")

        return PreprocessedImage(
            tensor=tensor,
            original_size=orig_size,
            original_mode=orig_mode,
            target_size=target_size or self.config.target_size,
            metadata=metadata,
        )
