"""Synchronized data augmentation engine for VisionInspect AI.

Applies identical spatial transformations to paired image and mask tensors while
restricting color transformations strictly to the image channel.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torchvision.transforms.functional as TF

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AugmentationConfig:
    """Hyperparameters and category constraints for synchronized data augmentation."""

    horizontal_flip_prob: float = 0.5
    vertical_flip_prob: float = 0.5
    rotation_90_prob: float = 0.5
    brightness_jitter: float = 0.1
    contrast_jitter: float = 0.1
    allow_vertical_flip: bool = True     # False for upright objects like 'bottle'
    allow_rotation_90: bool = True       # True for isotropic textures like 'tile'


CATEGORY_AUGMENTATION_RULES: dict[str, dict[str, bool]] = {
    "bottle": {
        "allow_vertical_flip": False,    # Bottles are strictly oriented upright on conveyors
        "allow_rotation_90": False,      # Arbitrary 90-deg rotation violates physical upright posture
    },
    "metal_nut": {
        "allow_vertical_flip": True,     # Nuts are rotationally symmetric with slight facet differences
        "allow_rotation_90": True,
    },
    "tile": {
        "allow_vertical_flip": True,     # Tiles exhibit planar D4 dihedral symmetry
        "allow_rotation_90": True,
    },
}


class SynchronizedAugmentor:
    """Applies strictly synchronized spatial transforms to image and mask tensors."""

    def __init__(
        self,
        config: Optional[AugmentationConfig] = None,
        category: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.config = config or AugmentationConfig()
        self.category = category
        self.rng = random.Random(seed) if seed is not None else random.Random()

        # Apply category-specific physical constraints
        if category and category.lower() in CATEGORY_AUGMENTATION_RULES:
            rules = CATEGORY_AUGMENTATION_RULES[category.lower()]
            self.allow_vertical_flip = rules.get("allow_vertical_flip", self.config.allow_vertical_flip)
            self.allow_rotation_90 = rules.get("allow_rotation_90", self.config.allow_rotation_90)
        else:
            self.allow_vertical_flip = self.config.allow_vertical_flip
            self.allow_rotation_90 = self.config.allow_rotation_90

    def transform(
        self,
        image: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Applies synchronized geometric and image-only color augmentations.

        Args:
            image: Image tensor of shape (3, H, W).
            mask: Optional binary mask tensor of shape (1, H, W) with values in {0.0, 1.0}.

        Returns:
            Tuple of (augmented_image, augmented_mask).
        """
        img_out = image.clone()
        mask_out = mask.clone() if mask is not None else None

        # 1. Synchronized Horizontal Flip
        if self.rng.random() < self.config.horizontal_flip_prob:
            img_out = TF.hflip(img_out)
            if mask_out is not None:
                mask_out = TF.hflip(mask_out)

        # 2. Synchronized Vertical Flip (if permitted by category)
        if self.allow_vertical_flip and (self.rng.random() < self.config.vertical_flip_prob):
            img_out = TF.vflip(img_out)
            if mask_out is not None:
                mask_out = TF.vflip(mask_out)

        # 3. Synchronized 90-Degree Incremental Rotation
        if self.allow_rotation_90 and (self.rng.random() < self.config.rotation_90_prob):
            angle = self.rng.choice([90, 180, 270])
            img_out = TF.rotate(img_out, angle, interpolation=TF.InterpolationMode.BILINEAR)
            if mask_out is not None:
                mask_out = TF.rotate(mask_out, angle, interpolation=TF.InterpolationMode.NEAREST)

        # 4. Color transformations (Image strictly; MASK UNTOUCHED)
        if self.config.brightness_jitter > 0:
            factor = 1.0 + self.rng.uniform(-self.config.brightness_jitter, self.config.brightness_jitter)
            img_out = TF.adjust_brightness(img_out, factor)

        if self.config.contrast_jitter > 0:
            factor = 1.0 + self.rng.uniform(-self.config.contrast_jitter, self.config.contrast_jitter)
            img_out = TF.adjust_contrast(img_out, factor)

        # 5. Guarantee binary label preservation on mask
        if mask_out is not None:
            mask_out = (mask_out > 0.5).to(dtype=torch.float32)

        return img_out, mask_out
