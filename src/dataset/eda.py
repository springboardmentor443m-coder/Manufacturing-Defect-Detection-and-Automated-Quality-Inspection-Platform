"""Exploratory Data Analysis (EDA) & Visualization Engine for VisionInspect AI.

Computes comprehensive image-level and mask-level statistics and renders visual
inspection artifacts (overlays, histograms, Pareto distributions) for MVTec AD.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class DefectAreaStats:
    """Statistical summary of defect pixel areas and coverage percentages."""

    min_area_pixels: int = 0
    max_area_pixels: int = 0
    mean_area_pixels: float = 0.0
    median_area_pixels: float = 0.0
    p25_area_pct: float = 0.0
    p50_area_pct: float = 0.0
    p75_area_pct: float = 0.0
    p95_area_pct: float = 0.0
    min_area_pct: float = 0.0
    max_area_pct: float = 0.0
    mean_area_pct: float = 0.0


@dataclass
class CategoryEDAProfile:
    """Comprehensive statistical profile for a single manufacturing category."""

    category: str
    total_images: int = 0
    train_normal_count: int = 0
    test_normal_count: int = 0
    test_defective_count: int = 0
    total_masks_count: int = 0
    empty_masks_count: int = 0
    defect_subtypes_breakdown: Dict[str, int] = field(default_factory=dict)
    resolutions_observed: List[Tuple[int, int]] = field(default_factory=list)
    formats_observed: List[str] = field(default_factory=list)
    channels_observed: List[int] = field(default_factory=list)
    file_size_bytes_stats: Dict[str, float] = field(default_factory=dict)
    defect_area_stats: DefectAreaStats = field(default_factory=DefectAreaStats)
    anomalies_detected: List[str] = field(default_factory=list)


class DatasetEDAAnalyzer:
    """Performs deep empirical exploratory data analysis over MVTec AD categories."""

    def __init__(
        self,
        dataset_root: Path,
        output_reports_dir: Optional[Path] = None,
        output_vis_dir: Optional[Path] = None,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.reports_dir = output_reports_dir or Path("data/reports")
        self.vis_dir = output_vis_dir or Path("data/visualizations")

    def run_eda(self, categories: Optional[List[str]] = None) -> Dict[str, CategoryEDAProfile]:
        """Executes full quantitative EDA across selected categories."""
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.vis_dir.mkdir(parents=True, exist_ok=True)

        profiles: Dict[str, CategoryEDAProfile] = {}

        if not self.dataset_root.is_dir():
            logger.warning("Dataset root does not exist at %s. EDA cannot run.", self.dataset_root)
            return profiles

        target_cats = categories or [
            d.name for d in self.dataset_root.iterdir() if d.is_dir() and not d.name.startswith(".")
        ]

        for cat in target_cats:
            cat_dir = self.dataset_root / cat
            if not cat_dir.is_dir():
                continue
            profile = self._analyze_category(cat, cat_dir)
            profiles[cat] = profile

        # Save aggregate JSON report
        report_data = {
            "eda_version": "1.0.0",
            "dataset_root": str(self.dataset_root).replace("\\", "/"),
            "categories": {cat: asdict(prof) for cat, prof in profiles.items()},
        }
        report_file = self.reports_dir / "eda_statistical_summary.json"
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2)
        logger.info("Saved EDA summary report to %s", report_file)

        return profiles

    def _analyze_category(self, category: str, category_dir: Path) -> CategoryEDAProfile:
        """Computes granular image and mask metrics for a given category."""
        profile = CategoryEDAProfile(category=category)

        file_sizes: List[int] = []
        resolutions_set: Set[Tuple[int, int]] = set()
        formats_set: Set[str] = set()
        channels_set: Set[int] = set()
        defect_areas_px: List[int] = []
        defect_areas_pct: List[float] = []

        train_good = category_dir / "train" / "good"
        test_dir = category_dir / "test"
        gt_dir = category_dir / "ground_truth"

        # 1. Analyze train/good images
        if train_good.is_dir():
            for f in sorted(train_good.iterdir()):
                if f.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                    profile.train_normal_count += 1
                    file_sizes.append(f.stat().st_size)
                    try:
                        with Image.open(f) as img:
                            resolutions_set.add(img.size)
                            formats_set.add(img.format or f.suffix[1:].upper())
                            channels_set.add(len(img.getbands()))
                    except Exception as e:
                        profile.anomalies_detected.append(f"Train image corrupt: {f.name} ({e})")

        # 2. Analyze test directory
        if test_dir.is_dir():
            for defect_folder in sorted(test_dir.iterdir()):
                if not defect_folder.is_dir() or defect_folder.name.startswith("."):
                    continue

                defect_name = defect_folder.name
                is_normal = defect_name == "good"

                images_in_folder = [
                    f for f in defect_folder.iterdir() if f.suffix.lower() in {".png", ".jpg", ".jpeg"}
                ]
                count = len(images_in_folder)

                if is_normal:
                    profile.test_normal_count = count
                else:
                    profile.defect_subtypes_breakdown[defect_name] = count
                    profile.test_defective_count += count

                for f in images_in_folder:
                    file_sizes.append(f.stat().st_size)
                    try:
                        with Image.open(f) as img:
                            w, h = img.size
                            resolutions_set.add((w, h))
                            formats_set.add(img.format or f.suffix[1:].upper())
                            channels_set.add(len(img.getbands()))
                            img_area = w * h
                    except Exception as e:
                        profile.anomalies_detected.append(f"Test image corrupt: {f.name} ({e})")
                        continue

                    # Mask analysis for defect images
                    if not is_normal and gt_dir.is_dir():
                        cand_1 = gt_dir / defect_name / f"{f.stem}_mask.png"
                        cand_2 = gt_dir / defect_name / f"{f.stem}.png"
                        mask_path = cand_1 if cand_1.is_file() else (cand_2 if cand_2.is_file() else None)

                        if mask_path:
                            profile.total_masks_count += 1
                            try:
                                with Image.open(mask_path) as m_img:
                                    m_arr = np.array(m_img)
                                    # Defect pixels are non-zero
                                    defective_px = int(np.count_nonzero(m_arr))
                                    defect_areas_px.append(defective_px)
                                    ratio_pct = (defective_px / img_area) * 100.0
                                    defect_areas_pct.append(ratio_pct)
                                    if defective_px == 0:
                                        profile.empty_masks_count += 1
                                        profile.anomalies_detected.append(
                                            f"Empty defect mask with zero defective pixels: {mask_path.name}"
                                        )
                            except Exception as e:
                                profile.anomalies_detected.append(f"Mask unreadable: {mask_path.name} ({e})")
                        else:
                            profile.anomalies_detected.append(
                                f"Missing ground truth mask for defect image {f.name}"
                            )

        profile.total_images = (
            profile.train_normal_count + profile.test_normal_count + profile.test_defective_count
        )
        profile.resolutions_observed = sorted(list(resolutions_set))
        profile.formats_observed = sorted(list(formats_set))
        profile.channels_observed = sorted(list(channels_set))

        # File size stats
        if file_sizes:
            profile.file_size_bytes_stats = {
                "min": float(np.min(file_sizes)),
                "max": float(np.max(file_sizes)),
                "mean": float(np.mean(file_sizes)),
                "std": float(np.std(file_sizes)),
            }

        # Defect area stats
        if defect_areas_px and defect_areas_pct:
            profile.defect_area_stats = DefectAreaStats(
                min_area_pixels=int(np.min(defect_areas_px)),
                max_area_pixels=int(np.max(defect_areas_px)),
                mean_area_pixels=float(np.mean(defect_areas_px)),
                median_area_pixels=float(np.median(defect_areas_px)),
                min_area_pct=float(np.min(defect_areas_pct)),
                max_area_pct=float(np.max(defect_areas_pct)),
                mean_area_pct=float(np.mean(defect_areas_pct)),
                p25_area_pct=float(np.percentile(defect_areas_pct, 25)),
                p50_area_pct=float(np.percentile(defect_areas_pct, 50)),
                p75_area_pct=float(np.percentile(defect_areas_pct, 75)),
                p95_area_pct=float(np.percentile(defect_areas_pct, 95)),
            )

        return profile

    def generate_visualizations(self, category: str, sample_limit: int = 3) -> List[Path]:
        """Generates representative multi-panel inspection visualizations if images are present."""
        generated_paths: List[Path] = []
        cat_dir = self.dataset_root / category
        test_dir = cat_dir / "test"
        gt_dir = cat_dir / "ground_truth"

        if not test_dir.is_dir() or not gt_dir.is_dir():
            return generated_paths

        # Discover defect images with masks
        for defect_folder in sorted(test_dir.iterdir()):
            if not defect_folder.is_dir() or defect_folder.name == "good":
                continue

            defect_name = defect_folder.name
            images = [f for f in defect_folder.iterdir() if f.suffix.lower() in {".png", ".jpg", ".jpeg"}][
                :sample_limit
            ]

            for img_path in images:
                cand_1 = gt_dir / defect_name / f"{img_path.stem}_mask.png"
                cand_2 = gt_dir / defect_name / f"{img_path.stem}.png"
                mask_path = cand_1 if cand_1.is_file() else (cand_2 if cand_2.is_file() else None)

                if mask_path:
                    out_path = self._render_sample_overlay(category, defect_name, img_path, mask_path)
                    if out_path:
                        generated_paths.append(out_path)

        return generated_paths

    def _render_sample_overlay(
        self,
        category: str,
        defect_name: str,
        img_path: Path,
        mask_path: Path,
    ) -> Optional[Path]:
        """Renders side-by-side composite visual inspection panel: Image | Mask | Blend."""
        try:
            with Image.open(img_path) as img, Image.open(mask_path) as mask:
                rgb_img = img.convert("RGB")
                w, h = rgb_img.size

                mask_gray = mask.convert("L")
                mask_arr = np.array(mask_gray)

                # Create red highlight overlay for defective pixels
                overlay_arr = np.array(rgb_img).copy()
                defect_mask = mask_arr > 0
                overlay_arr[defect_mask, 0] = np.clip(overlay_arr[defect_mask, 0] * 0.4 + 150, 0, 255)
                overlay_arr[defect_mask, 1] = np.clip(overlay_arr[defect_mask, 1] * 0.4, 0, 255)
                overlay_arr[defect_mask, 2] = np.clip(overlay_arr[defect_mask, 2] * 0.4, 0, 255)
                blend_img = Image.fromarray(overlay_arr)

                # Composite canvas: 3 panels horizontally
                composite = Image.new("RGB", (w * 3, h))
                composite.paste(rgb_img, (0, 0))
                composite.paste(mask_gray.convert("RGB"), (w, 0))
                composite.paste(blend_img, (w * 2, 0))

                out_filename = f"{category}_{defect_name}_{img_path.stem}_eda_panel.png"
                out_path = self.vis_dir / out_filename
                composite.save(out_path)
                logger.info("Generated EDA visualization: %s", out_path)
                return out_path
        except Exception as e:
            logger.error("Failed to render visualization overlay for %s: %s", img_path, e)
            return None
