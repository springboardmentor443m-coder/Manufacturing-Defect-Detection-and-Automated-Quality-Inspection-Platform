"""Command-line interface for VisionInspect AI dataset operations.

Provides entry points for licensing gate verification, safe archive extraction,
dataset structure validation, inventory manifest generation, exploratory data analysis,
and deterministic dataset splitting.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

from .acquisition import DatasetAcquisitionManager, LicensingGate
from .eda import DatasetEDAAnalyzer
from .manifest import DatasetManifestBuilder, ManifestRecord
from .splits import DatasetSplitter, SplitConfig, SplitProtocol
from .validator import DatasetValidator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("visioninspect.dataset.cli")


def build_parser() -> argparse.ArgumentParser:
    """Builds argument parser for the CLI tool."""
    parser = argparse.ArgumentParser(
        prog="python -m src.dataset.cli",
        description="VisionInspect AI — Dataset Management & EDA Tooling",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subcommand: check-gate
    gate_parser = subparsers.add_parser(
        "check-gate",
        help="Check legal approval status for MVTec AD dataset acquisition",
    )
    gate_parser.add_argument(
        "--confirm",
        type=str,
        default=None,
        help="Explicit confirmation string (e.g., 'CONFIRMED')",
    )

    # Subcommand: extract
    extract_parser = subparsers.add_parser(
        "extract",
        help="Safely extract authorized category archives into data/raw/mvtec_ad",
    )
    extract_parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory containing the .tar.xz archive files",
    )
    extract_parser.add_argument(
        "--target-dir",
        type=Path,
        default=Path("data/raw/mvtec_ad"),
        help="Destination directory for extracted dataset",
    )
    extract_parser.add_argument(
        "--categories",
        nargs="+",
        default=["bottle", "metal_nut", "tile"],
        help="Categories to extract",
    )
    extract_parser.add_argument(
        "--confirm",
        type=str,
        default=None,
        help="Explicit confirmation string (e.g., 'CONFIRMED')",
    )

    # Subcommand: validate
    val_parser = subparsers.add_parser(
        "validate",
        help="Validate MVTec AD directory structure, images, masks, and integrity",
    )
    val_parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("data/raw/mvtec_ad"),
        help="Root path to MVTec AD dataset directory",
    )
    val_parser.add_argument(
        "--categories",
        nargs="+",
        default=["bottle", "metal_nut", "tile"],
        help="Target categories to inspect",
    )
    val_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/reports"),
        help="Directory to save validation JSON and CSV reports",
    )

    # Subcommand: build-manifest
    manifest_parser = subparsers.add_parser(
        "build-manifest",
        help="Generate preliminary dataset inventory manifest (JSON and CSV)",
    )
    manifest_parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("data/raw/mvtec_ad"),
        help="Root path to MVTec AD dataset directory",
    )
    manifest_parser.add_argument(
        "--categories",
        nargs="+",
        default=["bottle", "metal_nut", "tile"],
        help="Target categories to index",
    )
    manifest_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/reports"),
        help="Directory to save manifest JSON and CSV files",
    )

    # Subcommand: eda
    eda_parser = subparsers.add_parser(
        "eda",
        help="Run quantitative Exploratory Data Analysis and generate visualizations",
    )
    eda_parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("data/raw/mvtec_ad"),
        help="Root path to MVTec AD dataset directory",
    )
    eda_parser.add_argument(
        "--categories",
        nargs="+",
        default=["bottle", "metal_nut", "tile"],
        help="Target categories to analyze",
    )
    eda_parser.add_argument(
        "--reports-dir",
        type=Path,
        default=Path("data/reports"),
        help="Directory to save statistical summary reports",
    )
    eda_parser.add_argument(
        "--vis-dir",
        type=Path,
        default=Path("data/visualizations"),
        help="Directory to save visual overlays and charts",
    )

    # Subcommand: split
    split_parser = subparsers.add_parser(
        "split",
        help="Generate train/val/test splits according to Protocol A or Protocol B",
    )
    split_parser.add_argument(
        "--manifest-file",
        type=Path,
        default=Path("data/reports/dataset_manifest.csv"),
        help="Path to dataset manifest CSV file",
    )
    split_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/splits"),
        help="Directory to save split manifest files",
    )
    split_parser.add_argument(
        "--protocol",
        type=str,
        default="protocol_a_patchcore",
        choices=["protocol_a_patchcore", "protocol_b_supervised_unet"],
        help="Splitting protocol to apply",
    )
    split_parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.80,
        help="Training split ratio",
    )
    split_parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.20,
        help="Validation split ratio",
    )
    split_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic splitting",
    )

    return parser


def main() -> int:
    """CLI execution dispatcher."""
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "check-gate":
        approved = LicensingGate.is_approved(args.confirm)
        if approved:
            logger.info("MVTec AD Dataset Acquisition: LEGAL APPROVAL CONFIRMED.")
            return 0
        else:
            logger.warning("MVTec AD Dataset Acquisition: LEGAL APPROVAL PENDING.")
            print("\n" + LicensingGate.generate_acquisition_checklist())
            return 1

    elif args.command == "extract":
        if not LicensingGate.is_approved(args.confirm):
            logger.error("Extraction aborted: Mandatory CC BY-NC-SA 4.0 legal authorization unconfirmed.")
            print("\n" + LicensingGate.generate_acquisition_checklist())
            return 1

        logger.info("Starting safe extraction for categories: %s", args.categories)
        acq_mgr = DatasetAcquisitionManager(raw_data_dir=args.raw_dir)
        args.target_dir.mkdir(parents=True, exist_ok=True)

        for cat in args.categories:
            archive_path = args.raw_dir / f"{cat}.tar.xz"
            if not archive_path.is_file():
                logger.error("Archive not found: %s", archive_path)
                return 1

            cat_dest = args.target_dir / cat
            if cat_dest.exists() and any(cat_dest.iterdir()):
                logger.info("Category '%s' already extracted at %s. Verifying contents.", cat, cat_dest)
                continue

            logger.info("Extracting %s -> %s", archive_path.name, args.target_dir)
            acq_mgr.safe_extract_tar(archive_path, args.target_dir)
            logger.info("Extracted category '%s' successfully.", cat)

        return 0

    elif args.command == "validate":
        logger.info("Executing dataset validation on: %s", args.dataset_dir)
        validator = DatasetValidator(
            dataset_root=args.dataset_dir,
            target_categories=args.categories,
        )
        report = validator.validate()

        args.output_dir.mkdir(parents=True, exist_ok=True)
        report.save_json(args.output_dir / "dataset_validation_report.json")
        report.save_csv(args.output_dir / "dataset_validation_issues.csv")

        logger.info(
            "Validation Completed: Valid=%s | Images=%d | Masks=%d | Errors=%d | Warnings=%d",
            report.is_valid,
            report.total_images_scanned,
            report.total_masks_scanned,
            report.error_count,
            report.warning_count,
        )
        return 0 if report.is_valid else 1

    elif args.command == "build-manifest":
        logger.info("Building dataset inventory manifest from: %s", args.dataset_dir)
        builder = DatasetManifestBuilder(
            dataset_root=args.dataset_dir,
            target_categories=args.categories,
        )
        records = builder.build_inventory()

        args.output_dir.mkdir(parents=True, exist_ok=True)
        builder.export_json(records, args.output_dir / "dataset_manifest.json")
        builder.export_csv(records, args.output_dir / "dataset_manifest.csv")

        logger.info("Manifest Generation Completed: %d records indexed.", len(records))
        return 0

    elif args.command == "eda":
        logger.info("Executing Exploratory Data Analysis on: %s", args.dataset_dir)
        analyzer = DatasetEDAAnalyzer(
            dataset_root=args.dataset_dir,
            output_reports_dir=args.reports_dir,
            output_vis_dir=args.vis_dir,
        )
        profiles = analyzer.run_eda(categories=args.categories)

        for cat, prof in profiles.items():
            vis_paths = analyzer.generate_visualizations(category=cat, sample_limit=3)
            logger.info(
                "Category '%s': Total=%d (TrainGood=%d, TestGood=%d, TestDefect=%d) | Masks=%d | VisGenerated=%d",
                cat,
                prof.total_images,
                prof.train_normal_count,
                prof.test_normal_count,
                prof.test_defective_count,
                prof.total_masks_count,
                len(vis_paths),
            )
        return 0

    elif args.command == "split":
        logger.info("Loading manifest from: %s", args.manifest_file)
        if not args.manifest_file.is_file():
            logger.error("Manifest file not found: %s", args.manifest_file)
            return 1

        records = []
        with open(args.manifest_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                records.append(
                    ManifestRecord(
                        image_path=row["image_path"],
                        mask_path=row.get("mask_path") or None,
                        category=row["category"],
                        split=row["split"],
                        defect_type=row["defect_type"],
                        is_anomalous=row["is_anomalous"].lower() in ("true", "1"),
                        image_width=int(row["image_width"]),
                        image_height=int(row["image_height"]),
                        image_format=row["image_format"],
                        file_size_bytes=int(row["file_size_bytes"]),
                        image_sha256=row["image_sha256"],
                        validation_status=row.get("validation_status", "VALID"),
                    )
                )

        protocol_enum = (
            SplitProtocol.PROTOCOL_A_PATCHCORE
            if args.protocol == "protocol_a_patchcore"
            else SplitProtocol.PROTOCOL_B_SUPERVISED_UNET
        )
        test_ratio = 0.0 if protocol_enum == SplitProtocol.PROTOCOL_A_PATCHCORE else round(1.0 - args.train_ratio - args.val_ratio, 4)

        config = SplitConfig(
            protocol=protocol_enum,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=test_ratio,
            seed=args.seed,
        )

        splitter = DatasetSplitter(config=config)
        split_result = splitter.split(records)
        split_result.verify_no_leakage()

        protocol_subdir = args.output_dir / args.protocol
        protocol_subdir.mkdir(parents=True, exist_ok=True)
        split_result.export_manifests(protocol_subdir)
        logger.info("Splitting completed successfully under protocol: %s", args.protocol)
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
