"""VisionInspect AI - Dataset Acquisition, Validation, Splitting, and Datasets Package."""

from .acquisition import DatasetAcquisitionManager, LicensingGate
from .validator import DatasetValidator, ValidationIssue, ValidationReport
from .manifest import DatasetManifestBuilder, ManifestRecord
from .eda import DatasetEDAAnalyzer
from .splits import DatasetSplitter, SplitConfig, SplitProtocol, SplitResult, DataLeakageError, InvalidSplitConfigError
from .torch_datasets import UNetDataset, PatchCoreDataset, create_dataloader

__all__ = [
    "DatasetAcquisitionManager",
    "LicensingGate",
    "DatasetValidator",
    "ValidationIssue",
    "ValidationReport",
    "DatasetManifestBuilder",
    "ManifestRecord",
    "DatasetEDAAnalyzer",
    "DatasetSplitter",
    "SplitConfig",
    "SplitProtocol",
    "SplitResult",
    "DataLeakageError",
    "InvalidSplitConfigError",
    "UNetDataset",
    "PatchCoreDataset",
    "create_dataloader",
]
