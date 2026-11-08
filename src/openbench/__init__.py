"""OpenBench AI: A modular, open-source LLM evaluation and benchmarking toolkit."""

from openbench.adapters import (
    BaseModelAdapter,
    DeterministicLookupAdapter,
    HuggingFaceAdapter,
    LocalCallableAdapter,
)
from openbench.catalog import DatasetCatalog, DatasetCatalogEntry
from openbench.comparison import (
    ComparisonResult,
    MetricLeader,
    ModelComparator,
    ModelComparisonEntry,
)
from openbench.configuration import EvaluationProfile, ProfileLoader
from openbench.dataset import Dataset, DatasetLoader, DatasetRecord
from openbench.evaluator import (
    EvaluationConfig,
    EvaluationResult,
    Evaluator,
    ItemEvaluationResult,
    ModelResponseSet,
    ResponseItem,
    ResponseLoader,
)
from openbench.exceptions import (
    ConfigurationError,
    DatasetValidationError,
    EmptyDatasetError,
    MetricError,
    OpenBenchError,
    ResponseValidationError,
)
from openbench.history import HistoryEntry, ResultHistoryStore
from openbench.reporting import ReportGenerator

__version__ = "0.2.0"

__all__ = [
    "__version__",
    "BaseModelAdapter",
    "ComparisonResult",
    "ConfigurationError",
    "Dataset",
    "DatasetCatalog",
    "DatasetCatalogEntry",
    "DatasetLoader",
    "DatasetRecord",
    "DatasetValidationError",
    "DeterministicLookupAdapter",
    "EmptyDatasetError",
    "EvaluationConfig",
    "EvaluationProfile",
    "EvaluationResult",
    "Evaluator",
    "HistoryEntry",
    "HuggingFaceAdapter",
    "ItemEvaluationResult",
    "LocalCallableAdapter",
    "MetricError",
    "MetricLeader",
    "ModelComparator",
    "ModelComparisonEntry",
    "ModelResponseSet",
    "OpenBenchError",
    "ProfileLoader",
    "ReportGenerator",
    "ResponseItem",
    "ResponseLoader",
    "ResponseValidationError",
    "ResultHistoryStore",
]
