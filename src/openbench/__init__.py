"""OpenBench AI: A modular, open-source LLM evaluation and benchmarking toolkit."""

from openbench.comparison import ComparisonResult, ModelComparator
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
from openbench.reporting import ReportGenerator

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "ComparisonResult",
    "ConfigurationError",
    "Dataset",
    "DatasetLoader",
    "DatasetRecord",
    "DatasetValidationError",
    "EmptyDatasetError",
    "EvaluationConfig",
    "EvaluationResult",
    "Evaluator",
    "ItemEvaluationResult",
    "MetricError",
    "ModelComparator",
    "ModelResponseSet",
    "OpenBenchError",
    "ReportGenerator",
    "ResponseItem",
    "ResponseLoader",
    "ResponseValidationError",
]
