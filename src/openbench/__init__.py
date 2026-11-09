"""OpenBench AI: A modular, open-source LLM evaluation, execution, and benchmarking platform."""

from openbench.adapters import (
    PROVIDER_REGISTRY,
    BaseModelAdapter,
    DeterministicLookupAdapter,
    GenerationConfig,
    HuggingFaceAdapter,
    LocalCallableAdapter,
    MockModelProvider,
    OllamaAdapter,
    OllamaConnectionError,
    ProviderHealthStatus,
    create_provider,
    list_registered_providers,
    register_provider,
)
from openbench.analytics import BenchmarkAnalytics
from openbench.cache import CacheRecord, ResponseCache
from openbench.catalog import DatasetCatalog, DatasetCatalogEntry
from openbench.comparison import (
    ComparisonResult,
    MetricLeader,
    ModelComparator,
    ModelComparisonEntry,
)
from openbench.configuration import EvaluationProfile, ProfileLoader
from openbench.dataset import (
    Dataset,
    DatasetLoader,
    DatasetRecord,
    IdFilterSpec,
    load_ids_file,
)
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
    IncompatibleComparisonError,
    MetricError,
    OpenBenchError,
    ResponseValidationError,
)
from openbench.execution import (
    ExecutionConfig,
    ExecutionEngine,
    ExecutionOutcome,
    RetryPolicy,
    classify_execution_error,
)
from openbench.experiments import ExperimentManager, ExperimentRecord, RepeatGroupResult
from openbench.history import HistoryEntry, ResultHistoryStore
from openbench.integrity import (
    IntegrityCheckResult,
    RunIntegrityVerifier,
    compute_dict_sha256,
    compute_file_sha256,
)
from openbench.lineage import RunLineage
from openbench.metrics import LexicalSimilarityMetric
from openbench.paired import (
    ComparisonArm,
    ComparisonPolicy,
    CompatibilityReport,
    PairedComparisonEngine,
    PairedComparisonResult,
)
from openbench.performance import (
    PerformanceMonitor,
    PerformanceReport,
    PricingConfig,
    RequestTelemetry,
    compute_percentile,
)
from openbench.preferences import (
    AnnotationImporter,
    PreferenceAnalyzer,
    PreferenceAnnotation,
    PreferenceTaskBuilder,
)
from openbench.reporting import ReportGenerator, escape_markdown_cell
from openbench.security import (
    redact_prompt_if_private,
    redact_sensitive_data,
    resolve_env_credential,
    sanitize_error_message,
    validate_endpoint_url,
    validate_safe_output_path,
)

__version__ = "0.3.1"

__all__ = [
    "__version__",
    "BaseModelAdapter",
    "BenchmarkAnalytics",
    "CacheRecord",
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
    "ExecutionConfig",
    "ExecutionEngine",
    "ExecutionOutcome",
    "ExperimentManager",
    "ExperimentRecord",
    "GenerationConfig",
    "HistoryEntry",
    "HuggingFaceAdapter",
    "IdFilterSpec",
    "IntegrityCheckResult",
    "ItemEvaluationResult",
    "LexicalSimilarityMetric",
    "LocalCallableAdapter",
    "MetricError",
    "MetricLeader",
    "MockModelProvider",
    "ModelComparator",
    "ModelComparisonEntry",
    "ModelResponseSet",
    "OllamaAdapter",
    "OllamaConnectionError",
    "OpenBenchError",
    "PROVIDER_REGISTRY",
    "PerformanceMonitor",
    "PerformanceReport",
    "PricingConfig",
    "ProfileLoader",
    "ProviderHealthStatus",
    "ReportGenerator",
    "RequestTelemetry",
    "ResponseCache",
    "ResponseItem",
    "ResponseLoader",
    "ResponseValidationError",
    "ResultHistoryStore",
    "RetryPolicy",
    "RunIntegrityVerifier",
    "classify_execution_error",
    "compute_dict_sha256",
    "compute_file_sha256",
    "compute_percentile",
    "create_provider",
    "escape_markdown_cell",
    "list_registered_providers",
    "load_ids_file",
    "redact_prompt_if_private",
    "redact_sensitive_data",
    "register_provider",
    "resolve_env_credential",
    "sanitize_error_message",
    "validate_endpoint_url",
    "validate_safe_output_path",
    # v0.4.0 additions (the 70 v0.3.1 names above are unchanged and keep their order)
    "ComparisonArm",
    "ComparisonPolicy",
    "CompatibilityReport",
    "IncompatibleComparisonError",
    "PairedComparisonEngine",
    "PairedComparisonResult",
    "RepeatGroupResult",
    "RunLineage",
    "AnnotationImporter",
    "PreferenceAnalyzer",
    "PreferenceAnnotation",
    "PreferenceTaskBuilder",
]
