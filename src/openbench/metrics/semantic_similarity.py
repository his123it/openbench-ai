"""Lexical and Semantic Similarity metrics with pluggable provider architecture (v0.3.1)."""

from __future__ import annotations

import hashlib
import math
import re
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

from openbench.dataset import DatasetRecord
from openbench.exceptions import MetricError
from openbench.metrics.base import AggregateMetricResult, BaseMetric, MetricResult
from openbench.metrics.normalization import NormalizationConfig, normalize_text


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Compute cosine similarity between two numeric vectors of equal length."""
    if len(vec_a) != len(vec_b) or not vec_a:
        raise MetricError("Embedding vectors must be non-empty and have equal dimensionality")
    dot = sum(a * b for a, b in zip(vec_a, vec_b, strict=False))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    sim = dot / (norm_a * norm_b)
    return min(max(sim, -1.0), 1.0)


class EmbeddingProvider(ABC):
    """Abstract adapter interface for text vector providers.

    Methodological Classification (`is_semantic_embedding`):
    - `is_semantic_embedding = False`: Indicates a lexical or surface n-gram hashing
      provider (such as `LocalNgramEmbeddingProvider`) that measures surface character/word
      overlap rather than semantic meaning.
    - `is_semantic_embedding = True`: Indicates a genuine neural/distributional embedding
      model (such as `HuggingFaceEmbeddingProvider` or a verified embedding callable).
    """

    provider_name: str = "base_embedding_provider"
    model_id: str = "unspecified"
    model_version: str = "0.1.0"
    is_semantic_embedding: bool = False

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """Convert a text string into a fixed-length float vector."""

    @property
    def similarity_kind(self) -> str:
        """Return `'semantic_similarity'` only for true embedding providers, else `'lexical_similarity'`."""
        return "semantic_similarity" if self.is_semantic_embedding else "lexical_similarity"

    def metadata(self) -> dict[str, Any]:
        return {
            "provider_name": self.provider_name,
            "model_id": self.model_id,
            "model_version": self.model_version,
            "is_semantic_embedding": self.is_semantic_embedding,
            "similarity_kind": self.similarity_kind,
        }


class LocalNgramEmbeddingProvider(EmbeddingProvider):
    r"""Offline, zero-dependency deterministic lexical feature-hashing provider.

    Projects word unigrams and character 3-grams into a fixed-dimensional vector space
    using SHA-256 feature hashing.

    Important Methodological Note:
    `LocalNgramEmbeddingProvider` performs **lexical comparison** (`is_semantic_embedding = False`,
    `similarity_kind = "lexical_similarity"`). It does NOT understand paraphrasing, negation,
    context, or semantic meaning.
    """

    provider_name = "local_ngram_hash"
    model_id = "sha256-word-char3-v1"
    model_version = "0.3.1"
    is_semantic_embedding = False

    def __init__(self, dimensions: int = 128) -> None:
        if dimensions < 8:
            raise MetricError("dimensions must be at least 8")
        self.dimensions = dimensions

    def embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dimensions
        cleaned = text.strip().lower()
        if not cleaned:
            return vec

        words = re.findall(r"[\w]+", cleaned)
        features: list[tuple[str, float]] = [(f"w:{w}", 2.0) for w in words]
        compact = re.sub(r"\s+", " ", cleaned)
        for i in range(max(len(compact) - 2, 0)):
            features.append((f"c3:{compact[i:i+3]}", 1.0))

        for feat, weight in features:
            digest = hashlib.sha256(feat.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if (digest[4] & 1) == 0 else -1.0
            vec[idx] += sign * weight

        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0.0:
            vec = [v / norm for v in vec]
        return vec


class CallableEmbeddingProvider(EmbeddingProvider):
    """Adapter allowing callers to supply a custom Python function `fn(text) -> list[float]`."""

    provider_name = "user_callable"

    def __init__(
        self,
        embed_fn: Callable[[str], list[float]],
        model_id: str = "custom-embedding-fn",
        model_version: str = "1.0",
        is_semantic_embedding: bool = True,
    ) -> None:
        if not callable(embed_fn):
            raise MetricError("embed_fn must be callable")
        self._embed_fn = embed_fn
        self.model_id = model_id
        self.model_version = model_version
        self.is_semantic_embedding = bool(is_semantic_embedding)

    def embed(self, text: str) -> list[float]:
        vec = self._embed_fn(text)
        if not isinstance(vec, list) or not vec:
            raise MetricError("Custom embedding function must return a non-empty list of floats")
        return [float(x) for x in vec]


class HuggingFaceEmbeddingProvider(EmbeddingProvider):
    """Optional adapter for local Hugging Face `sentence-transformers` neural embedding models.

    Safety & Reproducibility Guarantees:
    - Does NOT import `sentence_transformers` unless instantiated and called.
    - Defaults to `local_files_only=True` so model weights are NEVER automatically
      downloaded from the internet without explicit user opt-in (`allow_download=True`).
    - Marked `is_semantic_embedding = True`.
    """

    provider_name = "huggingface_sentence_transformers"
    is_semantic_embedding = True

    def __init__(
        self,
        model_id: str,
        model_version: str = "main",
        allow_download: bool = False,
        device: str = "cpu",
    ) -> None:
        if not model_id or not model_id.strip():
            raise MetricError("An explicit Hugging Face model_id is required")
        self.model_id = model_id.strip()
        self.model_version = model_version
        self.allow_download = allow_download
        self.device = device
        self._model: Any = None

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]
        except ImportError as exc:
            raise MetricError(
                "Optional dependency 'sentence-transformers' is not installed. "
                "Install it with `pip install -e '.[semantic]'` or supply a local EmbeddingProvider."
            ) from exc

        self._model = SentenceTransformer(
            self.model_id,
            revision=self.model_version,
            device=self.device,
            local_files_only=not self.allow_download,
        )
        return self._model

    def embed(self, text: str) -> list[float]:
        model = self._load_model()
        encoded = model.encode(text, convert_to_numpy=True)
        return [float(x) for x in encoded.tolist()]


class SemanticSimilarityMetric(BaseMetric):
    """Computes vector cosine similarity between normalized response and expected answer.

    Methodological Transparency & Naming (v0.3.1):
    - When configured with the default `LocalNgramEmbeddingProvider`, this metric performs
      surface word/character n-gram feature hashing (`is_semantic_embedding=False`). Results
      are explicitly labeled with `similarity_type = "lexical_similarity"` in item and
      aggregate details, and `semantic_similarity` serves as a backward-compatible legacy alias.
    - Only when configured with a genuine embedding provider (`is_semantic_embedding=True`,
      such as `HuggingFaceEmbeddingProvider` or a verified embedding `CallableEmbeddingProvider`)
      are results labeled `similarity_type = "semantic_similarity"`.
    - If `require_semantic_provider=True` is passed, attempting to evaluate with a lexical
      provider raises `MetricError`.
    - Neither lexical similarity nor embedding cosine similarity is equivalent to factual correctness.
    """

    name = "semantic_similarity"
    version = "0.3.1"
    score_kind = "bounded_continuous"
    description = (
        "Computes vector cosine similarity between the normalized expected answer and model response. "
        "Uses lexical n-gram hashing by default (labeled as 'lexical_similarity') unless an embedding "
        "provider (e.g., sentence-transformers) is configured."
    )
    limitation_note = (
        "The default local_ngram_hash provider measures lexical surface overlap (lexical_similarity) "
        "and does NOT understand paraphrasing, negation, context, or meaning. Semantic similarity is "
        "NOT factual correctness and is NOT proof of factual correctness."
    )
    is_optional = True
    optional_dependencies = ("sentence-transformers",)

    def __init__(
        self,
        normalization: NormalizationConfig | None = None,
        provider: EmbeddingProvider | None = None,
        similarity_threshold: float = 0.75,
        require_semantic_provider: bool = False,
    ) -> None:
        if not -1.0 <= similarity_threshold <= 1.0:
            raise MetricError("similarity_threshold must be between -1.0 and 1.0")
        self.normalization = normalization or NormalizationConfig()
        self.provider = provider or LocalNgramEmbeddingProvider()
        self.similarity_threshold = similarity_threshold
        self.require_semantic_provider = require_semantic_provider

        if self.require_semantic_provider and not self.provider.is_semantic_embedding:
            raise MetricError(
                f"Provider '{self.provider.provider_name}' performs lexical comparison "
                "(is_semantic_embedding=False). Configure a genuine embedding provider "
                "(e.g., HuggingFaceEmbeddingProvider) when require_semantic_provider=True."
            )

    @property
    def similarity_type(self) -> str:
        """Return `'semantic_similarity'` only if the active provider is a true embedding model, else `'lexical_similarity'`."""
        return self.provider.similarity_kind

    def compute_item(self, record: DatasetRecord, response_text: str) -> MetricResult:
        if self.require_semantic_provider and not self.provider.is_semantic_embedding:
            raise MetricError(
                f"Provider '{self.provider.provider_name}' is not a semantic embedding provider."
            )

        norm_expected = normalize_text(record.expected_answer, self.normalization)
        norm_response = normalize_text(response_text, self.normalization)

        if not norm_expected and not norm_response:
            sim = 1.0
        elif not norm_expected or not norm_response:
            sim = 0.0
        elif norm_expected == norm_response:
            sim = 1.0
        else:
            vec_exp = self.provider.embed(norm_expected)
            vec_resp = self.provider.embed(norm_response)
            raw_cos = cosine_similarity(vec_exp, vec_resp)
            sim = max(0.0, raw_cos)

        passed = sim >= self.similarity_threshold
        return MetricResult(
            metric_name=self.name,
            score=sim,
            passed=passed,
            details={
                "similarity_type": self.similarity_type,
                "is_semantic_embedding": self.provider.is_semantic_embedding,
                "cosine_similarity": round(sim, 6),
                "similarity_threshold": self.similarity_threshold,
                "provider": self.provider.metadata(),
                "limitation": self.limitation_note,
            },
        )

    def aggregate(
        self,
        item_results: list[MetricResult],
        total_dataset_items: int,
    ) -> AggregateMetricResult:
        base_agg = super().aggregate(item_results, total_dataset_items)
        details = dict(base_agg.details)
        details.update(
            {
                "similarity_type": self.similarity_type,
                "is_semantic_embedding": self.provider.is_semantic_embedding,
                "provider": self.provider.metadata(),
                "legacy_alias_note": (
                    "Metric key 'semantic_similarity' is retained as a legacy alias when using "
                    "the default local_ngram_hash provider; actual measurement is 'lexical_similarity'."
                    if not self.provider.is_semantic_embedding
                    else "Evaluated using a semantic embedding provider."
                ),
            }
        )
        return AggregateMetricResult(
            metric_name=base_agg.metric_name,
            metric_version=base_agg.metric_version,
            score=base_agg.score,
            total_items=base_agg.total_items,
            evaluated_items=base_agg.evaluated_items,
            passed_items=base_agg.passed_items,
            details=details,
        )

    def config_summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "similarity_type": self.similarity_type,
            "is_semantic_embedding": self.provider.is_semantic_embedding,
            "similarity_threshold": self.similarity_threshold,
            "require_semantic_provider": self.require_semantic_provider,
            "provider": self.provider.metadata(),
            "normalization": self.normalization.to_dict(),
        }


class LexicalSimilarityMetric(SemanticSimilarityMetric):
    """Explicitly named `lexical_similarity` metric using `LocalNgramEmbeddingProvider`."""

    name = "lexical_similarity"
    version = "0.3.1"
    description = (
        "Computes lexical word/character n-gram cosine similarity using deterministic SHA-256 feature hashing."
    )
    limitation_note = (
        "Measures surface lexical overlap only; does not understand paraphrasing, negation, context, or meaning."
    )
    is_optional = False
    optional_dependencies = ()

    def __init__(
        self,
        normalization: NormalizationConfig | None = None,
        similarity_threshold: float = 0.75,
    ) -> None:
        super().__init__(
            normalization=normalization,
            provider=LocalNgramEmbeddingProvider(),
            similarity_threshold=similarity_threshold,
            require_semantic_provider=False,
        )
