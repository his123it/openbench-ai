"""Semantic Similarity metric with pluggable embedding provider architecture."""

from __future__ import annotations

import hashlib
import math
import re
from abc import ABC, abstractmethod
from typing import Any, Callable

from openbench.dataset import DatasetRecord
from openbench.exceptions import MetricError
from openbench.metrics.base import BaseMetric, MetricResult
from openbench.metrics.normalization import NormalizationConfig, normalize_text


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Compute cosine similarity between two numeric vectors of equal length."""
    if len(vec_a) != len(vec_b) or not vec_a:
        raise MetricError("Embedding vectors must be non-empty and have equal dimensionality")
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    sim = dot / (norm_a * norm_b)
    return min(max(sim, -1.0), 1.0)


class EmbeddingProvider(ABC):
    """Abstract adapter interface for text embedding providers."""

    provider_name: str = "base_embedding_provider"
    model_id: str = "unspecified"
    model_version: str = "0.1.0"

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """Convert a text string into a fixed-length float vector."""

    def metadata(self) -> dict[str, Any]:
        return {
            "provider_name": self.provider_name,
            "model_id": self.model_id,
            "model_version": self.model_version,
        }


class LocalNgramEmbeddingProvider(EmbeddingProvider):
    """Offline, zero-dependency deterministic feature-hashing embedding provider.

    Projects word unigrams and character 3-grams into a fixed-dimensional vector space
    using SHA-256 feature hashing. Requires no external libraries, network access, or
    model weight downloads.
    """

    provider_name = "local_ngram_hash"
    model_id = "sha256-word-char3-v1"
    model_version = "0.2.0"

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
    ) -> None:
        if not callable(embed_fn):
            raise MetricError("embed_fn must be callable")
        self._embed_fn = embed_fn
        self.model_id = model_id
        self.model_version = model_version

    def embed(self, text: str) -> list[float]:
        vec = self._embed_fn(text)
        if not isinstance(vec, list) or not vec:
            raise MetricError("Custom embedding function must return a non-empty list of floats")
        return [float(x) for x in vec]


class HuggingFaceEmbeddingProvider(EmbeddingProvider):
    """Optional adapter for local Hugging Face `sentence-transformers` models.

    Safety & Reproducibility Guarantees:
    - Does NOT import `sentence_transformers` unless instantiated.
    - Defaults to `local_files_only=True` so model weights are NEVER automatically
      downloaded from the internet without explicit user opt-in (`allow_download=True`).
    """

    provider_name = "huggingface_sentence_transformers"

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
                "Install it with `pip install -e '\".[semantic]\"'` or supply a local EmbeddingProvider."
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
    """Computes embedding cosine similarity between normalized response and expected answer.

    Critical Methodological Warning:
    Semantic similarity measures vector-space topical/lexical proximity.
    It is NOT factual correctness and must never be represented as proof that an
    answer is factually or logically valid (for instance, 'The patient should take 10 mg'
    and 'The patient should NOT take 100 mg' have very high embedding similarity).
    """

    name = "semantic_similarity"
    version = "0.2.0"
    description = (
        "Computes cosine similarity between embeddings of the normalized expected answer "
        "and model response using a configurable EmbeddingProvider."
    )
    limitation_note = (
        "Semantic similarity is NOT factual correctness. High vector similarity indicates "
        "topical or lexical closeness and cannot verify truth, negation, or numerical precision."
    )
    is_optional = True
    optional_dependencies = ("sentence-transformers",)

    def __init__(
        self,
        normalization: NormalizationConfig | None = None,
        provider: EmbeddingProvider | None = None,
        similarity_threshold: float = 0.75,
    ) -> None:
        if not -1.0 <= similarity_threshold <= 1.0:
            raise MetricError("similarity_threshold must be between -1.0 and 1.0")
        self.normalization = normalization or NormalizationConfig()
        self.provider = provider or LocalNgramEmbeddingProvider()
        self.similarity_threshold = similarity_threshold

    def compute_item(self, record: DatasetRecord, response_text: str) -> MetricResult:
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
            # Clamp score to [0.0, 1.0] for standard metric reporting while recording raw cosine
            sim = max(0.0, raw_cos)

        passed = sim >= self.similarity_threshold
        return MetricResult(
            metric_name=self.name,
            score=sim,
            passed=passed,
            details={
                "cosine_similarity": round(sim, 6),
                "similarity_threshold": self.similarity_threshold,
                "provider": self.provider.metadata(),
                "limitation": self.limitation_note,
            },
        )

    def config_summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "similarity_threshold": self.similarity_threshold,
            "provider": self.provider.metadata(),
            "normalization": self.normalization.to_dict(),
        }
