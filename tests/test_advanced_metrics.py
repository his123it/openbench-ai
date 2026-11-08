"""Unit tests for v0.2.0 advanced metrics: Precision, Recall, F1, ROUGE, BLEU, and Semantic Similarity."""

from __future__ import annotations

import pytest

from openbench.dataset import DatasetRecord
from openbench.exceptions import MetricError
from openbench.metrics import (
    BleuMetric,
    CallableEmbeddingProvider,
    F1Metric,
    HuggingFaceEmbeddingProvider,
    LocalNgramEmbeddingProvider,
    PrecisionMetric,
    RecallMetric,
    Rouge1Metric,
    RougeLMetric,
    RougeMetric,
    SemanticSimilarityMetric,
    create_metric,
    list_registered_metrics,
)


def _rec(expected: str, metadata: dict | None = None) -> DatasetRecord:
    return DatasetRecord(
        id="adv_01",
        category="reasoning",
        prompt="Prompt?",
        expected_answer=expected,
        metadata=metadata or {},
    )


def test_precision_recall_f1_binary_classification_decisions() -> None:
    """Precision, Recall, and F1 accurately evaluate binary Yes/No classification decisions."""
    pos_rec = _rec("Yes", {"task_type": "binary_classification"})
    neg_rec = _rec("No", {"task_type": "binary_classification"})

    prec = PrecisionMetric()
    rec = RecallMetric()
    f1 = F1Metric()

    # True Positive
    assert prec.compute_item(pos_rec, "Yes").score == 1.0
    assert rec.compute_item(pos_rec, "Yes").score == 1.0
    assert f1.compute_item(pos_rec, "Yes").score == 1.0

    # False Negative
    assert rec.compute_item(pos_rec, "No").score == 0.0
    assert f1.compute_item(pos_rec, "No").score == 0.0

    # False Positive
    fp_res = prec.compute_item(neg_rec, "Yes")
    assert fp_res.score == 0.0
    assert fp_res.details["fp"] == 1

    # True Negative
    tn_res = f1.compute_item(neg_rec, "No")
    assert tn_res.score == 1.0
    assert tn_res.details["tn"] == 1


def test_precision_recall_f1_element_set_predictions() -> None:
    """Precision, Recall, and F1 compute exact TP, FP, and FN on comma-separated element sets."""
    record = _rec("blue, green, red")
    prec = PrecisionMetric()
    rec = RecallMetric()
    f1 = F1Metric()

    # Candidate predicts 2 true elements ('blue', 'green') and 2 false elements ('yellow', 'purple'), missing 'red'
    # TP = 2, FP = 2 -> Precision = 0.5; FN = 1 -> Recall = 2/3; F1 = 2 * 0.5 * (2/3) / (0.5 + 2/3) = 4/7
    cand = "blue, green, yellow, purple"
    p_res = prec.compute_item(record, cand)
    r_res = rec.compute_item(record, cand)
    f_res = f1.compute_item(record, cand)

    assert pytest.approx(p_res.score, rel=1e-5) == 0.5
    assert pytest.approx(r_res.score, rel=1e-5) == 2.0 / 3.0
    assert pytest.approx(f_res.score, rel=1e-5) == 4.0 / 7.0


def test_rouge_1_and_rouge_l_metrics() -> None:
    """Rouge1Metric and RougeLMetric compute unigram and LCS overlap scores accurately."""
    record = _rec("the quick brown fox jumps over the lazy dog")
    r1 = Rouge1Metric()
    rl = RougeLMetric()

    identical = r1.compute_item(record, "The quick brown fox jumps over the lazy dog")
    assert identical.score == 1.0

    partial = rl.compute_item(record, "the quick fox jumps over a dog")
    assert 0.5 < partial.score < 1.0
    assert partial.details["rouge_l"]["lcs_length"] == 6

    with pytest.raises(MetricError):
        RougeMetric(variant="rouge_99")


def test_bleu_metric_brevity_penalty_and_ngram_precision() -> None:
    """BleuMetric rewards exact matches (1.0) and applies brevity penalty to overly short candidates."""
    record = _rec("Migrating the billing service to an asynchronous queue reduced checkout latency")
    bleu = BleuMetric()

    exact = bleu.compute_item(
        record,
        "Migrating the billing service to an asynchronous queue reduced checkout latency",
    )
    assert pytest.approx(exact.score, rel=1e-5) == 1.0
    assert exact.details["brevity_penalty"] == 1.0

    short_cand = bleu.compute_item(record, "Migrating the billing service")
    assert 0.0 < short_cand.score < 0.6
    assert short_cand.details["brevity_penalty"] < 1.0


def test_semantic_similarity_local_and_custom_callable_providers() -> None:
    """SemanticSimilarityMetric supports LocalNgramEmbeddingProvider and custom CallableEmbeddingProvider."""
    record = _rec("Hash passwords with Argon2id or bcrypt using a unique cryptographic salt.")

    local_metric = SemanticSimilarityMetric(
        provider=LocalNgramEmbeddingProvider(dimensions=64),
        similarity_threshold=0.5,
    )
    high_sim = local_metric.compute_item(
        record,
        "Use Argon2id or bcrypt with a unique cryptographic salt to hash passwords.",
    )
    low_sim = local_metric.compute_item(record, "Jupiter is the largest planet in the solar system.")
    assert high_sim.score > low_sim.score
    assert high_sim.passed is True
    assert "NOT factual correctness" in high_sim.details["limitation"]

    # Custom deterministic callable provider
    custom_provider = CallableEmbeddingProvider(
        embed_fn=lambda text: [1.0, 0.0] if "argon2id" in text.lower() else [0.0, 1.0],
        model_id="mock-2d",
    )
    custom_metric = SemanticSimilarityMetric(provider=custom_provider, similarity_threshold=0.8)
    assert custom_metric.compute_item(record, "Use Argon2id").score == 1.0
    assert custom_metric.compute_item(record, "Plaintext").score == 0.0


def test_huggingface_embedding_provider_missing_dependency_error() -> None:
    """HuggingFaceEmbeddingProvider raises a clear MetricError when sentence-transformers is absent."""
    provider = HuggingFaceEmbeddingProvider(model_id="sentence-transformers/all-MiniLM-L6-v2")
    with pytest.raises(MetricError, match="sentence-transformers"):
        provider.embed("test text")


def test_metric_registry_lists_all_metrics_and_validates_kwargs() -> None:
    """list_registered_metrics returns metadata for all built-in metrics and create_metric validates kwargs."""
    metadata_list = list_registered_metrics()
    names = {m.name for m in metadata_list}
    assert {
        "exact_match",
        "accuracy",
        "keyword_coverage",
        "precision",
        "recall",
        "f1",
        "rouge",
        "rouge_1",
        "rouge_l",
        "bleu",
        "semantic_similarity",
        "structured_output",
    }.issubset(names)

    with pytest.raises(MetricError, match="does not accept configuration parameter"):
        create_metric("exact_match", unknown_param=123)
