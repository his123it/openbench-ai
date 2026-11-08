"""ROUGE-1 and ROUGE-L evaluation metrics for summarization and text overlap."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

from openbench.dataset import DatasetRecord
from openbench.exceptions import MetricError
from openbench.metrics.base import BaseMetric, MetricResult
from openbench.metrics.normalization import NormalizationConfig, normalize_text

_WORD_TOKEN_RE = re.compile(r"[\w]+", re.UNICODE)


def tokenize_words(text: str, normalization: NormalizationConfig) -> list[str]:
    """Tokenize text into normalized word tokens after applying NormalizationConfig.

    Falls back to whitespace-delimited tokens if `text` consists solely of symbolic
    characters (e.g., `'$?'`).
    """
    norm = normalize_text(text, normalization)
    tokens = _WORD_TOKEN_RE.findall(norm)
    if not tokens and norm.strip():
        # Symbol-only text (e.g. "$?") has no word tokens; fall back to whitespace tokens so
        # identical symbol answers match and different ones do not.
        tokens = norm.split()
    return tokens


def _lcs_length(xs: list[str], ys: list[str]) -> int:
    """Compute Longest Common Subsequence (LCS) length between two token lists."""
    if not xs or not ys:
        return 0
    m, n = len(xs), len(ys)
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        xi = xs[i - 1]
        for j in range(1, n + 1):
            if xi == ys[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = prev[j] if prev[j] >= curr[j - 1] else curr[j - 1]
        prev = curr
    return prev[n]


class RougeMetric(BaseMetric):
    r"""Computes ROUGE-1 (unigram overlap) and ROUGE-L (LCS overlap) F1, precision, and recall.

    Supported variants (`variant` parameter):
    - `"rouge_l"` (default when instantiated as `rouge` or `rouge_l`): Uses Longest Common
      Subsequence (LCS) of word tokens. Also computes and includes `rouge_1` in `details`.
    - `"rouge_1"`: Uses multiset unigram overlap as primary `score` while including `rouge_l`
      in `details`.

    Tokenization & Normalization:
    Text is normalized via `NormalizationConfig` (default: lowercase, whitespace stripped)
    and tokenized using Unicode alphanumeric word boundaries (`r"[\w]+"`).
    """

    name = "rouge"
    version = "0.2.0"
    description = (
        "Computes ROUGE-1 (unigram overlap) and ROUGE-L (Longest Common Subsequence) "
        "precision, recall, and F1 scores."
    )
    limitation_note = (
        "ROUGE measures lexical overlap against a reference text; it cannot detect "
        "factual hallucinations, negation flips, or valid paraphrases using distinct vocabulary."
    )

    def __init__(
        self,
        normalization: NormalizationConfig | None = None,
        variant: str = "rouge_l",
        pass_threshold: float = 0.5,
    ) -> None:
        if variant not in {"rouge_1", "rouge_l"}:
            raise MetricError(f"Unsupported ROUGE variant '{variant}'. Choose 'rouge_1' or 'rouge_l'.")
        self.normalization = normalization or NormalizationConfig()
        self.variant = variant
        self.pass_threshold = pass_threshold

    def compute_item(self, record: DatasetRecord, response_text: str) -> MetricResult:
        ref_tokens = tokenize_words(record.expected_answer, self.normalization)
        cand_tokens = tokenize_words(response_text, self.normalization)

        if not ref_tokens and not cand_tokens:
            return MetricResult(
                metric_name=self.name,
                score=1.0,
                passed=True,
                details={"rouge_1_f1": 1.0, "rouge_l_f1": 1.0},
            )
        if not ref_tokens or not cand_tokens:
            return MetricResult(
                metric_name=self.name,
                score=0.0,
                passed=False,
                details={
                    "variant": self.variant,
                    "rouge_1": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
                    "rouge_l": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
                },
            )

        # ROUGE-1 (unigram bag overlap)
        ref_counts = Counter(ref_tokens)
        cand_counts = Counter(cand_tokens)
        overlap_1 = sum(min(ref_counts[tok], cand_counts[tok]) for tok in ref_counts)
        r1_prec = overlap_1 / len(cand_tokens)
        r1_rec = overlap_1 / len(ref_tokens)
        r1_f1 = (2 * r1_prec * r1_rec / (r1_prec + r1_rec)) if (r1_prec + r1_rec) > 0 else 0.0

        # ROUGE-L (Longest Common Subsequence)
        lcs = _lcs_length(ref_tokens, cand_tokens)
        rl_prec = lcs / len(cand_tokens)
        rl_rec = lcs / len(ref_tokens)
        rl_f1 = (2 * rl_prec * rl_rec / (rl_prec + rl_rec)) if (rl_prec + rl_rec) > 0 else 0.0

        primary_score = r1_f1 if self.variant == "rouge_1" else rl_f1
        passed = primary_score >= self.pass_threshold

        return MetricResult(
            metric_name=self.name,
            score=primary_score,
            passed=passed,
            details={
                "variant": self.variant,
                "reference_token_count": len(ref_tokens),
                "candidate_token_count": len(cand_tokens),
                "rouge_1": {
                    "overlap_tokens": overlap_1,
                    "precision": round(r1_prec, 6),
                    "recall": round(r1_rec, 6),
                    "f1": round(r1_f1, 6),
                },
                "rouge_l": {
                    "lcs_length": lcs,
                    "precision": round(rl_prec, 6),
                    "recall": round(rl_rec, 6),
                    "f1": round(rl_f1, 6),
                },
            },
        )

    def config_summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "variant": self.variant,
            "pass_threshold": self.pass_threshold,
            "normalization": self.normalization.to_dict(),
        }


class Rouge1Metric(RougeMetric):
    """Dedicated metric entry for `rouge_1`."""

    name = "rouge_1"

    def __init__(
        self,
        normalization: NormalizationConfig | None = None,
        pass_threshold: float = 0.5,
    ) -> None:
        super().__init__(
            normalization=normalization,
            variant="rouge_1",
            pass_threshold=pass_threshold,
        )


class RougeLMetric(RougeMetric):
    """Dedicated metric entry for `rouge_l`."""

    name = "rouge_l"

    def __init__(
        self,
        normalization: NormalizationConfig | None = None,
        pass_threshold: float = 0.5,
    ) -> None:
        super().__init__(
            normalization=normalization,
            variant="rouge_l",
            pass_threshold=pass_threshold,
        )
