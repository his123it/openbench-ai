"""Sentence/item-level BLEU metric with clipped n-gram precision and brevity penalty."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any

from openbench.dataset import DatasetRecord
from openbench.metrics.base import BaseMetric, MetricResult
from openbench.metrics.normalization import NormalizationConfig
from openbench.metrics.rouge import tokenize_words


def _extract_ngrams(tokens: list[str], n: int) -> Counter[tuple[str, ...]]:
    if len(tokens) < n:
        return Counter()
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


class BleuMetric(BaseMetric):
    r"""Computes BLEU score with clipped n-gram precisions, brevity penalty, and Chen-Cherry smoothing.

    Methodology & Behavior:
    - Tokenizes candidate and reference strings into normalized word tokens (`tokenize_words`).
    - Dynamically adjusts maximum n-gram order `effective_max_n = min(max_n, len(ref_tokens))`
      so short 1-word or 2-word reference answers are not zeroed out by non-existent 4-grams.
    - Applies add-epsilon smoothing (`smoothing_epsilon=0.1`) for higher-order n-grams ($n > 1$)
      when a lower-order overlap exists, following sentence-BLEU best practices (Chen & Cherry, 2014).
    - Computes Brevity Penalty:
      $$\text{BP} = \begin{cases} 1.0 & \text{if } c > r \\ \exp(1 - r / c) & \text{if } c \le r \end{cases}$$
      where $c$ is candidate token length and $r$ is reference token length.
    """

    name = "bleu"
    version = "0.2.0"
    score_kind = "bounded_continuous"
    description = (
        "Computes smoothed sentence BLEU using clipped n-gram precisions (up to 4-grams) "
        "and the standard exponential brevity penalty."
    )
    limitation_note = (
        "BLEU is sensitive to exact lexical choice and tokenization, penalizes valid synonyms, "
        "and is unstable on very short single-token responses without smoothing."
    )

    def __init__(
        self,
        normalization: NormalizationConfig | None = None,
        max_n: int = 4,
        smoothing_epsilon: float = 0.1,
        pass_threshold: float = 0.5,
    ) -> None:
        self.normalization = normalization or NormalizationConfig()
        self.max_n = max(1, max_n)
        self.smoothing_epsilon = smoothing_epsilon
        self.pass_threshold = pass_threshold

    def compute_item(self, record: DatasetRecord, response_text: str) -> MetricResult:
        ref_tokens = tokenize_words(record.expected_answer, self.normalization)
        cand_tokens = tokenize_words(response_text, self.normalization)

        r_len = len(ref_tokens)
        c_len = len(cand_tokens)

        if r_len == 0 or c_len == 0:
            return MetricResult(
                metric_name=self.name,
                score=0.0,
                passed=False,
                details={
                    "brevity_penalty": 0.0,
                    "reference_length": r_len,
                    "candidate_length": c_len,
                    "ngram_precisions": {},
                },
            )

        effective_n = min(self.max_n, r_len, c_len)
        precisions: dict[str, float] = {}
        log_sum = 0.0
        unigram_matches = 0

        for n in range(1, effective_n + 1):
            ref_ngrams = _extract_ngrams(ref_tokens, n)
            cand_ngrams = _extract_ngrams(cand_tokens, n)
            total_cand = sum(cand_ngrams.values())
            clipped_matches = sum(
                min(count, ref_ngrams[ng]) for ng, count in cand_ngrams.items()
            )
            if n == 1:
                unigram_matches = clipped_matches

            if clipped_matches == 0:
                if n == 1 or unigram_matches == 0:
                    p_n = 0.0
                else:
                    p_n = self.smoothing_epsilon / total_cand
            else:
                p_n = clipped_matches / total_cand

            precisions[f"{n}-gram"] = round(p_n, 6)
            if p_n <= 0.0:
                return MetricResult(
                    metric_name=self.name,
                    score=0.0,
                    passed=False,
                    details={
                        "brevity_penalty": 0.0,
                        "reference_length": r_len,
                        "candidate_length": c_len,
                        "effective_max_n": effective_n,
                        "ngram_precisions": precisions,
                    },
                )
            log_sum += math.log(p_n) / effective_n

        bp = 1.0 if c_len > r_len else math.exp(1.0 - (r_len / c_len))
        bleu_score = bp * math.exp(log_sum)
        bleu_score = min(max(bleu_score, 0.0), 1.0)
        passed = bleu_score >= self.pass_threshold

        return MetricResult(
            metric_name=self.name,
            score=bleu_score,
            passed=passed,
            details={
                "brevity_penalty": round(bp, 6),
                "reference_length": r_len,
                "candidate_length": c_len,
                "effective_max_n": effective_n,
                "ngram_precisions": precisions,
            },
        )

    def config_summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "max_n": self.max_n,
            "smoothing_epsilon": self.smoothing_epsilon,
            "pass_threshold": self.pass_threshold,
            "normalization": self.normalization.to_dict(),
        }
