"""Statistical primitives for OpenBench AI (v0.4.0).

These functions operate on already aligned, filtered and ID-sorted per-item values. They
are standard-library only. See ``docs/statistical_methodology.md`` for the methodology,
assumptions and the exact/approximate status of every procedure.
"""

from openbench.stats.bootstrap import bootstrap_mean_ci, paired_bootstrap_ci
from openbench.stats.effects import (
    BinaryPairedCounts,
    PairedEffects,
    binary_paired_counts,
    classify_practical_importance,
    paired_effects,
)
from openbench.stats.equivalence import (
    BINARY_EXACT_MAX_N,
    binary_exact_interval,
    binary_exact_p_lower,
    binary_exact_p_upper,
    equivalence_family_holm,
    score_statistic,
    tost_bootstrap_t,
    tost_paired_binary_exact,
)
from openbench.stats.results import (
    AdjustedPValues,
    ConfidenceInterval,
    EquivalenceInterval,
    EquivalenceResult,
    StatWarning,
    TestResult,
)
from openbench.stats.rng import derive_seed, derive_stream
from openbench.stats.tests import (
    adjust_p_values,
    mcnemar_exact,
    select_difference_test,
    sign_flip_test,
    studentized_mean,
)
from openbench.stats.variability import decompose_variability

__all__ = [
    "BINARY_EXACT_MAX_N",
    "AdjustedPValues",
    "BinaryPairedCounts",
    "ConfidenceInterval",
    "EquivalenceInterval",
    "EquivalenceResult",
    "PairedEffects",
    "StatWarning",
    "TestResult",
    "adjust_p_values",
    "binary_exact_interval",
    "binary_exact_p_lower",
    "binary_exact_p_upper",
    "binary_paired_counts",
    "bootstrap_mean_ci",
    "classify_practical_importance",
    "decompose_variability",
    "derive_seed",
    "derive_stream",
    "equivalence_family_holm",
    "mcnemar_exact",
    "paired_bootstrap_ci",
    "paired_effects",
    "score_statistic",
    "select_difference_test",
    "sign_flip_test",
    "studentized_mean",
    "tost_bootstrap_t",
    "tost_paired_binary_exact",
]
