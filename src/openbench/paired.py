"""Paired model comparison on stable record IDs (plan sections 1.3-1.11, 4 and 5; v0.4.0).

The engine compares stored evaluations only; it never constructs or calls a provider.
Items are aligned on record IDs after lexicographic sorting, so input file and response
order never change a result. For each arm pair and metric it reports the sample size and
exclusions, arm means, the paired difference with a percentile bootstrap CI, the
metric-appropriate test of no difference with raw and multiplicity-adjusted p-values,
effect sizes, wins/losses/ties, disagreements, category and difficulty breakdowns,
missing-response coverage and a failure-rate comparison. Optional: practical-importance
classification (``min_practical_difference``) and TOST equivalence
(``equivalence_margin``, a separate Holm family).

Inputs with hard incompatibilities (section 5) are refused unless the caller asks for
an exploratory, descriptive-only comparison and states a reason.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openbench.evaluator import EvaluationResult
from openbench.exceptions import ConfigurationError, IncompatibleComparisonError, OpenBenchError
from openbench.itemscores import (
    INVALID,
    MISSING,
    VALID,
    ItemObservation,
    collect_item_scores,
    metric_score_kind,
    metric_version_of,
)
from openbench.security import redact_sensitive_data
from openbench.stats import (
    EquivalenceResult,
    adjust_p_values,
    binary_paired_counts,
    bootstrap_mean_ci,
    classify_practical_importance,
    equivalence_family_holm,
    mcnemar_exact,
    paired_bootstrap_ci,
    paired_effects,
    select_difference_test,
    sign_flip_test,
    tost_bootstrap_t,
    tost_paired_binary_exact,
)
from openbench.stats.bootstrap import DEFAULT_RESAMPLES
from openbench.stats.effects import TIE_TOLERANCE
from openbench.stats.tests import DEFAULT_EXACT_PERMUTATION_MAX, DEFAULT_PERMUTATIONS

SCHEMA_VERSION = "0.4.0"
MISSING_POLICIES = ("complete_case", "missing_as_failure")
CORRECTIONS = ("holm", "bh", "none")
SMALL_SAMPLE_THRESHOLDS = {
    "no_inference_below": 2,
    "very_small_sample_below": 10,
    "low_power_wide_intervals_below": 30,
    "few_discordant_pairs_below": 10,
}
GENERATION_KEYS = ("temperature", "top_p", "max_tokens", "stop_sequences", "stream", "extra_options")
EXPLORATORY_BANNER = "EXPLORATORY: inputs are incompatible for inferential comparison"
EXPLORATORY_REASON_CODE = "exploratory_incompatible_inputs"
INTERVAL_NOTE = (
    "Percentile bootstrap interval for the mean paired difference over items. It treats the "
    "analysed items as a sample of similar items; it does not include run-to-run generation "
    "noise and is not adjusted for multiple comparisons."
)
NOT_SIGNIFICANT_TEMPLATE = (
    "No statistically significant difference was detected (adjusted p = {p}). The {level} CI "
    "[{lo}, {hi}] is compatible with differences in this range."
)
LIMITATIONS = (
    "Results describe these fixed models, prompts, generation settings and references on the "
    "analysed items only.",
    "A non-significant difference is not evidence of equivalence; only a TOST with a "
    "pre-specified margin can support an equivalence claim.",
    "Per-comparison confidence intervals are not adjusted for multiple comparisons.",
    "Continuous-metric equivalence testing is disabled in v0.4.0 (descriptive values only).",
    "Category and difficulty breakdowns are descriptive unless stratum tests are requested.",
)


def _fmt(value: float | None, digits: int = 4) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Arms and policy
# ---------------------------------------------------------------------------


@dataclass
class ComparisonArm:
    """One side of a comparison: a single run, a repeat group, or an evaluation JSON file."""

    arm_id: str
    label: str
    evaluations: list[EvaluationResult]
    source: str = "evaluation"
    run_records: list[dict[str, Any]] = field(default_factory=list)
    source_paths: list[str] = field(default_factory=list)
    input_sha256: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.evaluations:
            raise ConfigurationError(f"Arm '{self.arm_id}' has no evaluations")
        if not self.arm_id or not str(self.arm_id).strip():
            raise ConfigurationError("Arm IDs must be non-empty")

    @property
    def repetitions(self) -> int:
        return len(self.evaluations)

    @property
    def model_identity(self) -> dict[str, Any]:
        ev = self.evaluations[0]
        return {
            "model_name": ev.model_name,
            "model_version": ev.model_version,
            "model_provider": ev.model_provider,
        }

    def summary(self) -> dict[str, Any]:
        lineage = self.run_records[0].get("lineage") if self.run_records else None
        return {
            "arm_id": self.arm_id,
            "label": self.label,
            "source": self.source,
            "repetitions": self.repetitions,
            **self.model_identity,
            "is_mock_provider": self.is_mock,
            "evaluation_ids": [ev.evaluation_id for ev in self.evaluations],
            "run_ids": [r.get("run_id") for r in self.run_records],
            "repeat_group_id": (lineage or {}).get("repeat_group_id") if lineage else None,
            "run_statuses": [r.get("status") for r in self.run_records],
            "source_paths": list(self.source_paths),
            "input_sha256": list(self.input_sha256),
        }

    @property
    def is_mock(self) -> bool:
        if self.run_records:
            return any(bool(r.get("is_mock_provider")) for r in self.run_records)
        cfg = self.evaluations[0].inference_config or {}
        return bool(cfg.get("is_mock")) or self.evaluations[0].model_provider == "mock"

    @classmethod
    def from_evaluation_file(cls, path: str | Path, arm_id: str | None = None) -> ComparisonArm:
        target = Path(path)
        if not target.is_file():
            raise FileNotFoundError(f"Evaluation result file does not exist: {target}")
        raw = target.read_bytes()
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise OpenBenchError(f"Malformed JSON in evaluation file {target}: {exc}") from exc
        if not isinstance(payload, dict):
            raise OpenBenchError(f"Evaluation file {target} must contain a JSON object")
        try:
            ev = EvaluationResult.from_dict(payload)
        except (KeyError, TypeError, ValueError) as exc:
            raise OpenBenchError(f"Invalid evaluation file {target}: {exc}") from exc
        return cls(
            arm_id=arm_id or target.stem,
            label=ev.model_name,
            evaluations=[ev],
            source="evaluation_file",
            source_paths=[str(target)],
            input_sha256=[hashlib.sha256(raw).hexdigest()],
        )

    @classmethod
    def from_experiments(cls, manager: Any, identifier: str) -> ComparisonArm:
        """Load a stored run (``run_...``) or repeat group (``grp_...``). No provider is built."""
        from openbench.integrity import compute_file_sha256
        from openbench.lineage import is_repeat_group_id

        if is_repeat_group_id(identifier):
            members = manager.list_runs(repeat_group_id=identifier)
            members = sorted(members, key=lambda r: (r.repetition_index, r.run_id))
            if not members:
                raise KeyError(f"Repeat group '{identifier}' has no member runs")
            evaluations = [manager.load_evaluation(m.run_id) for m in members]
            source = "repeat_group"
        else:
            members = [manager.load_run_record(identifier)]
            evaluations = [manager.load_evaluation(identifier)]
            source = "run"
        paths = [str(Path(m.run_dir) / "evaluation.json") for m in members]
        return cls(
            arm_id=identifier,
            label=evaluations[0].model_name,
            evaluations=evaluations,
            source=source,
            run_records=[m.to_dict() for m in members],
            source_paths=paths,
            input_sha256=[compute_file_sha256(p) for p in paths],
        )


@dataclass(frozen=True)
class ComparisonPolicy:
    """Analysis parameters (all recorded in the output)."""

    missing_policy: str = "complete_case"
    correction: str = "holm"
    confidence_level: float = 0.95
    resamples: int = DEFAULT_RESAMPLES
    permutations: int = DEFAULT_PERMUTATIONS
    exact_permutation_max: int = DEFAULT_EXACT_PERMUTATION_MAX
    analysis_seed: int = 0
    alpha: float = 0.05
    min_practical_difference: float | None = None
    equivalence_margin: float | None = None
    stratified_bootstrap: bool = False
    stratum_tests: bool = False
    exploratory: bool = False
    exploratory_reason: str | None = None
    baseline: str | None = None
    selected_ids: tuple[str, ...] | None = None
    tie_tolerance: float = TIE_TOLERANCE

    def __post_init__(self) -> None:
        if self.missing_policy not in MISSING_POLICIES:
            raise ConfigurationError(f"missing_policy must be one of {MISSING_POLICIES}")
        if self.correction not in CORRECTIONS:
            raise ConfigurationError(f"correction must be one of {CORRECTIONS}")
        if not 0.5 <= self.confidence_level <= 0.999:
            raise ConfigurationError("confidence_level must lie within [0.5, 0.999]")
        if not 0.0 < self.alpha < 0.5:
            raise ConfigurationError("alpha must lie in (0, 0.5)")
        for name in ("resamples", "permutations"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ConfigurationError(f"{name} must be a positive integer")
        if (
            isinstance(self.exact_permutation_max, bool)
            or not isinstance(self.exact_permutation_max, int)
            or not 0 <= self.exact_permutation_max <= DEFAULT_EXACT_PERMUTATION_MAX
        ):
            raise ConfigurationError(
                f"exact_permutation_max must be an integer in [0, {DEFAULT_EXACT_PERMUTATION_MAX}]"
            )
        if isinstance(self.analysis_seed, bool) or not isinstance(self.analysis_seed, int):
            raise ConfigurationError("analysis_seed must be an integer")
        for name in ("min_practical_difference", "equivalence_margin"):
            value = getattr(self, name)
            if value is not None and not (
                isinstance(value, (int, float)) and not isinstance(value, bool)
                and math.isfinite(value) and value > 0
            ):
                raise ConfigurationError(f"{name} must be a positive finite number")
        if self.exploratory and not (self.exploratory_reason and self.exploratory_reason.strip()):
            raise ConfigurationError(
                "Exploratory mode requires a non-empty exploratory reason"
            )
        if self.exploratory_reason and not self.exploratory:
            raise ConfigurationError("An exploratory reason was given without exploratory mode")

    def to_dict(self) -> dict[str, Any]:
        return {
            "missing_policy": self.missing_policy,
            "correction": self.correction,
            "confidence_level": self.confidence_level,
            "resamples": self.resamples,
            "permutations": self.permutations,
            "exact_permutation_max": self.exact_permutation_max,
            "analysis_seed": self.analysis_seed,
            "alpha": self.alpha,
            "min_practical_difference": self.min_practical_difference,
            "equivalence_margin": self.equivalence_margin,
            "equivalence_margin_prespecified": self.equivalence_margin is not None,
            "stratified_bootstrap": self.stratified_bootstrap,
            "stratum_tests": self.stratum_tests,
            "exploratory": self.exploratory,
            "exploratory_reason": self.exploratory_reason,
            "baseline": self.baseline,
            "selected_ids_count": None if self.selected_ids is None else len(self.selected_ids),
            "selected_ids_sha256": (
                None
                if self.selected_ids is None
                else _sha256_text(json.dumps(sorted(self.selected_ids)))
            ),
            "tie_tolerance": self.tie_tolerance,
            "small_sample_thresholds": dict(SMALL_SAMPLE_THRESHOLDS),
        }


# ---------------------------------------------------------------------------
# Compatibility (section 5)
# ---------------------------------------------------------------------------


@dataclass
class CompatibilityReport:
    hard: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[dict[str, Any]] = field(default_factory=list)
    info: list[dict[str, Any]] = field(default_factory=list)

    @property
    def compatible(self) -> bool:
        return not self.hard

    def add(self, level: str, code: str, message: str, **details: Any) -> None:
        entry = {"code": code, "message": message, "details": redact_sensitive_data(details)}
        {"hard": self.hard, "warning": self.warnings, "info": self.info}[level].append(entry)

    def to_dict(self) -> dict[str, Any]:
        return {
            "compatible_for_inference": self.compatible,
            "hard_incompatibilities": list(self.hard),
            "warnings": list(self.warnings),
            "info": list(self.info),
        }

    def describe(self) -> str:
        lines = [f"- [{e['code']}] {e['message']}" for e in self.hard]
        return "\n".join(lines)


def _record_hashes(arm: ComparisonArm) -> dict[str, str] | None:
    """Per-record content hashes from v0.4.0 run lineage (all members must agree)."""
    if not arm.run_records:
        return None
    maps = []
    for rec in arm.run_records:
        lineage = rec.get("lineage") or {}
        hashes = lineage.get("record_content_sha256")
        if not isinstance(hashes, dict):
            return None
        maps.append(hashes)
    merged: dict[str, str] = {}
    for m in maps:
        for k, v in m.items():
            merged.setdefault(k, v)
    return merged


def _arm_internal_checks(arm: ComparisonArm, report: CompatibilityReport, metrics: Sequence[str]) -> None:
    if arm.repetitions < 2:
        return
    checksums = {ev.dataset_summary.get("checksum") for ev in arm.evaluations}
    if len(checksums) > 1:
        report.add("hard", "arm_inconsistent_dataset", f"Repetitions of arm '{arm.arm_id}' used different dataset content.", arm=arm.arm_id)
    for metric in metrics:
        versions = {metric_version_of(ev, metric) for ev in arm.evaluations}
        if len(versions) > 1:
            report.add("hard", "arm_inconsistent_metric_version", f"Repetitions of arm '{arm.arm_id}' used different versions of metric '{metric}'.", arm=arm.arm_id, metric=metric)
    configs = {
        json.dumps({k: ev.config.get(k) for k in ("normalization", "metric_options")}, sort_keys=True, default=str)
        for ev in arm.evaluations
    }
    if len(configs) > 1:
        report.add("hard", "arm_inconsistent_scoring_configuration", f"Repetitions of arm '{arm.arm_id}' used different normalization or metric options.", arm=arm.arm_id)
    models = {json.dumps([ev.model_name, ev.model_version, ev.model_provider]) for ev in arm.evaluations}
    if len(models) > 1:
        report.add("hard", "arm_inconsistent_model", f"Repetitions of arm '{arm.arm_id}' are from different models.", arm=arm.arm_id)


def _generation_view(arm: ComparisonArm) -> dict[str, Any]:
    cfg = dict(arm.evaluations[0].inference_config or {})
    view = {k: cfg.get(k) for k in GENERATION_KEYS if k in cfg}
    base_seed = None
    if arm.run_records:
        base_seed = (arm.run_records[0].get("lineage") or {}).get("generation_base_seed")
        if base_seed is None:
            base_seed = cfg.get("seed")
    else:
        base_seed = cfg.get("seed")
    view["seed"] = base_seed
    return redact_sensitive_data(view)


def check_compatibility(
    baseline: ComparisonArm,
    challenger: ComparisonArm,
    metrics: Sequence[str],
    analysed_ids: Sequence[str],
) -> CompatibilityReport:
    """Classify the compatibility of one arm pair (plan section 5)."""
    report = CompatibilityReport()
    for arm in (baseline, challenger):
        _arm_internal_checks(arm, report, metrics)
    ev_a, ev_b = baseline.evaluations[0], challenger.evaluations[0]

    hashes_a, hashes_b = _record_hashes(baseline), _record_hashes(challenger)
    if hashes_a is not None and hashes_b is not None:
        differing = sorted(
            rid for rid in analysed_ids
            if rid in hashes_a and rid in hashes_b and hashes_a[rid] != hashes_b[rid]
        )
        if differing:
            report.add("hard", "record_content_differs", f"{len(differing)} analysed record(s) differ in content between arms.", record_ids=differing[:50], total=len(differing))
        content_verified = not differing
    else:
        ck_a = ev_a.dataset_summary.get("checksum")
        ck_b = ev_b.dataset_summary.get("checksum")
        if ck_a != ck_b:
            report.add("hard", "dataset_checksum_differs", "Dataset checksums differ and per-record content hashes are unavailable for at least one arm (pre-0.4.0 runs or evaluation files).", baseline_checksum=ck_a, challenger_checksum=ck_b)
            content_verified = False
        else:
            content_verified = True
            report.add("info", "content_verified_by_dataset_checksum", "Record content verified by whole-dataset checksum (per-record hashes unavailable).")

    for metric in metrics:
        in_a = metric in ev_a.aggregate_results
        in_b = metric in ev_b.aggregate_results
        if not (in_a and in_b):
            report.add("hard", "metric_not_in_arm", f"Metric '{metric}' was not computed for {'baseline' if not in_a else 'challenger'}.", metric=metric)
            continue
        va, vb = metric_version_of(ev_a, metric), metric_version_of(ev_b, metric)
        if va != vb:
            report.add("hard", "metric_version_differs", f"Metric '{metric}' version differs ({va} vs {vb}).", metric=metric, baseline=va, challenger=vb)
        opts_a = (ev_a.config.get("metric_options") or {}).get(metric)
        opts_b = (ev_b.config.get("metric_options") or {}).get(metric)
        if opts_a != opts_b:
            report.add("hard", "metric_options_differ", f"Options of metric '{metric}' differ between arms.", metric=metric, baseline=opts_a, challenger=opts_b)
    if ev_a.config.get("normalization") != ev_b.config.get("normalization"):
        report.add("hard", "normalization_differs", "Text normalization configuration differs between arms.", baseline=ev_a.config.get("normalization"), challenger=ev_b.config.get("normalization"))

    ver_a, ver_b = ev_a.dataset_summary.get("version"), ev_b.dataset_summary.get("version")
    if ver_a != ver_b and content_verified:
        report.add("warning", "dataset_version_label_differs", f"Dataset version labels differ ({ver_a} vs {ver_b}) but analysed content is verified identical.", baseline=ver_a, challenger=ver_b)

    ids_a = {i.id for i in ev_a.individual_results}
    ids_b = {i.id for i in ev_b.individual_results}
    if ids_a != ids_b:
        report.add("warning", "record_id_sets_differ", "Record ID sets differ; the intersection is analysed and exclusions are counted per arm.", only_in_baseline=len(ids_a - ids_b), only_in_challenger=len(ids_b - ids_a), shared=len(ids_a & ids_b))

    gen_a, gen_b = _generation_view(baseline), _generation_view(challenger)
    differing_keys = sorted(k for k in set(gen_a) | set(gen_b) if gen_a.get(k) != gen_b.get(k))
    if differing_keys:
        report.add("warning", "generation_configuration_differs", "Generation configuration differs: " + ", ".join(differing_keys), differences={k: {"baseline": gen_a.get(k), "challenger": gen_b.get(k)} for k in differing_keys})

    for arm in (baseline, challenger):
        if arm.is_mock:
            report.add("warning", "mock_provider", f"Arm '{arm.arm_id}' uses the mock provider; results do not represent a real model.", arm=arm.arm_id)
        incomplete = [r.get("run_id") for r in arm.run_records if r.get("status") not in (None, "completed")]
        if incomplete:
            report.add("warning", "incomplete_runs", f"Arm '{arm.arm_id}' includes runs that did not complete; missing items follow the missing-data policy.", arm=arm.arm_id, run_ids=incomplete)
        if arm.source == "repeat_group" and arm.run_records:
            planned = (arm.run_records[0].get("lineage") or {}).get("repetition_count")
            if planned and planned != len(arm.run_records):
                report.add("warning", "incomplete_repeat_group", f"Repeat group '{arm.arm_id}' has {len(arm.run_records)} of {planned} planned repetitions.", arm=arm.arm_id)
    if baseline.repetitions != challenger.repetitions:
        report.add("warning", "repetition_counts_differ", f"Repetition counts differ ({baseline.repetitions} vs {challenger.repetitions}); per-item means use different numbers of repetitions.")
    if ev_a.model_provider != ev_b.model_provider:
        report.add("info", "provider_differs", f"Providers differ ({ev_a.model_provider} vs {ev_b.model_provider}).")
    if baseline.model_identity == challenger.model_identity:
        report.add("info", "same_model_identity", "Both arms report the same model identity.")
    return report


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


def _arm_value(obs: ItemObservation, policy: str) -> float | None:
    if policy == "complete_case":
        return obs.mean
    vals = []
    for v, s in zip(obs.values, obs.statuses, strict=True):
        if s == VALID and v is not None:
            vals.append(v)
        elif s in {MISSING, INVALID}:
            vals.append(0.0)
    return math.fsum(vals) / len(vals) if vals else None


def _exclusion_reason(obs: ItemObservation) -> str:
    if any(s == MISSING for s in obs.statuses):
        return "missing"
    if any(s == INVALID for s in obs.statuses):
        return "invalid"
    return "metric_unavailable"


@dataclass
class PairedComparisonResult:
    """Full output of :meth:`PairedComparisonEngine.compare` (JSON via :meth:`to_dict`)."""

    payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return self.payload

    @property
    def comparisons(self) -> list[dict[str, Any]]:
        return self.payload["comparisons"]

    @property
    def analysis_mode(self) -> str:
        return self.payload["analysis_mode"]


class PairedComparisonEngine:
    """Compares stored arms on aligned record IDs. Never calls a model provider."""

    def compare(
        self,
        arms: Sequence[ComparisonArm],
        metrics: Sequence[str] | None = None,
        policy: ComparisonPolicy | None = None,
    ) -> PairedComparisonResult:
        policy = policy or ComparisonPolicy()
        arms = list(arms)
        if len(arms) < 2:
            raise ConfigurationError("At least two arms are required for a paired comparison")
        ids = [a.arm_id for a in arms]
        if len(set(ids)) != len(ids):
            raise ConfigurationError(f"Arm IDs must be unique: {ids}")
        metric_names = list(metrics) if metrics else self._default_metrics(arms)
        if not metric_names:
            raise ConfigurationError("No metrics to compare")
        pairs = self._pairs(arms, policy.baseline)

        pair_reports = []
        any_hard = False
        for base, chal in pairs:
            universe = self._universe(base, chal, policy)[0]
            report = check_compatibility(base, chal, metric_names, universe)
            any_hard = any_hard or not report.compatible
            pair_reports.append(report)
        if any_hard and not policy.exploratory:
            text = "\n".join(
                f"{b.arm_id} vs {c.arm_id}:\n{r.describe()}"
                for (b, c), r in zip(pairs, pair_reports, strict=True)
                if not r.compatible
            )
            raise IncompatibleComparisonError(
                "Inputs are incompatible for inferential comparison:\n" + text
                + "\nUse exploratory mode with a stated reason for a descriptive-only comparison.",
                report=pair_reports,
            )
        exploratory = bool(policy.exploratory)

        comparisons: list[dict[str, Any]] = []
        failure_sections: list[dict[str, Any]] = []
        for (base, chal), report in zip(pairs, pair_reports, strict=True):
            universe, universe_info = self._universe(base, chal, policy)
            for metric in metric_names:
                comparisons.append(
                    self._compare_metric(base, chal, metric, policy, universe, universe_info, report, exploratory)
                )
            failure_sections.append(self._failure_rates(base, chal, metric_names[0], universe, policy, exploratory))

        families = self._apply_families(comparisons, failure_sections, policy, exploratory)
        input_digest = _sha256_text(
            json.dumps(
                {
                    "arms": [a.summary() for a in arms],
                    "metrics": metric_names,
                    "policy": policy.to_dict(),
                },
                sort_keys=True,
                default=str,
            )
        )
        payload = {
            "schema_version": SCHEMA_VERSION,
            "analysis": "paired_comparison",
            "analysis_id": f"ana_{input_digest[:12]}",
            "analysis_mode": "exploratory" if exploratory else "inferential",
            "banner": EXPLORATORY_BANNER if exploratory else None,
            "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "arms": [a.summary() for a in arms],
            "pairs": [
                {"baseline": b.arm_id, "challenger": c.arm_id, "compatibility": r.to_dict()}
                for (b, c), r in zip(pairs, pair_reports, strict=True)
            ],
            "metrics": [
                {"name": m, "score_kind": metric_score_kind(m), "higher_is_better": True}
                for m in metric_names
            ],
            "parameters": policy.to_dict(),
            "analysis_seed": policy.analysis_seed,
            "comparisons": comparisons,
            "failure_rate_comparisons": failure_sections,
            "families": families,
            "provider_calls": 0,
            "limitations": list(LIMITATIONS),
            "input_digest_sha256": input_digest,
        }
        if exploratory:
            payload["exploratory_reason"] = policy.exploratory_reason
        return PairedComparisonResult(payload)

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _default_metrics(arms: Sequence[ComparisonArm]) -> list[str]:
        shared = None
        for arm in arms:
            names = set(arm.evaluations[0].aggregate_results)
            shared = names if shared is None else shared & names
        order = list(arms[0].evaluations[0].aggregate_results)
        return [m for m in order if m in (shared or set())]

    @staticmethod
    def _pairs(arms: list[ComparisonArm], baseline: str | None) -> list[tuple[ComparisonArm, ComparisonArm]]:
        if baseline is not None:
            matches = [a for a in arms if a.arm_id == baseline] or [a for a in arms if a.label == baseline]
            if len(matches) != 1:
                raise ConfigurationError(
                    f"Baseline '{baseline}' must match exactly one arm ID or label; "
                    f"arms: {[a.arm_id for a in arms]}"
                )
            base = matches[0]
            return [(base, a) for a in arms if a is not base]
        ordered = sorted(arms, key=lambda a: a.arm_id)
        return [
            (ordered[i], ordered[j]) for i in range(len(ordered)) for j in range(i + 1, len(ordered))
        ]

    @staticmethod
    def _universe(base: ComparisonArm, chal: ComparisonArm, policy: ComparisonPolicy) -> tuple[list[str], dict[str, Any]]:
        ids_a = {i.id for ev in base.evaluations for i in ev.individual_results}
        ids_b = {i.id for ev in chal.evaluations for i in ev.individual_results}
        shared = ids_a & ids_b
        info: dict[str, Any] = {
            "not_in_all_arms": len(ids_a ^ ids_b),
            "only_in_baseline": len(ids_a - ids_b),
            "only_in_challenger": len(ids_b - ids_a),
            "filtered_out": 0,
            "requested_ids_not_found": 0,
        }
        if policy.selected_ids is not None:
            selected = set(policy.selected_ids)
            info["filtered_out"] = len(shared - selected)
            info["requested_ids_not_found"] = len(selected - shared)
            shared &= selected
        return sorted(shared), info

    def _compare_metric(
        self,
        base: ComparisonArm,
        chal: ComparisonArm,
        metric: str,
        policy: ComparisonPolicy,
        universe: list[str],
        universe_info: dict[str, Any],
        report: CompatibilityReport,
        exploratory: bool,
    ) -> dict[str, Any]:
        kind = metric_score_kind(metric)
        obs_a = collect_item_scores(base.evaluations, metric) if metric in base.evaluations[0].aggregate_results else {}
        obs_b = collect_item_scores(chal.evaluations, metric) if metric in chal.evaluations[0].aggregate_results else {}
        rows = []
        breakdown = {"missing_in": {base.arm_id: 0, chal.arm_id: 0}, "invalid_in": {base.arm_id: 0, chal.arm_id: 0}, "metric_unavailable_in": {base.arm_id: 0, chal.arm_id: 0}}
        excluded_ids = []
        for rid in universe:
            oa, ob = obs_a.get(rid), obs_b.get(rid)
            va = _arm_value(oa, policy.missing_policy) if oa else None
            vb = _arm_value(ob, policy.missing_policy) if ob else None
            if va is None or vb is None:
                for arm, obs, val in ((base, oa, va), (chal, ob, vb)):
                    if val is None:
                        reason = _exclusion_reason(obs) if obs else "metric_unavailable"
                        breakdown[f"{reason}_in"][arm.arm_id] += 1
                excluded_ids.append(rid)
                continue
            src = oa or ob
            rows.append(
                {
                    "id": rid,
                    "category": src.category,
                    "difficulty": src.difficulty,
                    "baseline": va,
                    "challenger": vb,
                    "difference": vb - va,
                    "baseline_repetitions_used": oa.repetitions_used,
                    "challenger_repetitions_used": ob.repetitions_used,
                }
            )
        diffs = [r["difference"] for r in rows]
        a_vals = [r["baseline"] for r in rows]
        b_vals = [r["challenger"] for r in rows]
        n = len(rows)
        tol = policy.tie_tolerance
        wins = [r["id"] for r in rows if r["difference"] > tol]
        losses = [r["id"] for r in rows if r["difference"] < -tol]
        ties = [r["id"] for r in rows if abs(r["difference"]) <= tol]
        disagreements = sorted(
            (r for r in rows if abs(r["difference"]) > tol),
            key=lambda r: (-abs(r["difference"]), r["id"]),
        )
        coverage = {
            arm.arm_id: self._coverage(obs, universe)
            for arm, obs in ((base, obs_a), (chal, obs_b))
        }
        result: dict[str, Any] = {
            "comparison_id": f"{base.arm_id}|{chal.arm_id}|{metric}",
            "baseline": base.arm_id,
            "challenger": chal.arm_id,
            "baseline_label": base.label,
            "challenger_label": chal.label,
            "metric": {"name": metric, "version": metric_version_of(base.evaluations[0], metric), "score_kind": kind, "higher_is_better": True},
            "estimand": "mean over aligned items of (challenger - baseline)",
            "sample_count": n,
            "excluded_count": len(excluded_ids),
            "excluded_breakdown": {**breakdown, **universe_info},
            "excluded_ids": excluded_ids,
            "arm_means": {
                base.arm_id: (math.fsum(a_vals) / n) if n else None,
                chal.arm_id: (math.fsum(b_vals) / n) if n else None,
            },
            "point_estimate": (math.fsum(diffs) / n) if n else None,
            "wins_losses_ties": {
                "challenger_better": len(wins),
                "baseline_better": len(losses),
                "ties": len(ties),
                "challenger_better_ids": wins,
                "baseline_better_ids": losses,
                "tie_ids": ties,
                "tie_tolerance": tol,
            },
            "disagreements": [
                {"id": r["id"], "baseline": r["baseline"], "challenger": r["challenger"], "difference": r["difference"]}
                for r in disagreements
            ],
            "coverage": coverage,
            "warnings": [],
            "unavailable": {},
        }
        result["per_item"] = rows
        if exploratory:
            for name in ("confidence_interval", "arm_confidence_intervals", "test", "p_value", "adjusted_p_value", "equivalence", "practical_classification", "effect_size"):
                result[name] = None
                result["unavailable"][name] = EXPLORATORY_REASON_CODE
            result["breakdowns"] = {
                "by_category": self._breakdown(rows, "category", metric, base, chal, policy, inferential=False),
                "by_difficulty": self._breakdown(rows, "difficulty", metric, base, chal, policy, inferential=False),
            }
            result["interpretation"] = "Descriptive only (exploratory mode); no inferential statistics."
            return result

        seed_kwargs = {"analysis_seed": policy.analysis_seed, "metric": metric, "arm_a": base.arm_id, "arm_b": chal.arm_id}
        strata = [r["category"] for r in rows] if policy.stratified_bootstrap else None
        ci = paired_bootstrap_ci(diffs, confidence_level=policy.confidence_level, resamples=policy.resamples, strata=strata, **seed_kwargs)
        result["confidence_interval"] = {**ci.to_dict(), "note": INTERVAL_NOTE, "multiplicity_adjusted": False}
        result["arm_confidence_intervals"] = {
            base.arm_id: bootstrap_mean_ci(a_vals, confidence_level=policy.confidence_level, resamples=policy.resamples, analysis_seed=policy.analysis_seed, metric=metric, arm_a=base.arm_id).to_dict(),
            chal.arm_id: bootstrap_mean_ci(b_vals, confidence_level=policy.confidence_level, resamples=policy.resamples, analysis_seed=policy.analysis_seed, metric=metric, arm_a=chal.arm_id).to_dict(),
        }
        test_name = select_difference_test(kind, base.repetitions, chal.repetitions)
        effect: dict[str, Any] = {}
        counts = None
        if n < 2:
            test_dict = None
            result["unavailable"]["p_value"] = "insufficient_items"
            result["warnings"].append({"code": "insufficient_items", "message": f"Only {n} aligned item(s); no inference."})
        elif test_name == "mcnemar_exact":
            counts = binary_paired_counts(a_vals, b_vals)
            test_dict = mcnemar_exact(counts.b, counts.c).to_dict()
            effect["binary"] = counts.to_dict()
        elif test_name == "sign_flip":
            test_dict = sign_flip_test(diffs, exact_max=policy.exact_permutation_max, permutations=policy.permutations, **seed_kwargs).to_dict()
        else:
            test_dict = None
            result["unavailable"]["p_value"] = "unsupported_statistical_procedure"
            result["warnings"].append({"code": "unsupported_statistical_procedure", "message": f"Metric '{metric}' has score_kind '{kind}'; only a confidence interval is reported."})
        result["test"] = test_dict
        result["p_value"] = test_dict.get("p_value") if test_dict else None
        result["adjusted_p_value"] = None
        if n:
            effect["continuous"] = paired_effects(diffs).to_dict()
        result["effect_size"] = effect or None
        if n >= 2:
            if n < SMALL_SAMPLE_THRESHOLDS["very_small_sample_below"]:
                result["warnings"].append({"code": "very_small_sample", "message": f"n = {n} aligned items (< 10)."})
            elif n < SMALL_SAMPLE_THRESHOLDS["low_power_wide_intervals_below"]:
                result["warnings"].append({"code": "low_power_wide_intervals", "message": f"n = {n} aligned items (< 30); intervals are wide and power is low."})
        if policy.min_practical_difference is not None:
            ci_d = result["confidence_interval"]
            result["practical_classification"] = classify_practical_importance(ci_d["lower"], ci_d["upper"], policy.min_practical_difference)
            if result["practical_classification"] is None:
                result["unavailable"]["practical_classification"] = "confidence_interval_unavailable"
        else:
            result["practical_classification"] = None
            result["unavailable"]["practical_classification"] = "no_min_practical_difference"
        if policy.equivalence_margin is not None:
            result["_equivalence_obj"] = self._equivalence(kind, base, chal, a_vals, b_vals, diffs, counts, policy, seed_kwargs)
        else:
            result["equivalence"] = None
            result["unavailable"]["equivalence"] = "no_equivalence_margin"
        result["breakdowns"] = {
            "by_category": self._breakdown(rows, "category", metric, base, chal, policy, inferential=True),
            "by_difficulty": self._breakdown(rows, "difficulty", metric, base, chal, policy, inferential=True),
        }
        return result

    @staticmethod
    def _coverage(obs: dict[str, ItemObservation], universe: list[str]) -> dict[str, Any]:
        valid = missing = invalid = unavailable = 0
        for rid in universe:
            o = obs.get(rid)
            if o is None:
                unavailable += 1
                continue
            statuses = [s for s in o.statuses if s != "absent"]
            if any(s == VALID for s in statuses):
                valid += 1
            elif any(s == MISSING for s in statuses):
                missing += 1
            elif any(s == INVALID for s in statuses):
                invalid += 1
            else:
                unavailable += 1
        total = len(universe)
        return {
            "items": total,
            "with_valid_value": valid,
            "missing": missing,
            "invalid": invalid,
            "metric_unavailable": unavailable,
            "coverage_rate": (valid / total) if total else None,
        }

    def _equivalence(self, kind, base, chal, a_vals, b_vals, diffs, counts, policy, seed_kwargs) -> EquivalenceResult:
        margin = float(policy.equivalence_margin)
        n = len(diffs)
        if kind == "binary" and base.repetitions == 1 and chal.repetitions == 1:
            if n < 1:
                return _not_established("exact_unconditional_score", margin, policy.alpha, n, "insufficient_items")
            if not 0.0 < margin < 1.0:
                return _not_established("exact_unconditional_score", margin, policy.alpha, n, "margin_outside_unit_interval")
            counts = counts or binary_paired_counts(a_vals, b_vals)
            return tost_paired_binary_exact(counts.b, counts.c, counts.n, margin, policy.alpha)
        if kind == "unknown":
            return _not_established("none", margin, policy.alpha, n, "unsupported_statistical_procedure")
        return tost_bootstrap_t(diffs, margin, policy.alpha, analysis_seed=seed_kwargs["analysis_seed"], metric=seed_kwargs["metric"], arm_a=seed_kwargs["arm_a"], arm_b=seed_kwargs["arm_b"])

    def _breakdown(self, rows, key, metric, base, chal, policy, inferential: bool) -> dict[str, Any]:
        groups: dict[str, list[dict[str, Any]]] = {}
        for r in rows:
            groups.setdefault(str(r[key]), []).append(r)
        out = {}
        for label in sorted(groups):
            members = groups[label]
            d = [r["difference"] for r in members]
            n = len(members)
            entry: dict[str, Any] = {
                "sample_count": n,
                "baseline_mean": math.fsum(r["baseline"] for r in members) / n,
                "challenger_mean": math.fsum(r["challenger"] for r in members) / n,
                "point_estimate": math.fsum(d) / n,
            }
            if inferential:
                ci = paired_bootstrap_ci(d, confidence_level=policy.confidence_level, resamples=policy.resamples, analysis_seed=policy.analysis_seed, metric=f"{metric}|{key}={label}", arm_a=base.arm_id, arm_b=chal.arm_id)
                entry["confidence_interval"] = ci.to_dict()
                entry["p_value"] = None
                entry["adjusted_p_value"] = None
                if policy.stratum_tests and n >= 2:
                    kind = metric_score_kind(metric)
                    test_name = select_difference_test(kind, base.repetitions, chal.repetitions)
                    if test_name == "mcnemar_exact":
                        c = binary_paired_counts([r["baseline"] for r in members], [r["challenger"] for r in members])
                        t = mcnemar_exact(c.b, c.c).to_dict()
                    elif test_name == "sign_flip":
                        t = sign_flip_test(d, exact_max=policy.exact_permutation_max, permutations=policy.permutations, analysis_seed=policy.analysis_seed, metric=f"{metric}|{key}={label}", arm_a=base.arm_id, arm_b=chal.arm_id).to_dict()
                    else:
                        t = None
                    entry["test"] = t
                    entry["p_value"] = t.get("p_value") if t else None
                else:
                    entry["descriptive_only"] = True
            out[label] = entry
        return out

    def _failure_rates(self, base, chal, metric, universe, policy, exploratory) -> dict[str, Any]:
        obs_a = collect_item_scores(base.evaluations, metric)
        obs_b = collect_item_scores(chal.evaluations, metric)
        fa, fb, used = [], [], []
        for rid in universe:
            oa, ob = obs_a.get(rid), obs_b.get(rid)
            sa = oa.failure_share if oa else None
            sb = ob.failure_share if ob else None
            if sa is None or sb is None:
                continue
            fa.append(sa)
            fb.append(sb)
            used.append(rid)
        n = len(used)
        diffs = [b - a for a, b in zip(fa, fb, strict=True)]
        out: dict[str, Any] = {
            "baseline": base.arm_id,
            "challenger": chal.arm_id,
            "definition": "failure = missing or invalid response (execution failures produce missing responses)",
            "sample_count": n,
            "failure_rate": {
                base.arm_id: (math.fsum(fa) / n) if n else None,
                chal.arm_id: (math.fsum(fb) / n) if n else None,
            },
            "difference": (math.fsum(diffs) / n) if n else None,
            "items_failed_only_in_baseline": [r for r, a, b in zip(used, fa, fb, strict=True) if a > 0 and b == 0],
            "items_failed_only_in_challenger": [r for r, a, b in zip(used, fa, fb, strict=True) if b > 0 and a == 0],
        }
        if exploratory:
            out["confidence_interval"] = None
            out["test"] = None
            out["p_value"] = None
            out["unavailable"] = {"inference": EXPLORATORY_REASON_CODE}
            return out
        seed = {"analysis_seed": policy.analysis_seed, "metric": "failure_rate", "arm_a": base.arm_id, "arm_b": chal.arm_id}
        out["confidence_interval"] = paired_bootstrap_ci(diffs, confidence_level=policy.confidence_level, resamples=policy.resamples, **seed).to_dict()
        if n < 2:
            out["test"] = None
        elif base.repetitions == 1 and chal.repetitions == 1:
            counts = binary_paired_counts(fa, fb)
            out["test"] = mcnemar_exact(counts.b, counts.c).to_dict()
            out["discordance"] = counts.to_dict()
        else:
            out["test"] = sign_flip_test(diffs, exact_max=policy.exact_permutation_max, permutations=policy.permutations, **seed).to_dict()
        out["p_value"] = out["test"].get("p_value") if out["test"] else None
        out["adjusted_p_value"] = None
        return out

    def _apply_families(self, comparisons, failure_sections, policy, exploratory) -> dict[str, Any]:
        families: dict[str, Any] = {}
        if exploratory:
            for comp in comparisons:
                comp.pop("_equivalence_obj", None)
            return {"difference": None, "equivalence": None, "failure_rate": None, "note": "No inferential families in exploratory mode."}
        members: list[tuple[str, dict[str, Any], str | None]] = []
        for comp in comparisons:
            if comp.get("p_value") is not None:
                members.append((comp["comparison_id"], comp, None))
            if policy.stratum_tests:
                for bkey in ("by_category", "by_difficulty"):
                    for label, entry in comp["breakdowns"][bkey].items():
                        if entry.get("p_value") is not None:
                            members.append((f"{comp['comparison_id']}|{bkey}={label}", entry, None))
        diff_family_id = "difference"
        if members:
            adjusted = adjust_p_values([m[1]["p_value"] for m in members], policy.correction)
            for (_, target, _), adj in zip(members, adjusted.adjusted, strict=True):
                target["adjusted_p_value"] = adj
            correction_warnings = [w.to_dict() for w in adjusted.warnings]
        else:
            correction_warnings = []
        for comp in comparisons:
            if comp.get("p_value") is None:
                comp["unavailable"].setdefault("adjusted_p_value", comp["unavailable"].get("p_value", "no_test"))
            comp["family"] = {"id": diff_family_id, "size": len(members), "correction": policy.correction}
            comp["interpretation"] = self._interpretation(comp, policy)
        families["difference"] = {
            "id": diff_family_id,
            "correction": policy.correction,
            "size": len(members),
            "members": [m[0] for m in members],
            "warnings": correction_warnings,
        }
        fr = [s for s in failure_sections if s.get("p_value") is not None]
        if fr:
            adj = adjust_p_values([s["p_value"] for s in fr], "holm")
            for s, a in zip(fr, adj.adjusted, strict=True):
                s["adjusted_p_value"] = a
        families["failure_rate"] = {
            "id": "failure_rate",
            "correction": "holm",
            "size": len(fr),
            "members": [f"{s['baseline']}|{s['challenger']}" for s in fr],
        }
        eq_members = [c for c in comparisons if "_equivalence_obj" in c]
        if eq_members:
            results = [c.pop("_equivalence_obj") for c in eq_members]
            adjusted_eq = equivalence_family_holm(results, policy.alpha, family_id="equivalence")
            n_family = adjusted_eq[0].family.get("size", 0) if adjusted_eq else 0
            for comp, eq in zip(eq_members, adjusted_eq, strict=True):
                d = eq.to_dict()
                if d.get("interval"):
                    d["interval"]["label"] = (
                        f"{int(round((1 - 2 * policy.alpha) * 100))}% interval for this comparison "
                        f"alone (not adjusted for the {n_family} comparisons in this analysis). "
                        "The familywise equivalence decision uses the Holm-adjusted p-value."
                    )
                if d.get("result_unadjusted") != d.get("familywise_result"):
                    d["decision_disagreement"] = (
                        "The unadjusted and familywise equivalence decisions differ; the "
                        "familywise (Holm-adjusted) decision is the headline."
                    )
                comp["equivalence"] = d
            families["equivalence"] = {
                "id": "equivalence",
                "correction": "holm",
                "size": n_family,
                "members": [c["comparison_id"] for c in eq_members if c["equivalence"].get("family", {}).get("member")],
            }
        else:
            families["equivalence"] = None
        return families

    @staticmethod
    def _interpretation(comp: dict[str, Any], policy: ComparisonPolicy) -> str:
        ci = comp.get("confidence_interval") or {}
        level = f"{int(round(policy.confidence_level * 100))}%"
        adj = comp.get("adjusted_p_value")
        if adj is None:
            return (
                f"No test of no difference is available ({comp['unavailable'].get('p_value', 'n/a')}). "
                f"{level} CI [{_fmt(ci.get('lower'))}, {_fmt(ci.get('upper'))}]."
            )
        if adj <= policy.alpha:
            return (
                f"Statistically significant difference (adjusted p = {_fmt(adj)}; "
                f"{policy.correction} correction). {level} CI [{_fmt(ci.get('lower'))}, {_fmt(ci.get('upper'))}]."
            )
        return NOT_SIGNIFICANT_TEMPLATE.format(
            p=_fmt(adj), level=level, lo=_fmt(ci.get("lower")), hi=_fmt(ci.get("upper"))
        )


def _not_established(procedure: str, margin: float, alpha: float, n: int, reason: str) -> EquivalenceResult:
    return EquivalenceResult(
        procedure=procedure,
        margin=margin,
        alpha=alpha,
        sample_count=n,
        estimate=None,
        p_lower=None,
        p_upper=None,
        p_tost=None,
        p_value_method="none",
        result_unadjusted="not_established",
        interval=None,
        reason=reason,
        descriptive_only=True,
    )
