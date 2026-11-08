"""YAML evaluation and model execution configuration profile loader and validator (v0.3.0)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from openbench.adapters.base import PROVIDER_REGISTRY, GenerationConfig
from openbench.dataset import ALLOWED_DIFFICULTIES
from openbench.evaluator import DEFAULT_METRIC_NAMES, EvaluationConfig
from openbench.exceptions import ConfigurationError
from openbench.execution import ExecutionConfig, RetryPolicy
from openbench.metrics import METRIC_REGISTRY
from openbench.metrics.normalization import NormalizationConfig
from openbench.performance import PricingConfig
from openbench.security import redact_sensitive_data

ALLOWED_TOP_LEVEL_KEYS = {
    "name",
    "description",
    "dataset",
    "metrics",
    "metric_options",
    "normalization",
    "options",
    "output_format",
    # Additive v0.3.0 sections
    "provider",
    "generation",
    "execution",
    "performance",
    "privacy",
}
ALLOWED_DATASET_KEYS = {"path", "category", "categories", "language", "languages", "tags", "ids_file", "selected_ids"}
ALLOWED_OPTION_KEYS = {
    "difficulty",
    "difficulties",
    "strict_unknown_ids",
    "record_history",
    "repeat_count",
    "experiments_dir",
    "ids_file",
    "selected_ids",
}
ALLOWED_NORMALIZATION_KEYS = {
    "strip_whitespace",
    "case_sensitive",
    "collapse_whitespace",
    "strip_punctuation",
}
ALLOWED_PROVIDER_KEYS = {
    "name",
    "model",
    "model_version",
    "revision",
    "endpoint",
    "device",
    "allow_download",
    "allow_remote_http",
    "use_expected_answer",
    "simulated_latency_ms",
    "api_key_env",
}
ALLOWED_GENERATION_KEYS = {
    "temperature",
    "max_tokens",
    "top_p",
    "seed",
    "stop_sequences",
    "stream",
    "extra_options",
}
ALLOWED_EXECUTION_KEYS = {
    "timeout_seconds",
    "concurrency",
    "use_cache",
    "cache_dir",
    "max_attempts",
    "initial_backoff_seconds",
    "backoff_multiplier",
    "max_backoff_seconds",
    "retry_on_timeouts",
}
ALLOWED_PERFORMANCE_KEYS = {
    "estimate_missing_tokens",
    "pricing",
}
ALLOWED_PRICING_KEYS = {
    "prompt_cost_per_1k_tokens",
    "completion_cost_per_1k_tokens",
    "currency",
}
ALLOWED_PRIVACY_KEYS = {
    "privacy_mode",
}
ALLOWED_OUTPUT_FORMATS = {"json", "markdown", "md", "csv"}


@dataclass
class EvaluationProfile:
    """Represents a validated YAML evaluation and/or model execution configuration profile."""

    name: str
    dataset_path: str | None = None
    categories: list[str] | None = None
    difficulties: list[str] | None = None
    languages: list[str] | None = None
    tags: list[str] | None = None
    selected_ids: list[str] | None = None
    ids_file: str | None = None
    metrics: list[str] = field(default_factory=lambda: list(DEFAULT_METRIC_NAMES))
    metric_options: dict[str, dict[str, Any]] = field(default_factory=dict)
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)
    strict_unknown_ids: bool = False
    record_history: bool = True
    output_formats: list[str] = field(default_factory=lambda: ["json", "markdown"])
    description: str = ""
    # v0.3.0 execution & provider fields
    provider_config: dict[str, Any] = field(default_factory=dict)
    generation_config: GenerationConfig = field(default_factory=GenerationConfig)
    execution_config: ExecutionConfig = field(default_factory=ExecutionConfig)
    cache_dir: str | None = None
    experiments_dir: str | None = None
    repeat_count: int = 1

    def to_evaluation_config(
        self,
        override_metrics: list[str] | None = None,
    ) -> EvaluationConfig:
        """Convert profile into an EvaluationConfig instance, applying optional CLI overrides."""
        active_metrics = list(override_metrics) if override_metrics else list(self.metrics)
        return EvaluationConfig(
            metric_names=active_metrics,
            normalization=self.normalization,
            strict_unknown_ids=self.strict_unknown_ids,
            metric_options=dict(self.metric_options),
            categories=list(self.categories) if self.categories else None,
            difficulties=list(self.difficulties) if self.difficulties else None,
            tags=list(self.tags) if self.tags else None,
            languages=list(self.languages) if self.languages else None,
            selected_ids=list(self.selected_ids) if self.selected_ids else None,
            ids_file=self.ids_file,
            profile_name=self.name,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "dataset_path": self.dataset_path,
            "categories": self.categories,
            "difficulties": self.difficulties,
            "languages": self.languages,
            "tags": self.tags,
            "selected_ids": self.selected_ids,
            "ids_file": self.ids_file,
            "metrics": list(self.metrics),
            "metric_options": dict(self.metric_options),
            "normalization": self.normalization.to_dict(),
            "strict_unknown_ids": self.strict_unknown_ids,
            "record_history": self.record_history,
            "output_formats": list(self.output_formats),
            "provider_config": redact_sensitive_data(dict(self.provider_config)),
            "generation_config": self.generation_config.to_dict(),
            "execution_config": self.execution_config.to_dict(),
            "repeat_count": self.repeat_count,
        }


class ProfileLoader:
    """Loads and validates YAML configuration profiles."""

    @staticmethod
    def load_file(file_path: str | Path) -> EvaluationProfile:
        """Load and validate a YAML evaluation configuration profile from disk."""
        target = Path(file_path)
        if not target.exists():
            raise FileNotFoundError(f"Configuration profile file does not exist: {target}")
        if target.is_dir():
            raise ConfigurationError(f"Expected a YAML configuration file, got directory: {target}")

        try:
            import yaml  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ConfigurationError(
                "Optional dependency 'PyYAML' is required to load YAML configuration profiles. "
                "Install it with `pip install -e '.[config]'`."
            ) from exc

        raw_text = target.read_text(encoding="utf-8")
        if not raw_text.strip():
            raise ConfigurationError(f"Configuration file is empty: {target}")

        try:
            payload = yaml.safe_load(raw_text)
        except Exception as exc:
            raise ConfigurationError(f"Invalid YAML syntax in {target}: {exc}") from exc

        return ProfileLoader.validate_dict(payload, source_path=target)

    @staticmethod
    def validate_dict(
        payload: Any,
        source_path: Path | None = None,
    ) -> EvaluationProfile:
        """Validate a parsed configuration dictionary strictly, rejecting unknown fields."""
        loc = f" in {source_path}" if source_path else ""
        if not isinstance(payload, dict):
            raise ConfigurationError(f"Configuration profile{loc} must be a YAML mapping (dict)")

        unknown_top = set(payload.keys()) - ALLOWED_TOP_LEVEL_KEYS
        if unknown_top:
            raise ConfigurationError(
                f"Unknown top-level configuration field(s){loc}: {sorted(unknown_top)}. "
                f"Allowed fields: {sorted(ALLOWED_TOP_LEVEL_KEYS)}"
            )

        name = payload.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ConfigurationError(f"Configuration profile{loc} requires a non-empty string 'name'")

        description = str(payload.get("description", "")).strip()

        # 1. Dataset section
        ds_section = payload.get("dataset", {})
        if not isinstance(ds_section, dict):
            raise ConfigurationError(f"Field 'dataset'{loc} must be a mapping")
        unknown_ds = set(ds_section.keys()) - ALLOWED_DATASET_KEYS
        if unknown_ds:
            raise ConfigurationError(
                f"Unknown field(s) in 'dataset' section{loc}: {sorted(unknown_ds)}"
            )

        dataset_path = ds_section.get("path")
        if dataset_path is not None:
            if not isinstance(dataset_path, str) or not dataset_path.strip():
                raise ConfigurationError(f"Field 'dataset.path'{loc} must be a non-empty string")
            raw_ds_path = dataset_path.strip()
            if not Path(raw_ds_path).exists() and source_path is not None:
                candidate_repo_rel = source_path.resolve().parent.parent / raw_ds_path
                if candidate_repo_rel.exists():
                    dataset_path = str(candidate_repo_rel)

        categories: list[str] | None = None
        if "category" in ds_section and ds_section["category"] is not None:
            cat_val = ds_section["category"]
            categories = [cat_val] if isinstance(cat_val, str) else list(cat_val)
        elif "categories" in ds_section and ds_section["categories"] is not None:
            categories = list(ds_section["categories"])

        languages: list[str] | None = None
        if "language" in ds_section and ds_section["language"] is not None:
            lang_val = ds_section["language"]
            languages = [lang_val] if isinstance(lang_val, str) else list(lang_val)
        elif "languages" in ds_section and ds_section["languages"] is not None:
            languages = list(ds_section["languages"])

        tags: list[str] | None = None
        if "tags" in ds_section and ds_section["tags"] is not None:
            if not isinstance(ds_section["tags"], list):
                raise ConfigurationError(f"Field 'dataset.tags'{loc} must be a list of strings")
            tags = [str(t) for t in ds_section["tags"]]

        ids_file: str | None = ds_section.get("ids_file")
        selected_ids: list[str] | None = None
        if "selected_ids" in ds_section and ds_section["selected_ids"] is not None:
            if not isinstance(ds_section["selected_ids"], list):
                raise ConfigurationError(f"Field 'dataset.selected_ids'{loc} must be a list of strings")
            selected_ids = [str(i) for i in ds_section["selected_ids"]]

        # 2. Metrics section
        metrics_raw = payload.get("metrics", list(DEFAULT_METRIC_NAMES))
        if not isinstance(metrics_raw, list) or not metrics_raw:
            raise ConfigurationError(f"Field 'metrics'{loc} must be a non-empty list of metric names")
        metrics: list[str] = []
        for m in metrics_raw:
            if not isinstance(m, str) or not m.strip():
                raise ConfigurationError(f"Metric names{loc} must be non-empty strings")
            m_clean = m.strip().lower()
            if m_clean not in METRIC_REGISTRY:
                raise ConfigurationError(
                    f"Unknown metric '{m}' in profile{loc}. "
                    f"Available: {sorted(METRIC_REGISTRY.keys())}"
                )
            metrics.append(m_clean)

        metric_options = payload.get("metric_options", {})
        if not isinstance(metric_options, dict):
            raise ConfigurationError(f"Field 'metric_options'{loc} must be a mapping")

        # 3. Normalization section
        norm_raw = payload.get("normalization", {})
        if not isinstance(norm_raw, dict):
            raise ConfigurationError(f"Field 'normalization'{loc} must be a mapping")
        unknown_norm = set(norm_raw.keys()) - ALLOWED_NORMALIZATION_KEYS
        if unknown_norm:
            raise ConfigurationError(
                f"Unknown field(s) in 'normalization'{loc}: {sorted(unknown_norm)}"
            )
        norm_cfg = NormalizationConfig(
            strip_whitespace=bool(norm_raw.get("strip_whitespace", True)),
            case_sensitive=bool(norm_raw.get("case_sensitive", False)),
            collapse_whitespace=bool(norm_raw.get("collapse_whitespace", True)),
            strip_punctuation=bool(norm_raw.get("strip_punctuation", False)),
        )

        # 4. Options section
        opts_raw = payload.get("options", {})
        if not isinstance(opts_raw, dict):
            raise ConfigurationError(f"Field 'options'{loc} must be a mapping")
        unknown_opts = set(opts_raw.keys()) - ALLOWED_OPTION_KEYS
        if unknown_opts:
            raise ConfigurationError(
                f"Unknown field(s) in 'options'{loc}: {sorted(unknown_opts)}"
            )

        difficulties: list[str] | None = None
        diff_raw = opts_raw.get("difficulty", opts_raw.get("difficulties"))
        if diff_raw is not None:
            diff_list = [diff_raw] if isinstance(diff_raw, str) else list(diff_raw)
            for d in diff_list:
                if not isinstance(d, str) or d not in ALLOWED_DIFFICULTIES:
                    raise ConfigurationError(
                        f"Invalid difficulty '{d}' in profile{loc}. Allowed: {sorted(ALLOWED_DIFFICULTIES)}"
                    )
            difficulties = diff_list

        strict_unknown_ids = bool(opts_raw.get("strict_unknown_ids", False))
        record_history = bool(opts_raw.get("record_history", True))
        repeat_count = int(opts_raw.get("repeat_count", 1))
        if repeat_count < 1:
            raise ConfigurationError(f"Field 'options.repeat_count'{loc} must be >= 1")
        experiments_dir = opts_raw.get("experiments_dir")
        if "ids_file" in opts_raw and opts_raw["ids_file"] is not None:
            ids_file = str(opts_raw["ids_file"])
        if "selected_ids" in opts_raw and opts_raw["selected_ids"] is not None:
            if not isinstance(opts_raw["selected_ids"], list):
                raise ConfigurationError(f"Field 'options.selected_ids'{loc} must be a list of strings")
            selected_ids = [str(i) for i in opts_raw["selected_ids"]]

        # 5. Output formats
        out_raw = payload.get("output_format", ["json", "markdown"])
        if isinstance(out_raw, str):
            out_list = [out_raw]
        elif isinstance(out_raw, list) and out_raw:
            out_list = [str(x).strip().lower() for x in out_raw]
        else:
            raise ConfigurationError(
                f"Field 'output_format'{loc} must be a non-empty list of formats"
            )

        normalized_formats: list[str] = []
        for fmt in out_list:
            if fmt not in ALLOWED_OUTPUT_FORMATS:
                raise ConfigurationError(
                    f"Unsupported output_format '{fmt}'{loc}. Allowed: {sorted(ALLOWED_OUTPUT_FORMATS)}"
                )
            canonical_fmt = "markdown" if fmt == "md" else fmt
            if canonical_fmt not in normalized_formats:
                normalized_formats.append(canonical_fmt)

        # 6. Provider section (v0.3.0)
        provider_raw = payload.get("provider", {})
        if not isinstance(provider_raw, dict):
            raise ConfigurationError(f"Field 'provider'{loc} must be a mapping")
        unknown_prov = set(provider_raw.keys()) - ALLOWED_PROVIDER_KEYS
        if unknown_prov:
            raise ConfigurationError(
                f"Unknown field(s) in 'provider'{loc}: {sorted(unknown_prov)}"
            )
        if "name" in provider_raw and provider_raw["name"] is not None:
            p_name = str(provider_raw["name"]).strip().lower()
            if p_name not in PROVIDER_REGISTRY:
                raise ConfigurationError(
                    f"Unknown model provider '{p_name}' in profile{loc}. "
                    f"Registered providers: {sorted(PROVIDER_REGISTRY.keys())}"
                )

        # 7. Generation section (v0.3.0)
        gen_raw = payload.get("generation", {})
        if not isinstance(gen_raw, dict):
            raise ConfigurationError(f"Field 'generation'{loc} must be a mapping")
        unknown_gen = set(gen_raw.keys()) - ALLOWED_GENERATION_KEYS
        if unknown_gen:
            raise ConfigurationError(
                f"Unknown field(s) in 'generation'{loc}: {sorted(unknown_gen)}"
            )
        gen_cfg = GenerationConfig(
            temperature=float(gen_raw.get("temperature", 0.0)),
            max_tokens=int(gen_raw.get("max_tokens", 256)),
            top_p=float(gen_raw.get("top_p", 1.0)),
            seed=int(gen_raw["seed"]) if gen_raw.get("seed") is not None else None,
            stop_sequences=list(gen_raw.get("stop_sequences", [])),
            stream=bool(gen_raw.get("stream", False)),
            extra_options=dict(gen_raw.get("extra_options", {})),
        )

        # 8. Performance & Pricing section (v0.3.0)
        perf_raw = payload.get("performance", {})
        if not isinstance(perf_raw, dict):
            raise ConfigurationError(f"Field 'performance'{loc} must be a mapping")
        unknown_perf = set(perf_raw.keys()) - ALLOWED_PERFORMANCE_KEYS
        if unknown_perf:
            raise ConfigurationError(
                f"Unknown field(s) in 'performance'{loc}: {sorted(unknown_perf)}"
            )
        estimate_tokens = bool(perf_raw.get("estimate_missing_tokens", False))
        pricing_cfg: PricingConfig | None = None
        if "pricing" in perf_raw and perf_raw["pricing"] is not None:
            pr_raw = perf_raw["pricing"]
            if not isinstance(pr_raw, dict):
                raise ConfigurationError(f"Field 'performance.pricing'{loc} must be a mapping")
            unknown_pr = set(pr_raw.keys()) - ALLOWED_PRICING_KEYS
            if unknown_pr:
                raise ConfigurationError(
                    f"Unknown field(s) in 'performance.pricing'{loc}: {sorted(unknown_pr)}"
                )
            if "prompt_cost_per_1k_tokens" not in pr_raw or "completion_cost_per_1k_tokens" not in pr_raw:
                raise ConfigurationError(
                    f"Field 'performance.pricing'{loc} requires both 'prompt_cost_per_1k_tokens' "
                    "and 'completion_cost_per_1k_tokens'"
                )
            pricing_cfg = PricingConfig(
                prompt_cost_per_1k_tokens=float(pr_raw["prompt_cost_per_1k_tokens"]),
                completion_cost_per_1k_tokens=float(pr_raw["completion_cost_per_1k_tokens"]),
                currency=str(pr_raw.get("currency", "USD")),
            )

        # 9. Privacy section (v0.3.0)
        priv_raw = payload.get("privacy", {})
        if not isinstance(priv_raw, dict):
            raise ConfigurationError(f"Field 'privacy'{loc} must be a mapping")
        unknown_priv = set(priv_raw.keys()) - ALLOWED_PRIVACY_KEYS
        if unknown_priv:
            raise ConfigurationError(
                f"Unknown field(s) in 'privacy'{loc}: {sorted(unknown_priv)}"
            )
        privacy_mode = bool(priv_raw.get("privacy_mode", False))

        # 10. Execution section (v0.3.0)
        exec_raw = payload.get("execution", {})
        if not isinstance(exec_raw, dict):
            raise ConfigurationError(f"Field 'execution'{loc} must be a mapping")
        unknown_exec = set(exec_raw.keys()) - ALLOWED_EXECUTION_KEYS
        if unknown_exec:
            raise ConfigurationError(
                f"Unknown field(s) in 'execution'{loc}: {sorted(unknown_exec)}"
            )
        retry_policy = RetryPolicy(
            max_attempts=int(exec_raw.get("max_attempts", 3)),
            initial_backoff_seconds=float(exec_raw.get("initial_backoff_seconds", 0.05)),
            backoff_multiplier=float(exec_raw.get("backoff_multiplier", 2.0)),
            max_backoff_seconds=float(exec_raw.get("max_backoff_seconds", 2.0)),
            retry_on_timeouts=bool(exec_raw.get("retry_on_timeouts", True)),
        )
        exec_cfg = ExecutionConfig(
            timeout_seconds=float(exec_raw.get("timeout_seconds", 30.0)),
            concurrency=int(exec_raw.get("concurrency", 1)),
            retry_policy=retry_policy,
            use_cache=bool(exec_raw.get("use_cache", True)),
            estimate_missing_tokens=estimate_tokens,
            privacy_mode=privacy_mode,
            pricing=pricing_cfg,
        )
        cache_dir = exec_raw.get("cache_dir")

        return EvaluationProfile(
            name=name.strip(),
            dataset_path=dataset_path.strip() if dataset_path else None,
            categories=categories,
            difficulties=difficulties,
            languages=languages,
            tags=tags,
            selected_ids=selected_ids,
            ids_file=ids_file,
            metrics=metrics,
            metric_options=dict(metric_options),
            normalization=norm_cfg,
            strict_unknown_ids=strict_unknown_ids,
            record_history=record_history,
            output_formats=normalized_formats,
            description=description,
            provider_config=dict(provider_raw),
            generation_config=gen_cfg,
            execution_config=exec_cfg,
            cache_dir=str(cache_dir) if cache_dir else None,
            experiments_dir=str(experiments_dir) if experiments_dir else None,
            repeat_count=repeat_count,
        )
