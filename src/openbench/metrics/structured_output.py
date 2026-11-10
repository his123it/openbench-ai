"""Structured Output validation metric for JSON syntax, keys, field types, and JSON Schema."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any

from openbench.dataset import DatasetRecord
from openbench.metrics.base import AggregateMetricResult, BaseMetric, MetricResult
from openbench.metrics.normalization import NormalizationConfig, normalize_text

_TYPE_CHECKERS: dict[str, Callable[[Any], bool]] = {
    "str": lambda v: isinstance(v, str),
    "string": lambda v: isinstance(v, str),
    "int": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "float": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "bool": lambda v: isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
    "list": lambda v: isinstance(v, list),
    "array": lambda v: isinstance(v, list),
    "dict": lambda v: isinstance(v, dict),
    "object": lambda v: isinstance(v, dict),
}

# Matches a single surrounding Markdown code fence (optional `json` or `JSON` language tag)
# Anchored to the entire trimmed string so arbitrary surrounding prose is never stripped.
_SINGLE_FENCE_RE = re.compile(
    r"^```(?:json|JSON)?[ \t]*\r?\n([\s\S]*?)\r?\n```$"
)


def extract_fenced_json(text: str) -> tuple[str, bool]:
    """Remove at most one valid surrounding Markdown code fence from `text`.

    Rules:
    - The entire trimmed text must begin with ```` ``` ```` (or ```` ```json ````) on the
      first line and end with ```` ``` ```` on the final line.
    - Any surrounding conversational prose before the opening fence or after the closing
      fence causes this function to return `(text, False)` without stripping.
    - If the inner body itself contains another ```` ``` ```` fence marker (multiple fences),
      returns `(text, False)` so multiple code blocks are never silently concatenated or repaired.
    """
    trimmed = text.strip()
    match = _SINGLE_FENCE_RE.match(trimmed)
    if not match:
        return trimmed, False
    inner = match.group(1)
    if "```" in inner:
        return trimmed, False
    return inner.strip(), True


class StructuredOutputMetric(BaseMetric):
    r"""Validates JSON syntax, required keys, field types, and optional JSON Schema compliance.

    Crucial Separation of Concerns:
    `StructuredOutputMetric` reports **structural validity** (`valid_structure`, `structure_score`)
    separately from **value/semantic match** (`value_match`, `semantic_score`). A response with
    valid JSON structure and all required keys/types but different values will have
    `valid_structure=True` (`structure_score=1.0`) and `value_match=False`, clearly distinguishing
    schema compliance from factual accuracy.

    Optional Fenced JSON Support (`strip_markdown_fence`):
    - Defaults to `False` (preserving strict v0.2.0 behavior).
    - When `strip_markdown_fence=True`, removes at most one valid surrounding Markdown
      code fence (e.g., ```` ```json ... ``` ````) before parsing JSON, without stripping
      arbitrary surrounding prose or repairing malformed JSON.
    """

    name = "structured_output"
    version = "0.3.1"
    score_kind = "bounded_continuous"
    description = (
        "Validates JSON parsing, expected/required keys, field types, and optional "
        "JSON Schema compliance while reporting structural validity separately from value correctness."
    )
    limitation_note = (
        "Structural validation confirms syntactic and schema compliance; a structurally valid "
        "JSON object may still contain factually incorrect field values."
    )
    is_optional = False
    optional_dependencies = ("jsonschema",)

    def __init__(
        self,
        normalization: NormalizationConfig | None = None,
        require_value_match: bool = False,
        use_jsonschema: bool = True,
        strip_markdown_fence: bool = False,
    ) -> None:
        self.normalization = normalization or NormalizationConfig()
        self.require_value_match = require_value_match
        self.use_jsonschema = use_jsonschema
        self.strip_markdown_fence = strip_markdown_fence

    def compute_item(self, record: DatasetRecord, response_text: str) -> MetricResult:
        candidate_text = response_text.strip()
        fence_stripped = False
        if self.strip_markdown_fence:
            candidate_text, fence_stripped = extract_fenced_json(response_text)

        try:
            parsed_resp = json.loads(candidate_text)
        except json.JSONDecodeError as exc:
            return MetricResult(
                metric_name=self.name,
                score=0.0,
                passed=False,
                details={
                    "valid_json": False,
                    "valid_structure": False,
                    "value_match": False,
                    "structure_score": 0.0,
                    "fence_stripped": fence_stripped,
                    "parse_error": f"{exc.msg} (line {exc.lineno}, column {exc.colno})",
                    "missing_keys": [],
                    "type_errors": [],
                    "schema_errors": [],
                },
            )

        expected_json: Any = None
        try:
            expected_json = json.loads(record.expected_answer.strip())
        except json.JSONDecodeError:
            expected_json = None

        required_keys: list[str] = list(record.metadata.get("required_keys", []))
        if not required_keys and isinstance(expected_json, dict):
            required_keys = list(expected_json.keys())

        field_types: dict[str, str] = dict(record.metadata.get("field_types", {}))
        expected_schema: dict[str, Any] | None = record.metadata.get("expected_schema")

        missing_keys: list[str] = []
        type_errors: list[str] = []
        schema_errors: list[str] = []

        if required_keys or field_types:
            if not isinstance(parsed_resp, dict):
                type_errors.append(
                    f"Expected top-level JSON object (dict), got {type(parsed_resp).__name__}"
                )
            else:
                for req_key in required_keys:
                    if req_key not in parsed_resp:
                        missing_keys.append(req_key)

                for field_name, expected_type_name in field_types.items():
                    if field_name in parsed_resp:
                        checker = _TYPE_CHECKERS.get(expected_type_name)
                        if checker and not checker(parsed_resp[field_name]):
                            actual_t = type(parsed_resp[field_name]).__name__
                            type_errors.append(
                                f"Field '{field_name}' expected type '{expected_type_name}', got '{actual_t}'"
                            )

        schema_validated = False
        if expected_schema and self.use_jsonschema:
            try:
                import jsonschema  # type: ignore[import-not-found]

                schema_validated = True
                try:
                    jsonschema.validate(instance=parsed_resp, schema=expected_schema)
                except jsonschema.ValidationError as exc:
                    schema_errors.append(exc.message)
            except ImportError:
                schema_validated = False

        total_checks = max(len(required_keys) + len(field_types) + (1 if expected_schema else 0), 1)
        failed_checks = len(missing_keys) + len(type_errors) + len(schema_errors)
        if not isinstance(parsed_resp, dict) and (required_keys or field_types):
            structure_score = 0.0
            valid_structure = False
        else:
            structure_score = max(0.0, (total_checks - failed_checks) / total_checks)
            valid_structure = failed_checks == 0

        if expected_json is not None:
            value_match = parsed_resp == expected_json
        else:
            value_match = normalize_text(candidate_text, self.normalization) == normalize_text(
                record.expected_answer, self.normalization
            )

        if self.require_value_match:
            final_score = 1.0 if (valid_structure and value_match) else (0.5 * structure_score)
            passed = valid_structure and value_match
        else:
            final_score = structure_score
            passed = valid_structure

        return MetricResult(
            metric_name=self.name,
            score=final_score,
            passed=passed,
            details={
                "valid_json": True,
                "valid_structure": valid_structure,
                "value_match": value_match,
                "structure_score": round(structure_score, 6),
                "fence_stripped": fence_stripped,
                "missing_keys": missing_keys,
                "type_errors": type_errors,
                "schema_validated": schema_validated,
                "schema_errors": schema_errors,
            },
        )

    def aggregate(
        self,
        item_results: list[MetricResult],
        total_dataset_items: int,
    ) -> AggregateMetricResult:
        base_agg = super().aggregate(item_results, total_dataset_items)
        valid_json_count = sum(1 for r in item_results if r.details.get("valid_json"))
        valid_struct_count = sum(1 for r in item_results if r.details.get("valid_structure"))
        value_match_count = sum(1 for r in item_results if r.details.get("value_match"))
        fence_stripped_count = sum(1 for r in item_results if r.details.get("fence_stripped"))
        denom = max(total_dataset_items, 1)

        details = dict(base_agg.details)
        details.update(
            {
                "valid_json_count": valid_json_count,
                "valid_structure_count": valid_struct_count,
                "invalid_structure_count": len(item_results) - valid_struct_count,
                "value_match_count": value_match_count,
                "fence_stripped_count": fence_stripped_count,
                "structural_validity_rate": round(valid_struct_count / denom, 6),
                "value_match_rate": round(value_match_count / denom, 6),
                "strip_markdown_fence": self.strip_markdown_fence,
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
            "require_value_match": self.require_value_match,
            "use_jsonschema": self.use_jsonschema,
            "strip_markdown_fence": self.strip_markdown_fence,
            "normalization": self.normalization.to_dict(),
        }
