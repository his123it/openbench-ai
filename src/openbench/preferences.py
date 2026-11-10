"""Local-first, blinded pairwise human preference evaluation (plan section 6; v0.4.0).

Workflow:

1. :class:`PreferenceTaskBuilder` builds blind annotation tasks from two stored runs (or
   evaluation files). It writes two separate files:

   * ``tasks.jsonl``: what annotators see. It contains the prompt, two responses labelled
     only ``response_1`` / ``response_2``, optional rubric and instructions, and the
     reference answer only on request. It never contains model names, versions,
     providers, run or group IDs, latencies, token counts, seeds, record IDs or the display
     order.
   * ``tasks.key.json``: the unblinded key (model identities, record IDs and each task's
     display order). Keep it away from annotators.

   The display order is reproducible from ``preference_order_seed``.

2. :class:`AnnotationImporter` validates completed annotations (JSONL or CSV) as untrusted
   input: unknown fields, unknown task IDs, duplicates, invalid outcomes and rubric scores
   are errors; nothing is evaluated or executed.

3. :class:`PreferenceAnalyzer` joins annotations with the key and reports task-weighted win
   rates with task-clustered bootstrap CIs, tie and neither rates, position bias,
   per-category, per-difficulty and per-annotator summaries, inter-annotator agreement
   (percent agreement, Cohen's and Fleiss' kappa in display and model space) and,
   optionally, Elo ratings with their limitations.

No network access is used; no hosted service is involved.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import random
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openbench.exceptions import ConfigurationError
from openbench.stats.bootstrap import bootstrap_mean_ci

SCHEMA_VERSION = "0.4.0"
OUTCOMES = ("response_1", "response_2", "tie", "neither")
MODEL_SPACE = {"AB": {"response_1": "A", "response_2": "B"}, "BA": {"response_1": "B", "response_2": "A"}}
ANNOTATOR_ID_RE = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
ANNOTATION_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
MAX_RATIONALE_CHARS = 5_000
MAX_IMPORT_BYTES = 50 * 1024 * 1024
MAX_METADATA_KEYS = 32
MAX_METADATA_VALUE_CHARS = 1_000
REQUIRED_FIELDS = ("annotation_id", "task_id", "annotator_id", "outcome")
OPTIONAL_FIELDS = ("rubric_scores", "rationale", "confidence", "timestamp", "metadata", "schema_version")
ALLOWED_FIELDS = set(REQUIRED_FIELDS) | set(OPTIONAL_FIELDS)

DEFAULT_INSTRUCTIONS = (
    "Read the prompt and both responses. Choose 'response_1' or 'response_2' if one is better, "
    "'tie' if they are equally good, or 'neither' if both are unacceptable. Judge only the text "
    "shown; the order of the responses is random."
)
ELO_LIMITATION = (
    "Elo ratings are relative to this pool of models, prompts and annotators, depend on update "
    "order, and are not absolute measures of answer quality."
)
BLINDING_NOTE = (
    "tasks.jsonl is the blind presentation file. tasks.key.json maps tasks to models and must "
    "not be given to annotators. Model outputs are never edited; if a response itself names a "
    "model, the leakage audit warns but cannot hide it."
)


def csv_safe(value: Any) -> str:
    """CSV cell protection (see :func:`openbench.security.csv_safe`); ``None`` becomes ''."""
    from openbench.security import csv_safe as _safe

    return "" if value is None else str(_safe(value) if isinstance(value, str) else value)


def _seed_int(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------
# Rubric
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RubricCriterion:
    name: str
    minimum: float
    maximum: float
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "min": self.minimum, "max": self.maximum, "description": self.description}


def load_rubric(path: str | Path) -> tuple[list[RubricCriterion], str | None]:
    """Load a JSON rubric: ``{"criteria": [{"name", "min", "max", "description"}], "instructions"}``."""
    target = Path(path)
    if target.stat().st_size > 1_000_000:
        raise ConfigurationError("Rubric file is too large (limit 1 MB)")
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ConfigurationError(f"Rubric file {target} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict) or not isinstance(data.get("criteria"), list) or not data["criteria"]:
        raise ConfigurationError("Rubric must be an object with a non-empty 'criteria' list")
    unknown = set(data) - {"criteria", "instructions"}
    if unknown:
        raise ConfigurationError(f"Unknown rubric fields: {sorted(unknown)}")
    criteria = []
    seen = set()
    for raw in data["criteria"]:
        if not isinstance(raw, dict) or set(raw) - {"name", "min", "max", "description"}:
            raise ConfigurationError("Each rubric criterion must have only name, min, max, description")
        name = raw.get("name")
        lo, hi = raw.get("min"), raw.get("max")
        if not isinstance(name, str) or not re.match(r"^[A-Za-z0-9_.-]{1,64}$", name) or name in seen:
            raise ConfigurationError(f"Invalid or duplicate rubric criterion name: {name!r}")
        for v in (lo, hi):
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
                raise ConfigurationError(f"Rubric criterion '{name}' needs finite numeric min and max")
        if lo >= hi:
            raise ConfigurationError(f"Rubric criterion '{name}' needs min < max")
        seen.add(name)
        criteria.append(RubricCriterion(name, float(lo), float(hi), str(raw.get("description", ""))))
    instructions = data.get("instructions")
    if instructions is not None and not isinstance(instructions, str):
        raise ConfigurationError("Rubric 'instructions' must be a string")
    return criteria, instructions


# ---------------------------------------------------------------------------
# Task builder
# ---------------------------------------------------------------------------


@dataclass
class PreferenceSource:
    """One side of a preference comparison: identity plus per-record prompt/response."""

    source_id: str
    model_name: str
    model_version: str
    model_provider: str
    items: dict[str, dict[str, Any]]

    def identity(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "model_provider": self.model_provider,
        }

    @classmethod
    def from_evaluation(cls, source_id: str, evaluation: Any) -> PreferenceSource:
        items = {}
        for item in evaluation.individual_results:
            items[item.id] = {
                "prompt": item.prompt,
                "response": item.model_response,
                "reference": item.expected_answer,
                "category": item.category,
                "difficulty": item.difficulty,
                "valid": item.status == "evaluated" and item.model_response is not None and item.is_valid_response,
            }
        return cls(source_id, evaluation.model_name, evaluation.model_version, evaluation.model_provider, items)


@dataclass
class PreferenceTaskSet:
    tasks: list[dict[str, Any]]
    key: dict[str, Any]
    leakage_warnings: list[dict[str, Any]]

    def tasks_jsonl(self) -> str:
        return "".join(json.dumps(t, ensure_ascii=False, sort_keys=True) + "\n" for t in self.tasks)

    def tasks_csv(self) -> str:
        buf = io.StringIO()
        columns = ["task_id", "prompt", "response_1", "response_2"]
        if any("reference" in t for t in self.tasks):
            columns.append("reference")
        writer = csv.writer(buf, lineterminator="\n")
        writer.writerow(columns)
        for t in self.tasks:
            writer.writerow([csv_safe(t.get(c)) for c in columns])
        return buf.getvalue()


class PreferenceTaskBuilder:
    """Build blind A/B tasks and a separate identity key from two sources."""

    def __init__(
        self,
        preference_order_seed: int = 0,
        rubric: Sequence[RubricCriterion] | None = None,
        instructions: str | None = None,
        show_reference: bool = False,
    ) -> None:
        if isinstance(preference_order_seed, bool) or not isinstance(preference_order_seed, int):
            raise ConfigurationError("preference_order_seed must be an integer")
        self.seed = preference_order_seed
        self.rubric = list(rubric or [])
        self.instructions = instructions or DEFAULT_INSTRUCTIONS
        self.show_reference = show_reference

    def task_id(self, record_id: str, source_a: str, source_b: str) -> str:
        return hashlib.sha256(f"{self.seed}|{record_id}|{source_a}|{source_b}".encode()).hexdigest()[:16]

    def build(self, source_a: PreferenceSource, source_b: PreferenceSource) -> PreferenceTaskSet:
        if source_a.source_id == source_b.source_id:
            raise ConfigurationError("The two preference sources must be different")
        shared = sorted(set(source_a.items) & set(source_b.items))
        eligible = []
        excluded = {"missing_or_invalid_in_a": 0, "missing_or_invalid_in_b": 0, "not_in_both": 0, "prompt_mismatch": 0}
        excluded["not_in_both"] = len(set(source_a.items) ^ set(source_b.items))
        for rid in shared:
            ia, ib = source_a.items[rid], source_b.items[rid]
            if not ia["valid"]:
                excluded["missing_or_invalid_in_a"] += 1
            if not ib["valid"]:
                excluded["missing_or_invalid_in_b"] += 1
            if not (ia["valid"] and ib["valid"]):
                continue
            if ia["prompt"] != ib["prompt"]:
                excluded["prompt_mismatch"] += 1
                continue
            eligible.append(rid)
        if not eligible:
            raise ConfigurationError("No record has a valid response in both sources; no tasks created")
        rng = random.Random(_seed_int(f"{self.seed}|preferences"))
        tasks, key_tasks = [], []
        for rid in eligible:
            order = "AB" if rng.random() < 0.5 else "BA"
            ia, ib = source_a.items[rid], source_b.items[rid]
            first, second = (ia, ib) if order == "AB" else (ib, ia)
            tid = self.task_id(rid, source_a.source_id, source_b.source_id)
            task: dict[str, Any] = {
                "schema_version": SCHEMA_VERSION,
                "task_id": tid,
                "prompt": ia["prompt"],
                "response_1": first["response"],
                "response_2": second["response"],
                "instructions": self.instructions,
                "rubric": [c.to_dict() for c in self.rubric],
            }
            if self.show_reference:
                task["reference"] = ia["reference"]
            tasks.append(task)
            key_tasks.append(
                {
                    "task_id": tid,
                    "record_id": rid,
                    "category": ia["category"],
                    "difficulty": ia["difficulty"],
                    "model_a_id": source_a.source_id,
                    "model_b_id": source_b.source_id,
                    "display_order": order,
                }
            )
        task_ids = [t["task_id"] for t in tasks]
        if len(set(task_ids)) != len(task_ids):
            raise ConfigurationError("Task ID collision; use a different preference_order_seed")
        set_ = PreferenceTaskSet(tasks, {}, [])
        tasks_sha = _sha256_bytes(set_.tasks_jsonl().encode("utf-8"))
        set_.key = {
            "schema_version": SCHEMA_VERSION,
            "kind": "preference_task_key",
            "warning": "UNBLINDED KEY: do not share with annotators.",
            "preference_order_seed": self.seed,
            "creation_parameters": {
                "show_reference": self.show_reference,
                "rubric": [c.to_dict() for c in self.rubric],
                "instructions_sha256": _sha256_bytes(self.instructions.encode("utf-8")),
                "order_rng": "random.Random(sha256(f'{seed}|preferences')[:8]) in sorted record-ID order",
            },
            "models": {"A": source_a.identity(), "B": source_b.identity()},
            "eligible_tasks": len(tasks),
            "excluded_counts": excluded,
            "tasks_jsonl_sha256": tasks_sha,
            "tasks": key_tasks,
        }
        set_.leakage_warnings = self._leakage_audit(tasks, source_a, source_b)
        set_.key["leakage_warnings"] = set_.leakage_warnings
        return set_

    @staticmethod
    def _identity_strings(source: PreferenceSource) -> list[str]:
        out = [source.model_name, source.source_id]
        if source.model_provider not in ("unspecified", ""):
            out.append(source.model_provider)
        version = source.model_version or ""
        # Very short purely numeric versions ("1.0") would match ordinary text; skip them.
        if len(version) >= 4 or re.search(r"[A-Za-z]", version):
            out.append(version)
        return [s for s in out if s and len(s.strip()) >= 3]

    def _leakage_audit(self, tasks, source_a, source_b) -> list[dict[str, Any]]:
        needles = sorted({s.lower() for s in self._identity_strings(source_a) + self._identity_strings(source_b)})
        hits: dict[str, list[str]] = {}
        for task in tasks:
            text = " ".join(
                str(task.get(k) or "") for k in ("prompt", "response_1", "response_2", "reference")
            ).lower()
            for needle in needles:
                if needle in text:
                    hits.setdefault(needle, []).append(task["task_id"])
        return [
            {
                "code": "possible_identity_leak",
                "message": "An identity string appears in task text (model output or prompt); outputs are never edited.",
                "string_sha256": _sha256_bytes(needle.encode("utf-8"))[:16],
                "task_ids": ids,
            }
            for needle, ids in sorted(hits.items())
        ]


def write_task_set(
    task_set: PreferenceTaskSet,
    output_dir: str | Path,
    key_path: str | Path | None = None,
    *,
    write_csv: bool = False,
    overwrite: bool = False,
) -> dict[str, str]:
    """Write ``tasks.jsonl`` (and optionally ``tasks.csv``) plus the separate key file."""
    from openbench.security import validate_safe_output_path

    out_dir = validate_safe_output_path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks_path = out_dir / "tasks.jsonl"
    key_file = validate_safe_output_path(key_path) if key_path else out_dir / "tasks.key.json"
    if key_file.resolve() == tasks_path.resolve():
        raise ConfigurationError("The key file must be separate from tasks.jsonl")
    targets = [tasks_path, key_file] + ([out_dir / "tasks.csv"] if write_csv else [])
    if not overwrite:
        existing = [str(p) for p in targets if p.exists()]
        if existing:
            raise ConfigurationError(f"Refusing to overwrite existing files: {existing}")
    tasks_path.write_text(task_set.tasks_jsonl(), encoding="utf-8")
    key_file.parent.mkdir(parents=True, exist_ok=True)
    key_file.write_text(json.dumps(task_set.key, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    written = {"tasks": str(tasks_path), "key": str(key_file)}
    if write_csv:
        (out_dir / "tasks.csv").write_text(task_set.tasks_csv(), encoding="utf-8")
        written["tasks_csv"] = str(out_dir / "tasks.csv")
    return written


def load_key(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    if target.stat().st_size > MAX_IMPORT_BYTES:
        raise ConfigurationError("Key file is too large")
    try:
        key = json.loads(target.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ConfigurationError(f"Key file {target} is not valid JSON: {exc}") from exc
    if not isinstance(key, dict) or key.get("kind") != "preference_task_key" or not isinstance(key.get("tasks"), list):
        raise ConfigurationError(f"{target} is not an OpenBench preference task key")
    if key.get("schema_version") != SCHEMA_VERSION:
        raise ConfigurationError(f"Unsupported key schema_version {key.get('schema_version')!r}")
    for t in key["tasks"]:
        if not isinstance(t, dict) or t.get("display_order") not in MODEL_SPACE or not isinstance(t.get("task_id"), str):
            raise ConfigurationError("Malformed task entry in key")
    return key


# ---------------------------------------------------------------------------
# Annotation import and validation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PreferenceAnnotation:
    annotation_id: str
    task_id: str
    annotator_id: str
    outcome: str
    rubric_scores: dict[str, float] = field(default_factory=dict)
    rationale: str | None = None
    confidence: float | None = None
    timestamp: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "annotation_id": self.annotation_id,
            "task_id": self.task_id,
            "annotator_id": self.annotator_id,
            "outcome": self.outcome,
            "rubric_scores": dict(self.rubric_scores),
            "rationale": self.rationale,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
            "metadata": dict(self.metadata),
        }


@dataclass
class AnnotationValidationReport:
    annotations: list[PreferenceAnnotation]
    errors: list[dict[str, Any]]
    warnings: list[dict[str, Any]]
    records_read: int

    @property
    def valid(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "valid": self.valid,
            "records_read": self.records_read,
            "valid_annotations": len(self.annotations) if self.valid else 0,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
        }


def _parse_timestamp(value: str) -> datetime:
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


class AnnotationImporter:
    """Parse and validate annotation files (JSONL or CSV) against a task key."""

    def __init__(self, key: dict[str, Any], rubric: Sequence[RubricCriterion] | None = None,
                 annotator_roster: Iterable[str] | None = None) -> None:
        self.key = key
        self.task_ids = {t["task_id"] for t in key["tasks"]}
        if rubric is None:
            rubric = [
                RubricCriterion(c["name"], float(c["min"]), float(c["max"]), c.get("description", ""))
                for c in key.get("creation_parameters", {}).get("rubric", [])
            ]
        self.rubric = {c.name: c for c in rubric}
        self.roster = set(annotator_roster) if annotator_roster is not None else None

    @staticmethod
    def _read_rows(path: Path) -> list[tuple[int, Any]]:
        if path.stat().st_size > MAX_IMPORT_BYTES:
            raise ConfigurationError(f"Annotation file {path} exceeds the {MAX_IMPORT_BYTES} byte limit")
        text = path.read_text(encoding="utf-8-sig")
        rows: list[tuple[int, Any]] = []
        if path.suffix.lower() == ".csv":
            reader = csv.DictReader(io.StringIO(text))
            for line_no, row in enumerate(reader, start=2):
                if None in row:
                    rows.append((line_no, {"__error__": "row has more cells than the header"}))
                    continue
                rows.append((line_no, AnnotationImporter._csv_row_to_record(row)))
            return rows
        for line_no, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                rows.append((line_no, json.loads(line)))
            except json.JSONDecodeError as exc:
                rows.append((line_no, {"__error__": f"invalid JSON: {exc.msg}"}))
        return rows

    @staticmethod
    def _csv_row_to_record(row: dict[str, str]) -> dict[str, Any]:
        record: dict[str, Any] = {}
        rubric: dict[str, Any] = {}
        for col, raw in row.items():
            value = raw.strip() if isinstance(raw, str) else raw
            if col is None:
                continue
            if col.startswith("rubric:"):
                if value != "":
                    try:
                        rubric[col[len("rubric:"):]] = float(value)
                    except ValueError:
                        rubric[col[len("rubric:"):]] = value
                continue
            if value == "":
                continue
            if col == "confidence":
                try:
                    record[col] = float(value)
                except ValueError:
                    record[col] = value
            elif col == "metadata":
                try:
                    record[col] = json.loads(value)
                except json.JSONDecodeError:
                    record[col] = value
            else:
                record[col] = value
        if rubric:
            record["rubric_scores"] = rubric
        return record

    def validate_record(self, line_no: int, raw: Any, errors: list, warnings: list) -> PreferenceAnnotation | None:
        start = len(errors)

        def err(code: str, message: str) -> None:
            errors.append({"line": line_no, "code": code, "message": message})

        def failed() -> bool:
            return len(errors) > start

        if not isinstance(raw, dict):
            err("not_an_object", "Each annotation must be a JSON object")
            return None
        if "__error__" in raw:
            err("parse_error", raw["__error__"])
            return None
        unknown = sorted(set(raw) - ALLOWED_FIELDS)
        if unknown:
            err("unknown_fields", f"Unknown fields: {unknown}")
        for name in REQUIRED_FIELDS:
            if name not in raw or raw[name] in (None, ""):
                err(f"missing_{name}", f"Required field '{name}' is missing")
        if failed():
            return None
        for name in ("annotation_id", "task_id", "annotator_id", "outcome"):
            if not isinstance(raw[name], str):
                err("invalid_type", f"'{name}' must be a string")
                return None
        if not ANNOTATION_ID_RE.match(raw["annotation_id"]):
            err("invalid_annotation_id", "annotation_id must match [A-Za-z0-9_.:-]{1,128}")
        if not ANNOTATOR_ID_RE.match(raw["annotator_id"]):
            err("invalid_annotator_id", "annotator_id must match [A-Za-z0-9_.-]{1,64} (pseudonymous ID)")
        elif EMAIL_RE.match(raw["annotator_id"]) or "@" in raw["annotator_id"]:
            warnings.append({"line": line_no, "code": "annotator_id_looks_like_email", "message": "Use pseudonymous annotator IDs, not email addresses."})
        if self.roster is not None and raw["annotator_id"] not in self.roster:
            err("unknown_annotator", f"Annotator '{raw['annotator_id']}' is not in the roster")
        if raw["task_id"] not in self.task_ids:
            err("unknown_task_id", f"task_id '{raw['task_id']}' is not in the task key")
        if raw["outcome"] not in OUTCOMES:
            err("invalid_outcome", f"outcome must be one of {list(OUTCOMES)}")
        version = raw.get("schema_version")
        if version is not None and version != SCHEMA_VERSION:
            err("schema_version_mismatch", f"schema_version {version!r} is not {SCHEMA_VERSION!r}")
        rubric_scores: dict[str, float] = {}
        if "rubric_scores" in raw:
            scores = raw["rubric_scores"]
            if not isinstance(scores, dict):
                err("invalid_rubric_scores", "rubric_scores must be an object")
            else:
                for name, value in scores.items():
                    crit = self.rubric.get(name)
                    if crit is None:
                        err("unknown_rubric_criterion", f"Rubric criterion '{name}' is not in the rubric")
                    elif isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                        err("invalid_rubric_score", f"Score for '{name}' must be a finite number")
                    elif not crit.minimum <= value <= crit.maximum:
                        err("rubric_score_out_of_range", f"Score for '{name}' must lie in [{crit.minimum}, {crit.maximum}]")
                    else:
                        rubric_scores[name] = float(value)
        rationale = raw.get("rationale")
        if rationale is not None:
            if not isinstance(rationale, str):
                err("invalid_rationale", "rationale must be plain text")
            elif len(rationale) > MAX_RATIONALE_CHARS:
                err("rationale_too_long", f"rationale exceeds {MAX_RATIONALE_CHARS} characters")
        confidence = raw.get("confidence")
        if confidence is not None and (
            isinstance(confidence, bool) or not isinstance(confidence, (int, float))
            or not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0
        ):
            err("invalid_confidence", "confidence must be a number in [0, 1]")
        timestamp = raw.get("timestamp")
        if timestamp is not None:
            try:
                if not isinstance(timestamp, str):
                    raise ValueError("not a string")
                _parse_timestamp(timestamp)
            except ValueError:
                err("invalid_timestamp", "timestamp must be an ISO 8601 string")
        metadata = raw.get("metadata", {})
        if not isinstance(metadata, dict):
            err("invalid_metadata", "metadata must be an object of scalar values")
        else:
            if len(metadata) > MAX_METADATA_KEYS:
                err("metadata_too_large", f"metadata may have at most {MAX_METADATA_KEYS} keys")
            for k, v in metadata.items():
                if not isinstance(v, (str, int, float, bool)) or v is None or (isinstance(v, str) and len(v) > MAX_METADATA_VALUE_CHARS):
                    err("invalid_metadata", f"metadata value for '{k}' must be a short scalar")
                    break
        if failed():
            return None
        return PreferenceAnnotation(
            annotation_id=raw["annotation_id"],
            task_id=raw["task_id"],
            annotator_id=raw["annotator_id"],
            outcome=raw["outcome"],
            rubric_scores=rubric_scores,
            rationale=rationale,
            confidence=float(confidence) if confidence is not None else None,
            timestamp=timestamp,
            metadata=dict(metadata),
        )

    def load(self, paths: Sequence[str | Path]) -> AnnotationValidationReport:
        errors: list[dict[str, Any]] = []
        warnings: list[dict[str, Any]] = []
        annotations: list[PreferenceAnnotation] = []
        read = 0
        for path in paths:
            target = Path(path)
            if not target.is_file():
                raise FileNotFoundError(f"Annotation file does not exist: {target}")
            start = len(errors)
            for line_no, raw in self._read_rows(target):
                read += 1
                ann = self.validate_record(line_no, raw, errors, warnings)
                if ann is not None:
                    annotations.append(ann)
            for e in errors[start:]:
                e["file"] = target.name
        seen_ids: dict[str, int] = {}
        seen_pairs: dict[tuple[str, str], int] = {}
        for ann in annotations:
            seen_ids[ann.annotation_id] = seen_ids.get(ann.annotation_id, 0) + 1
            pair = (ann.annotator_id, ann.task_id)
            seen_pairs[pair] = seen_pairs.get(pair, 0) + 1
        for aid, count in sorted(seen_ids.items()):
            if count > 1:
                errors.append({"line": None, "code": "duplicate_annotation_id", "message": f"annotation_id '{aid}' appears {count} times"})
        for (annotator, task), count in sorted(seen_pairs.items()):
            if count > 1:
                errors.append({"line": None, "code": "duplicate_annotation", "message": f"Annotator '{annotator}' annotated task '{task}' {count} times"})
        return AnnotationValidationReport(annotations, errors, warnings, read)


def export_annotations_csv(annotations: Sequence[PreferenceAnnotation], rubric_names: Sequence[str] = ()) -> str:
    """CSV export of annotations with formula-injection protection."""
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    header = ["annotation_id", "task_id", "annotator_id", "outcome", "confidence", "timestamp", "rationale"]
    header += [f"rubric:{n}" for n in rubric_names]
    writer.writerow(header)
    for a in annotations:
        row = [a.annotation_id, a.task_id, a.annotator_id, a.outcome, a.confidence, a.timestamp, a.rationale]
        row += [a.rubric_scores.get(n) for n in rubric_names]
        writer.writerow([csv_safe(v) for v in row])
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Agreement
# ---------------------------------------------------------------------------


def cohens_kappa(labels_1: Sequence[str], labels_2: Sequence[str]) -> float | None:
    """Cohen's kappa for two raters over the same items; ``None`` if undefined."""
    if len(labels_1) != len(labels_2):
        raise ValueError("label sequences must have equal length")
    n = len(labels_1)
    if n < 2:
        return None
    po = sum(1 for a, b in zip(labels_1, labels_2, strict=True) if a == b) / n
    c1, c2 = Counter(labels_1), Counter(labels_2)
    pe = math.fsum((c1[k] / n) * (c2[k] / n) for k in sorted(set(c1) | set(c2)))
    if math.isclose(pe, 1.0):
        return None
    return (po - pe) / (1.0 - pe)


def fleiss_kappa(table: Sequence[Sequence[int]]) -> float | None:
    """Fleiss' kappa for an N x k table of counts with the same number of ratings per row."""
    rows = [list(r) for r in table]
    if len(rows) < 2:
        return None
    m = sum(rows[0])
    if m < 2 or any(sum(r) != m for r in rows):
        raise ValueError("every row must have the same number (>= 2) of ratings")
    n_items = len(rows)
    k = len(rows[0])
    p_j = [sum(r[j] for r in rows) / (n_items * m) for j in range(k)]
    p_i = [(sum(c * c for c in r) - m) / (m * (m - 1)) for r in rows]
    p_bar = math.fsum(p_i) / n_items
    pe = math.fsum(p * p for p in p_j)
    if math.isclose(pe, 1.0):
        return None
    return (p_bar - pe) / (1.0 - pe)


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


class PreferenceAnalyzer:
    """Analyse validated annotations joined with one or more task keys."""

    def __init__(self, keys: Sequence[dict[str, Any]], *, analysis_seed: int = 0,
                 confidence_level: float = 0.95, resamples: int = 10_000) -> None:
        self.keys = list(keys)
        self.tasks: dict[str, dict[str, Any]] = {}
        self.models: dict[str, dict[str, Any]] = {}
        for key in self.keys:
            for side in ("A", "B"):
                ident = key["models"][side]
                self.models[ident["source_id"]] = ident
            for t in key["tasks"]:
                if t["task_id"] in self.tasks:
                    raise ConfigurationError(f"Task ID {t['task_id']} appears in more than one key")
                self.tasks[t["task_id"]] = t
        self.seed = analysis_seed
        self.confidence_level = confidence_level
        self.resamples = resamples

    def _join(self, annotations: Sequence[PreferenceAnnotation]) -> list[dict[str, Any]]:
        joined = []
        for a in annotations:
            task = self.tasks.get(a.task_id)
            if task is None:
                raise ConfigurationError(f"Annotation {a.annotation_id} refers to an unknown task")
            order = task["display_order"]
            model_outcome = MODEL_SPACE[order].get(a.outcome, a.outcome)
            joined.append({**a.to_dict(), **{k: task[k] for k in ("record_id", "category", "difficulty", "model_a_id", "model_b_id", "display_order")}, "outcome_model_space": model_outcome})
        return joined

    def _win_rate(self, rows: list[dict[str, Any]], label: str) -> dict[str, Any]:
        by_task: dict[str, list[float]] = {}
        outcome_counts = Counter(r["outcome_model_space"] for r in rows)
        for r in rows:
            score = {"A": 1.0, "tie": 0.5, "B": 0.0}.get(r["outcome_model_space"])
            if score is None:
                continue
            by_task.setdefault(r["task_id"], []).append(score)
        task_ids = sorted(by_task)
        task_values = [math.fsum(by_task[t]) / len(by_task[t]) for t in task_ids]
        annotation_values = [v for t in task_ids for v in by_task[t]]
        ci = bootstrap_mean_ci(task_values, confidence_level=self.confidence_level, resamples=self.resamples,
                               analysis_seed=self.seed, metric=f"preference_win_rate|{label}")
        total = len(rows)
        return {
            "annotations": total,
            "tasks_annotated": len({r["task_id"] for r in rows}),
            "tasks_in_win_rate": len(task_ids),
            "tasks_only_neither": len({r["task_id"] for r in rows}) - len(task_ids),
            "outcome_counts": {k: outcome_counts.get(k, 0) for k in ("A", "B", "tie", "neither")},
            "tie_rate": (outcome_counts.get("tie", 0) / total) if total else None,
            "neither_rate": (outcome_counts.get("neither", 0) / total) if total else None,
            "win_rate_a_task_weighted": (math.fsum(task_values) / len(task_values)) if task_values else None,
            "win_rate_a_annotation_weighted": (math.fsum(annotation_values) / len(annotation_values)) if annotation_values else None,
            "win_rate_a_confidence_interval": {**ci.to_dict(), "method": "percentile_bootstrap_over_tasks", "clustering": "annotations are resampled with their task"},
            "definition": "task value = mean of A-win 1, tie 0.5, B-win 0 over the task's non-'neither' annotations; headline = mean over tasks",
        }

    def _position_bias(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        by_task: dict[str, list[float]] = {}
        for r in rows:
            if r["outcome"] in ("response_1", "response_2"):
                by_task.setdefault(r["task_id"], []).append(1.0 if r["outcome"] == "response_1" else 0.0)
        values = [math.fsum(v) / len(v) for _, v in sorted(by_task.items())]
        ci = bootstrap_mean_ci(values, confidence_level=self.confidence_level, resamples=self.resamples,
                               analysis_seed=self.seed, metric="preference_position_bias")
        share = (math.fsum(values) / len(values)) if values else None
        warnings = []
        if ci.lower is not None and ci.upper is not None and (ci.lower > 0.5 or ci.upper < 0.5):
            warnings.append({"code": "position_bias", "message": "The share of decisive choices for response_1 differs from 0.5 beyond its CI."})
        return {"share_choosing_response_1": share, "tasks": len(values), "confidence_interval": ci.to_dict(), "warnings": warnings}

    def _agreement(self, rows: list[dict[str, Any]], space: str) -> dict[str, Any]:
        field_name = "outcome" if space == "display" else "outcome_model_space"
        cats = list(OUTCOMES) if space == "display" else ["A", "B", "tie", "neither"]
        by_task: dict[str, dict[str, str]] = {}
        for r in rows:
            by_task.setdefault(r["task_id"], {})[r["annotator_id"]] = r[field_name]
        agree_pairs = total_pairs = 0
        for labels in by_task.values():
            vals = list(labels.values())
            for i in range(len(vals)):
                for j in range(i + 1, len(vals)):
                    total_pairs += 1
                    agree_pairs += vals[i] == vals[j]
        annotators = sorted({r["annotator_id"] for r in rows})
        pairwise = []
        for i in range(len(annotators)):
            for j in range(i + 1, len(annotators)):
                a1, a2 = annotators[i], annotators[j]
                shared = sorted(t for t, labels in by_task.items() if a1 in labels and a2 in labels)
                kappa = cohens_kappa([by_task[t][a1] for t in shared], [by_task[t][a2] for t in shared]) if len(shared) >= 2 else None
                pairwise.append({"annotators": [a1, a2], "shared_tasks": len(shared), "cohens_kappa": kappa,
                                 "unavailable_reason": None if kappa is not None else ("fewer_than_2_shared_tasks" if len(shared) < 2 else "expected_agreement_is_1")})
        counts = Counter(len(labels) for labels in by_task.values() if len(labels) >= 2)
        fleiss = None
        modal_m = None
        used = excluded = 0
        if counts:
            modal_m = max(sorted(counts), key=lambda m: (counts[m], m))
            table = []
            for labels in by_task.values():
                if len(labels) == modal_m:
                    c = Counter(labels.values())
                    table.append([c.get(k, 0) for k in cats])
            used = len(table)
            excluded = sum(1 for labels in by_task.values() if len(labels) >= 2 and len(labels) != modal_m)
            fleiss = fleiss_kappa(table) if len(table) >= 2 else None
        return {
            "space": space,
            "percent_agreement": (agree_pairs / total_pairs) if total_pairs else None,
            "annotator_pairs_compared": total_pairs,
            "cohens_kappa_pairs": pairwise,
            "fleiss_kappa": fleiss,
            "fleiss_ratings_per_task": modal_m,
            "fleiss_tasks_used": used,
            "fleiss_tasks_excluded_other_rating_counts": excluded,
        }

    def _elo(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        def order_key(r):
            ts = r.get("timestamp")
            if ts:
                return (0, _parse_timestamp(ts).isoformat(), r["annotation_id"])
            return (1, "", r["annotation_id"])

        ratings: dict[str, float] = {mid: 1000.0 for mid in sorted(self.models)}
        comparisons: Counter = Counter()
        for r in sorted(rows, key=order_key):
            outcome = r["outcome_model_space"]
            if outcome == "neither":
                continue
            a, b = r["model_a_id"], r["model_b_id"]
            score_a = {"A": 1.0, "B": 0.0, "tie": 0.5}[outcome]
            expected_a = 1.0 / (1.0 + 10 ** ((ratings[b] - ratings[a]) / 400.0))
            ratings[a] += 32.0 * (score_a - expected_a)
            ratings[b] += 32.0 * ((1.0 - score_a) - (1.0 - expected_a))
            comparisons[a] += 1
            comparisons[b] += 1
        return {
            "k_factor": 32,
            "initial_rating": 1000,
            "update_order": "annotations with a timestamp (ascending, then annotation_id), then those without (by annotation_id)",
            "ratings": {m: ratings[m] for m in sorted(ratings)},
            "comparisons_per_model": {m: comparisons.get(m, 0) for m in sorted(ratings)},
            "limitations": ELO_LIMITATION,
            "warnings": [{"code": "single_ordering", "message": "Ratings come from one update ordering; their sensitivity to order is not quantified."}],
        }

    def analyze(self, annotations: Sequence[PreferenceAnnotation], *, elo: bool = False) -> dict[str, Any]:
        rows = self._join(annotations)
        warnings: list[dict[str, Any]] = []
        annotated_tasks = {r["task_id"] for r in rows}
        unannotated = sorted(set(self.tasks) - annotated_tasks)
        annotator_counts = Counter(r["annotator_id"] for r in rows)
        if rows and max(annotator_counts.values()) > 0.5 * len(rows) and len(annotator_counts) > 1:
            warnings.append({"code": "annotator_imbalance", "message": "One annotator contributed more than 50% of annotations."})
        pairs = []
        for key in self.keys:
            a_id, b_id = key["models"]["A"]["source_id"], key["models"]["B"]["source_id"]
            key_tasks = {t["task_id"] for t in key["tasks"]}
            pair_rows = [r for r in rows if r["task_id"] in key_tasks]
            entry = {
                "model_a": key["models"]["A"],
                "model_b": key["models"]["B"],
                "tasks": len(key_tasks),
                "unannotated_tasks": len(key_tasks - annotated_tasks),
                "overall": self._win_rate(pair_rows, f"{a_id}|{b_id}"),
                "by_category": {},
                "by_difficulty": {},
            }
            for dim in ("category", "difficulty"):
                for label in sorted({r[dim] for r in pair_rows}):
                    entry[f"by_{dim}"][label] = self._win_rate([r for r in pair_rows if r[dim] == label], f"{a_id}|{b_id}|{dim}={label}")
            pairs.append(entry)
        per_annotator = {}
        for annotator in sorted(annotator_counts):
            arows = [r for r in rows if r["annotator_id"] == annotator]
            c = Counter(r["outcome_model_space"] for r in arows)
            decisive = c.get("A", 0) + c.get("B", 0) + c.get("tie", 0)
            per_annotator[annotator] = {
                "annotations": len(arows),
                "tasks": len({r["task_id"] for r in arows}),
                "outcome_counts": {k: c.get(k, 0) for k in ("A", "B", "tie", "neither")},
                "win_rate_a_annotation_weighted": ((c.get("A", 0) + 0.5 * c.get("tie", 0)) / decisive) if decisive else None,
            }
        rubric_summary: dict[str, Any] = {}
        for r in rows:
            for name, value in (r.get("rubric_scores") or {}).items():
                rubric_summary.setdefault(name, []).append(value)
        result = {
            "schema_version": SCHEMA_VERSION,
            "analysis": "preference_analysis",
            "unblinded": True,
            "note": "This analysis joins annotations with the unblinded key; do not share it with annotators who are still annotating.",
            "analysis_seed": self.seed,
            "parameters": {"confidence_level": self.confidence_level, "resamples": self.resamples},
            "annotation_count": len(rows),
            "annotator_count": len(annotator_counts),
            "task_count": len(self.tasks),
            "unannotated_tasks": len(unannotated),
            "unannotated_task_ids": unannotated,
            "outcome_distribution_display_space": {k: sum(1 for r in rows if r["outcome"] == k) for k in OUTCOMES},
            "pairs": pairs,
            "position_bias": self._position_bias(rows),
            "per_annotator": per_annotator,
            "agreement": {"display_space": self._agreement(rows, "display"), "model_space": self._agreement(rows, "model"),
                          "note": ("Observed agreement is identical in both spaces because all annotators of a task share its display order; "
                                   "kappa can differ because category marginals change when outcomes are relabelled per task.")},
            "rubric_scores": {name: {"count": len(v), "mean": math.fsum(v) / len(v), "min": min(v), "max": max(v)} for name, v in sorted(rubric_summary.items())},
            "methodology": {
                "task_selection": "records with a valid response from both models, sorted by record ID",
                "blinding": BLINDING_NOTE,
                "win_rate": "task-weighted headline with a percentile bootstrap over tasks (annotations clustered within tasks); annotation-weighted also reported",
                "neither": "kept in counts, excluded from win rates",
                "agreement": "percent agreement over annotator pairs per task; Cohen's kappa per annotator pair (>= 2 shared tasks); Fleiss' kappa over tasks with the modal number of ratings (>= 2)",
            },
            "warnings": warnings,
        }
        result["elo"] = self._elo(rows) if elo else None
        from openbench.provenance import provenance_block

        result["provenance"] = provenance_block(
            parameters={"confidence_level": self.confidence_level, "resamples": self.resamples, "elo": elo},
            inputs=[{"key_tasks_jsonl_sha256": k.get("tasks_jsonl_sha256"), "preference_order_seed": k.get("preference_order_seed")} for k in self.keys],
            analysis_seed=self.seed,
        )
        return result
