"""Dataset loading, validation, filtering, and checksum calculation for OpenBench AI."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from openbench.exceptions import DatasetValidationError, EmptyDatasetError

REQUIRED_RECORD_FIELDS = ("id", "category", "prompt", "expected_answer")
ALLOWED_DIFFICULTIES = {"easy", "medium", "hard"}
ALLOWED_FIELD_TYPES = {"str", "int", "float", "bool", "list", "dict", "number", "string", "integer", "boolean", "array", "object"}


@dataclass(frozen=True)
class DatasetRecord:
    """Represents a single standardized evaluation item."""

    id: str
    category: str
    prompt: str
    expected_answer: str
    metadata: dict[str, Any] = field(default_factory=dict)
    source_file: str | None = None
    line_number: int | None = None

    @property
    def difficulty(self) -> str:
        """Return the difficulty level from metadata, defaulting to 'unspecified'."""
        val = self.metadata.get("difficulty", "unspecified")
        return str(val) if val is not None else "unspecified"

    @property
    def tags(self) -> list[str]:
        """Return list of tags from metadata."""
        raw_tags = self.metadata.get("tags", [])
        if isinstance(raw_tags, list):
            return [str(tag) for tag in raw_tags]
        return []

    @property
    def keywords(self) -> list[str]:
        """Return expected keywords for keyword coverage evaluation.

        If `metadata.keywords` is explicitly provided as a non-empty list, use it.
        Otherwise, fall back to non-empty tokens extracted from `expected_answer`.
        """
        raw_keywords = self.metadata.get("keywords")
        if isinstance(raw_keywords, list) and raw_keywords:
            return [str(k) for k in raw_keywords if str(k).strip()]
        cleaned = self.expected_answer.strip()
        if not cleaned:
            return []
        return [cleaned]

    @property
    def language(self) -> str:
        """Return ISO language code from metadata, defaulting to 'en'."""
        val = self.metadata.get("language", "en")
        return str(val).strip() if val is not None and str(val).strip() else "en"

    @property
    def source(self) -> str:
        """Return dataset record provenance/source from metadata, defaulting to 'original'."""
        val = self.metadata.get("source", "original")
        return str(val).strip() if val is not None and str(val).strip() else "original"

    @property
    def record_version(self) -> str:
        """Return record-level version from metadata, defaulting to '0.1.0'."""
        val = self.metadata.get("version", "0.1.0")
        return str(val).strip() if val is not None and str(val).strip() else "0.1.0"

    @property
    def rationale(self) -> str | None:
        """Return optional evaluation rationale (used especially in safety datasets)."""
        val = self.metadata.get("rationale")
        return str(val).strip() if val is not None and str(val).strip() else None

    def to_dict(self) -> dict[str, Any]:
        """Serialize record to a plain dictionary."""
        return {
            "id": self.id,
            "category": self.category,
            "prompt": self.prompt,
            "expected_answer": self.expected_answer,
            "metadata": dict(self.metadata),
        }


@dataclass
class Dataset:
    """Represents a validated collection of DatasetRecord items."""

    name: str
    records: list[DatasetRecord]
    source_path: str
    version: str = "0.2.0"

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self):
        return iter(self.records)

    @property
    def categories(self) -> list[str]:
        """Return sorted unique categories present in the dataset."""
        return sorted({record.category for record in self.records})

    @property
    def difficulties(self) -> list[str]:
        """Return sorted unique difficulties present in the dataset."""
        return sorted({record.difficulty for record in self.records})

    @property
    def languages(self) -> list[str]:
        """Return sorted unique language codes present in the dataset."""
        return sorted({record.language for record in self.records})

    @property
    def record_map(self) -> dict[str, DatasetRecord]:
        """Return a mapping of record ID to DatasetRecord."""
        return {record.id: record for record in self.records}

    @property
    def checksum(self) -> str:
        """Compute a deterministic SHA-256 checksum across all records in the dataset."""
        hasher = hashlib.sha256()
        for rec in self.records:
            canonical = json.dumps(rec.to_dict(), sort_keys=True, ensure_ascii=False)
            hasher.update(canonical.encode("utf-8"))
            hasher.update(b"\n")
        return hasher.hexdigest()

    def filter(
        self,
        categories: list[str] | None = None,
        difficulties: list[str] | None = None,
        tags: list[str] | None = None,
        languages: list[str] | None = None,
    ) -> Dataset:
        """Return a new Dataset filtered by the specified metadata criteria."""
        cat_set = {c.strip().lower() for c in categories if c.strip()} if categories else None
        diff_set = {d.strip().lower() for d in difficulties if d.strip()} if difficulties else None
        tag_set = {t.strip().lower() for t in tags if t.strip()} if tags else None
        lang_set = {lg.strip().lower() for lg in languages if lg.strip()} if languages else None

        filtered: list[DatasetRecord] = []
        for rec in self.records:
            if cat_set is not None and rec.category.lower() not in cat_set:
                continue
            if diff_set is not None and rec.difficulty.lower() not in diff_set:
                continue
            if tag_set is not None:
                rec_tags = {t.lower() for t in rec.tags}
                if not (rec_tags & tag_set):
                    continue
            if lang_set is not None and rec.language.lower() not in lang_set:
                continue
            filtered.append(rec)

        if not filtered:
            raise EmptyDatasetError(
                f"Applying metadata filters resulted in 0 matching records for dataset '{self.name}'",
                file_path=self.source_path,
            )

        return Dataset(
            name=self.name,
            records=filtered,
            source_path=self.source_path,
            version=self.version,
        )

    def summary(self) -> dict[str, Any]:
        """Return summary statistics for the dataset (preserving v0.1.0 keys and adding v0.2.0 metadata)."""
        category_counts: dict[str, int] = {}
        difficulty_counts: dict[str, int] = {}
        language_counts: dict[str, int] = {}
        tag_counts: dict[str, int] = {}

        for rec in self.records:
            category_counts[rec.category] = category_counts.get(rec.category, 0) + 1
            diff = rec.difficulty
            difficulty_counts[diff] = difficulty_counts.get(diff, 0) + 1
            lang = rec.language
            language_counts[lang] = language_counts.get(lang, 0) + 1
            for tag in rec.tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1

        return {
            "name": self.name,
            "version": self.version,
            "source_path": self.source_path,
            "total_records": len(self.records),
            "categories": category_counts,
            "difficulties": difficulty_counts,
            "languages": language_counts,
            "tags": tag_counts,
            "checksum": self.checksum,
        }


class DatasetLoader:
    """Loads and validates JSONL evaluation datasets from files or directories."""

    @staticmethod
    def load(
        path: str | Path,
        dataset_name: str | None = None,
        categories: list[str] | None = None,
        difficulties: list[str] | None = None,
        tags: list[str] | None = None,
        languages: list[str] | None = None,
    ) -> Dataset:
        """Load a dataset from either a single JSONL file or a directory of JSONL files."""
        target = Path(path)
        if not target.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {target}")

        if target.is_dir():
            ds = DatasetLoader.load_directory(target, dataset_name=dataset_name)
        else:
            ds = DatasetLoader.load_file(target, dataset_name=dataset_name)

        if any(x is not None for x in (categories, difficulties, tags, languages)):
            return ds.filter(
                categories=categories,
                difficulties=difficulties,
                tags=tags,
                languages=languages,
            )
        return ds

    @staticmethod
    def load_file(file_path: str | Path, dataset_name: str | None = None) -> Dataset:
        """Load and validate a single JSONL dataset file."""
        target = Path(file_path)
        if not target.exists():
            raise FileNotFoundError(f"Dataset file does not exist: {target}")
        if target.is_dir():
            raise DatasetValidationError(
                "Expected a JSONL file path, but received a directory",
                file_path=target,
            )

        records: list[DatasetRecord] = []
        seen_ids: dict[str, int] = {}

        raw_text = target.read_text(encoding="utf-8")
        lines = raw_text.splitlines()

        if not lines or all(not line.strip() for line in lines):
            raise EmptyDatasetError(
                "Dataset file is empty or contains only whitespace",
                file_path=target,
            )

        for idx, raw_line in enumerate(lines, start=1):
            stripped = raw_line.strip()
            if not stripped:
                raise DatasetValidationError(
                    "Blank line encountered in JSONL dataset; invalid or empty lines are not ignored",
                    file_path=target,
                    line_number=idx,
                )

            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise DatasetValidationError(
                    f"Malformed JSON on line {idx}: {exc.msg}",
                    file_path=target,
                    line_number=idx,
                ) from exc

            record = DatasetLoader._validate_record(payload, file_path=target, line_number=idx)
            if record.id in seen_ids:
                first_line = seen_ids[record.id]
                raise DatasetValidationError(
                    f"Duplicate record ID '{record.id}' found on line {idx} "
                    f"(first defined on line {first_line})",
                    file_path=target,
                    line_number=idx,
                    record_id=record.id,
                )
            seen_ids[record.id] = idx
            records.append(record)

        if not records:
            raise EmptyDatasetError("Dataset contains no valid records", file_path=target)

        resolved_name = dataset_name or DatasetLoader._infer_dataset_name(target)
        # Infer dataset version from records if explicitly consistent, else default to 0.1.0 for backward compatibility
        versions = {
            str(r.metadata["version"])
            for r in records
            if "version" in r.metadata and r.metadata["version"] is not None
        }
        inferred_version = sorted(versions)[-1] if len(versions) == 1 else "0.1.0"

        return Dataset(
            name=resolved_name,
            records=records,
            source_path=str(target),
            version=inferred_version,
        )

    @staticmethod
    def load_directory(dir_path: str | Path, dataset_name: str | None = None) -> Dataset:
        """Load and combine all `.jsonl` files in a directory tree."""
        target = Path(dir_path)
        if not target.exists():
            raise FileNotFoundError(f"Dataset directory does not exist: {target}")
        if not target.is_dir():
            raise DatasetValidationError(
                "Expected a directory path, but received a file",
                file_path=target,
            )

        jsonl_files = sorted(target.rglob("*.jsonl"))
        if not jsonl_files:
            raise EmptyDatasetError(
                f"No .jsonl dataset files found in directory: {target}",
                file_path=target,
            )

        combined_records: list[DatasetRecord] = []
        seen_ids: dict[str, tuple[str, int | None]] = {}

        for jsonl_file in jsonl_files:
            sub_dataset = DatasetLoader.load_file(jsonl_file)
            for rec in sub_dataset.records:
                if rec.id in seen_ids:
                    prev_file, prev_line = seen_ids[rec.id]
                    raise DatasetValidationError(
                        f"Duplicate record ID '{rec.id}' across dataset directory "
                        f"(previously defined in {prev_file} at line {prev_line})",
                        file_path=jsonl_file,
                        line_number=rec.line_number,
                        record_id=rec.id,
                    )
                seen_ids[rec.id] = (str(jsonl_file), rec.line_number)
                combined_records.append(rec)

        resolved_name = dataset_name or target.name
        return Dataset(
            name=resolved_name,
            records=combined_records,
            source_path=str(target),
            version="0.2.0",
        )

    @staticmethod
    def _validate_record(
        payload: Any,
        file_path: Path,
        line_number: int,
    ) -> DatasetRecord:
        """Validate a parsed JSON object against the OpenBench dataset schema."""
        if not isinstance(payload, dict):
            raise DatasetValidationError(
                f"Expected a JSON object on line {line_number}, got {type(payload).__name__}",
                file_path=file_path,
                line_number=line_number,
            )

        if "id" not in payload or payload["id"] is None:
            raise DatasetValidationError(
                "Missing required field 'id'",
                file_path=file_path,
                line_number=line_number,
            )
        if not isinstance(payload["id"], str) or not payload["id"].strip():
            raise DatasetValidationError(
                "Field 'id' must be a non-empty string",
                file_path=file_path,
                line_number=line_number,
            )
        record_id = payload["id"].strip()

        for field_name in REQUIRED_RECORD_FIELDS:
            if field_name not in payload or payload[field_name] is None:
                raise DatasetValidationError(
                    f"Missing required field '{field_name}'",
                    file_path=file_path,
                    line_number=line_number,
                    record_id=record_id,
                )
            if not isinstance(payload[field_name], str) or not payload[field_name].strip():
                raise DatasetValidationError(
                    f"Field '{field_name}' must be a non-empty string",
                    file_path=file_path,
                    line_number=line_number,
                    record_id=record_id,
                )

        metadata = payload.get("metadata", {})
        if metadata is None:
            metadata = {}
        if not isinstance(metadata, dict):
            raise DatasetValidationError(
                "Field 'metadata' must be a JSON object if provided",
                file_path=file_path,
                line_number=line_number,
                record_id=record_id,
            )

        if "difficulty" in metadata and metadata["difficulty"] is not None:
            diff = metadata["difficulty"]
            if not isinstance(diff, str) or diff not in ALLOWED_DIFFICULTIES:
                raise DatasetValidationError(
                    f"Metadata field 'difficulty' must be one of {sorted(ALLOWED_DIFFICULTIES)}, got '{diff}'",
                    file_path=file_path,
                    line_number=line_number,
                    record_id=record_id,
                )

        if "tags" in metadata and metadata["tags"] is not None:
            tags = metadata["tags"]
            if not isinstance(tags, list) or not all(isinstance(t, str) and t.strip() for t in tags):
                raise DatasetValidationError(
                    "Metadata field 'tags' must be a list of non-empty strings",
                    file_path=file_path,
                    line_number=line_number,
                    record_id=record_id,
                )

        if "keywords" in metadata and metadata["keywords"] is not None:
            keywords = metadata["keywords"]
            if not isinstance(keywords, list) or not all(
                isinstance(k, str) and k.strip() for k in keywords
            ):
                raise DatasetValidationError(
                    "Metadata field 'keywords' must be a list of non-empty strings",
                    file_path=file_path,
                    line_number=line_number,
                    record_id=record_id,
                )

        # Additive v0.2.0 optional metadata validation
        for str_meta_field in ("language", "source", "version", "rationale"):
            if str_meta_field in metadata and metadata[str_meta_field] is not None:
                val = metadata[str_meta_field]
                if not isinstance(val, str) or not val.strip():
                    raise DatasetValidationError(
                        f"Metadata field '{str_meta_field}' must be a non-empty string if provided",
                        file_path=file_path,
                        line_number=line_number,
                        record_id=record_id,
                    )

        if "required_keys" in metadata and metadata["required_keys"] is not None:
            req_keys = metadata["required_keys"]
            if not isinstance(req_keys, list) or not all(
                isinstance(k, str) and k.strip() for k in req_keys
            ):
                raise DatasetValidationError(
                    "Metadata field 'required_keys' must be a list of non-empty strings",
                    file_path=file_path,
                    line_number=line_number,
                    record_id=record_id,
                )

        if "field_types" in metadata and metadata["field_types"] is not None:
            ft = metadata["field_types"]
            if not isinstance(ft, dict) or not all(
                isinstance(k, str) and isinstance(v, str) and v in ALLOWED_FIELD_TYPES
                for k, v in ft.items()
            ):
                raise DatasetValidationError(
                    f"Metadata field 'field_types' must map field names to valid types {sorted(ALLOWED_FIELD_TYPES)}",
                    file_path=file_path,
                    line_number=line_number,
                    record_id=record_id,
                )

        if "expected_schema" in metadata and metadata["expected_schema"] is not None:
            if not isinstance(metadata["expected_schema"], dict):
                raise DatasetValidationError(
                    "Metadata field 'expected_schema' must be a JSON object (JSON Schema)",
                    file_path=file_path,
                    line_number=line_number,
                    record_id=record_id,
                )

        return DatasetRecord(
            id=record_id,
            category=payload["category"].strip(),
            prompt=payload["prompt"],
            expected_answer=payload["expected_answer"],
            metadata=dict(metadata),
            source_file=str(file_path),
            line_number=line_number,
        )

    @staticmethod
    def _infer_dataset_name(file_path: Path) -> str:
        """Infer a descriptive dataset name from a file path."""
        parent = file_path.parent.name
        if parent and parent not in {".", "datasets"}:
            return f"{parent}/{file_path.stem}"
        return file_path.stem
