"""Dataset loading and validation for OpenBench AI."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from openbench.exceptions import DatasetValidationError, EmptyDatasetError

REQUIRED_RECORD_FIELDS = ("id", "category", "prompt", "expected_answer")
ALLOWED_DIFFICULTIES = {"easy", "medium", "hard"}


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
        # Fallback: extract meaningful tokens or use expected_answer as a single keyword phrase
        cleaned = self.expected_answer.strip()
        if not cleaned:
            return []
        return [cleaned]

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
    version: str = "0.1.0"

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self):
        return iter(self.records)

    @property
    def categories(self) -> list[str]:
        """Return sorted unique categories present in the dataset."""
        return sorted({record.category for record in self.records})

    @property
    def record_map(self) -> dict[str, DatasetRecord]:
        """Return a mapping of record ID to DatasetRecord."""
        return {record.id: record for record in self.records}

    def summary(self) -> dict[str, Any]:
        """Return summary statistics for the dataset."""
        category_counts: dict[str, int] = {}
        difficulty_counts: dict[str, int] = {}
        for rec in self.records:
            category_counts[rec.category] = category_counts.get(rec.category, 0) + 1
            diff = rec.difficulty
            difficulty_counts[diff] = difficulty_counts.get(diff, 0) + 1

        return {
            "name": self.name,
            "version": self.version,
            "source_path": self.source_path,
            "total_records": len(self.records),
            "categories": category_counts,
            "difficulties": difficulty_counts,
        }


class DatasetLoader:
    """Loads and validates JSONL evaluation datasets from files or directories."""

    @staticmethod
    def load(path: str | Path, dataset_name: str | None = None) -> Dataset:
        """Load a dataset from either a single JSONL file or a directory of JSONL files."""
        target = Path(path)
        if not target.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {target}")

        if target.is_dir():
            return DatasetLoader.load_directory(target, dataset_name=dataset_name)
        return DatasetLoader.load_file(target, dataset_name=dataset_name)

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
        return Dataset(
            name=resolved_name,
            records=records,
            source_path=str(target),
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

        # Check ID first so we can report missing IDs clearly
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
