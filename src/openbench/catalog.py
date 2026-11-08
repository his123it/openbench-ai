"""Dataset catalog, discovery, inspection, filtering, and statistics for OpenBench AI."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openbench.dataset import DatasetLoader


@dataclass(frozen=True)
class DatasetCatalogEntry:
    """Metadata summary for a single discovered dataset file in the catalog."""

    name: str
    file_path: str
    relative_path: str
    version: str
    total_records: int
    categories: dict[str, int]
    difficulties: dict[str, int]
    languages: dict[str, int]
    tags: dict[str, int]
    checksum: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "file_path": self.file_path,
            "relative_path": self.relative_path,
            "version": self.version,
            "total_records": self.total_records,
            "categories": dict(self.categories),
            "difficulties": dict(self.difficulties),
            "languages": dict(self.languages),
            "tags": dict(self.tags),
            "checksum": self.checksum,
        }


class DatasetCatalog:
    """Manages discovery, inspection, filtering, and statistics across OpenBench datasets."""

    def __init__(self, datasets_root: str | Path | None = None) -> None:
        if datasets_root is not None:
            self.root = Path(datasets_root)
        else:
            # Default to repository's `datasets/` directory relative to package location or CWD
            pkg_repo_root = Path(__file__).resolve().parent.parent.parent / "datasets"
            cwd_datasets = Path.cwd() / "datasets"
            self.root = pkg_repo_root if pkg_repo_root.exists() else cwd_datasets

    def list_datasets(
        self,
        category: str | None = None,
        difficulty: str | None = None,
    ) -> list[DatasetCatalogEntry]:
        """Discover and validate all `.jsonl` datasets under `self.root`."""
        if not self.root.exists():
            raise FileNotFoundError(f"Datasets catalog root does not exist: {self.root}")

        files = sorted(self.root.rglob("*.jsonl")) if self.root.is_dir() else [self.root]
        entries: list[DatasetCatalogEntry] = []

        for file_path in files:
            dataset = DatasetLoader.load_file(file_path)
            if category and category.lower() not in {c.lower() for c in dataset.categories}:
                continue
            if difficulty and difficulty.lower() not in {d.lower() for d in dataset.difficulties}:
                continue

            summary = dataset.summary()
            try:
                rel_path = str(file_path.relative_to(self.root.parent))
            except ValueError:
                rel_path = str(file_path)

            entries.append(
                DatasetCatalogEntry(
                    name=dataset.name,
                    file_path=str(file_path),
                    relative_path=rel_path,
                    version=dataset.version,
                    total_records=len(dataset),
                    categories=summary["categories"],
                    difficulties=summary["difficulties"],
                    languages=summary["languages"],
                    tags=summary["tags"],
                    checksum=dataset.checksum,
                )
            )
        return entries

    @staticmethod
    def inspect_dataset(
        path: str | Path,
        category: str | None = None,
        difficulty: str | None = None,
        sample_limit: int = 3,
    ) -> dict[str, Any]:
        """Inspect a dataset file or directory, returning metadata, schema details, and sample records."""
        dataset = DatasetLoader.load(
            path,
            categories=[category] if category else None,
            difficulties=[difficulty] if difficulty else None,
        )
        summary = dataset.summary()
        samples = [rec.to_dict() for rec in dataset.records[: max(sample_limit, 0)]]
        return {
            "name": dataset.name,
            "version": dataset.version,
            "source_path": dataset.source_path,
            "checksum": dataset.checksum,
            "total_records": len(dataset),
            "categories": summary["categories"],
            "difficulties": summary["difficulties"],
            "languages": summary["languages"],
            "tags": summary["tags"],
            "record_ids": [r.id for r in dataset.records],
            "sample_records": samples,
        }

    @staticmethod
    def compute_stats(
        path: str | Path,
        category: str | None = None,
        difficulty: str | None = None,
    ) -> dict[str, Any]:
        """Compute comprehensive statistics for a dataset file or directory."""
        target = Path(path)
        dataset = DatasetLoader.load(
            target,
            categories=[category] if category else None,
            difficulties=[difficulty] if difficulty else None,
        )
        summary = dataset.summary()

        prompt_lengths = [len(r.prompt) for r in dataset.records]
        answer_lengths = [len(r.expected_answer) for r in dataset.records]
        sources: dict[str, int] = {}
        versions: dict[str, int] = {}
        with_keywords = 0
        with_rationale = 0
        with_structured_schema = 0

        for r in dataset.records:
            sources[r.source] = sources.get(r.source, 0) + 1
            versions[r.record_version] = versions.get(r.record_version, 0) + 1
            if "keywords" in r.metadata and r.metadata["keywords"]:
                with_keywords += 1
            if r.rationale:
                with_rationale += 1
            if any(k in r.metadata for k in ("required_keys", "field_types", "expected_schema")):
                with_structured_schema += 1

        file_count = len(list(target.rglob("*.jsonl"))) if target.is_dir() else 1

        return {
            "name": dataset.name,
            "source_path": dataset.source_path,
            "version": dataset.version,
            "checksum": dataset.checksum,
            "files_count": file_count,
            "total_records": len(dataset),
            "categories": summary["categories"],
            "difficulties": summary["difficulties"],
            "languages": summary["languages"],
            "tags": summary["tags"],
            "sources": sources,
            "record_versions": versions,
            "records_with_explicit_keywords": with_keywords,
            "records_with_rationale": with_rationale,
            "records_with_structured_validation": with_structured_schema,
            "prompt_char_length": {
                "min": min(prompt_lengths) if prompt_lengths else 0,
                "max": max(prompt_lengths) if prompt_lengths else 0,
                "avg": round(sum(prompt_lengths) / len(prompt_lengths), 2) if prompt_lengths else 0.0,
            },
            "expected_answer_char_length": {
                "min": min(answer_lengths) if answer_lengths else 0,
                "max": max(answer_lengths) if answer_lengths else 0,
                "avg": round(sum(answer_lengths) / len(answer_lengths), 2) if answer_lengths else 0.0,
            },
        }
