"""Unit tests for OpenBench AI dataset catalog, filtering, and statistics (v0.2.0)."""

from __future__ import annotations

from pathlib import Path

import pytest

from openbench.catalog import DatasetCatalog
from openbench.dataset import DatasetLoader
from openbench.exceptions import EmptyDatasetError


def test_catalog_discovers_all_ten_builtin_categories() -> None:
    """DatasetCatalog discovers all 10 built-in dataset categories totaling 200 examples."""
    repo_root = Path(__file__).resolve().parent.parent
    catalog = DatasetCatalog(repo_root / "datasets")
    entries = catalog.list_datasets()

    assert len(entries) == 10
    total_records = sum(e.total_records for e in entries)
    assert total_records == 200

    discovered_categories = {cat for e in entries for cat in e.categories}
    expected_categories = {
        "reasoning",
        "mathematics",
        "coding",
        "debugging",
        "instruction_following",
        "factual_knowledge",
        "safety",
        "multilingual",
        "summarization",
        "structured_output",
    }
    assert discovered_categories == expected_categories


def test_catalog_filtering_by_category_and_difficulty() -> None:
    """DatasetCatalog and Dataset.filter accurately filter by category, difficulty, tag, and language."""
    repo_root = Path(__file__).resolve().parent.parent
    catalog = DatasetCatalog(repo_root / "datasets")

    math_entries = catalog.list_datasets(category="mathematics")
    assert len(math_entries) == 1
    assert math_entries[0].total_records == 20

    multilingual_ds = DatasetLoader.load(
        repo_root / "datasets" / "multilingual" / "sample.jsonl",
        languages=["es"],
    )
    assert len(multilingual_ds) >= 2
    assert all(r.language == "es" for r in multilingual_ds.records)

    with pytest.raises(EmptyDatasetError):
        multilingual_ds.filter(languages=["nonexistent_lang_code"])


def test_catalog_inspect_and_compute_stats() -> None:
    """DatasetCatalog.inspect_dataset and compute_stats return checksums, lengths, and provenance."""
    repo_root = Path(__file__).resolve().parent.parent
    factual_path = repo_root / "datasets" / "factual_knowledge" / "sample.jsonl"

    inspection = DatasetCatalog.inspect_dataset(factual_path, sample_limit=2)
    assert inspection["total_records"] == 20
    assert len(inspection["sample_records"]) == 2
    assert len(inspection["checksum"]) == 64

    stats = DatasetCatalog.compute_stats(repo_root / "datasets")
    assert stats["files_count"] == 10
    assert stats["total_records"] == 200
    assert stats["records_with_structured_validation"] == 20
    assert stats["records_with_rationale"] >= 8
    assert stats["prompt_char_length"]["min"] > 0
