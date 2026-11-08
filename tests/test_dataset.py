"""Unit tests for OpenBench AI dataset loading and validation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from openbench.dataset import DatasetLoader, DatasetRecord
from openbench.exceptions import DatasetValidationError, EmptyDatasetError


def _write_lines(path: Path, lines: list[str]) -> Path:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_load_valid_single_jsonl_file(tmp_path: Path) -> None:
    """Valid JSONL file loads all records and populates metadata properly."""
    ds_file = tmp_path / "valid.jsonl"
    records = [
        {
            "id": "r_001",
            "category": "reasoning",
            "prompt": "What is 6 * 7?",
            "expected_answer": "42",
            "metadata": {"difficulty": "easy", "tags": ["math"], "keywords": ["42"]},
        },
        {
            "id": "r_002",
            "category": "reasoning",
            "prompt": "What is 9 + 10?",
            "expected_answer": "19",
            "metadata": {"difficulty": "easy", "tags": ["math"]},
        },
    ]
    _write_lines(ds_file, [json.dumps(r) for r in records])

    dataset = DatasetLoader.load_file(ds_file)
    assert len(dataset) == 2
    assert dataset.categories == ["reasoning"]
    assert dataset.records[0].id == "r_001"
    assert dataset.records[0].difficulty == "easy"
    assert dataset.records[0].tags == ["math"]
    assert dataset.records[0].keywords == ["42"]
    assert dataset.records[1].keywords == ["19"]


def test_load_missing_file_raises_file_not_found(tmp_path: Path) -> None:
    """Attempting to load a non-existent file raises FileNotFoundError."""
    missing = tmp_path / "does_not_exist.jsonl"
    with pytest.raises(FileNotFoundError, match="does not exist"):
        DatasetLoader.load(missing)


def test_load_empty_file_raises_empty_dataset_error(tmp_path: Path) -> None:
    """An empty or whitespace-only JSONL file raises EmptyDatasetError."""
    empty_file = tmp_path / "empty.jsonl"
    empty_file.write_text("   \n  \n", encoding="utf-8")

    with pytest.raises(EmptyDatasetError, match="empty"):
        DatasetLoader.load_file(empty_file)


def test_malformed_json_reports_exact_line_number(tmp_path: Path) -> None:
    """Malformed JSON on line 2 raises DatasetValidationError with line_number == 2."""
    bad_file = tmp_path / "malformed.jsonl"
    valid_line = json.dumps(
        {
            "id": "r_001",
            "category": "reasoning",
            "prompt": "2+2?",
            "expected_answer": "4",
        }
    )
    _write_lines(bad_file, [valid_line, "{bad json on line 2"])

    with pytest.raises(DatasetValidationError) as exc_info:
        DatasetLoader.load_file(bad_file)

    assert exc_info.value.line_number == 2
    assert "line 2" in str(exc_info.value)


def test_missing_id_field_raises_validation_error(tmp_path: Path) -> None:
    """Record without 'id' raises DatasetValidationError with exact line number."""
    bad_file = tmp_path / "missing_id.jsonl"
    record_without_id = json.dumps(
        {
            "category": "reasoning",
            "prompt": "What is 10 / 2?",
            "expected_answer": "5",
        }
    )
    _write_lines(bad_file, [record_without_id])

    with pytest.raises(DatasetValidationError) as exc_info:
        DatasetLoader.load_file(bad_file)

    assert exc_info.value.line_number == 1
    assert "Missing required field 'id'" in str(exc_info.value)


def test_missing_required_prompt_or_expected_answer_raises_error(tmp_path: Path) -> None:
    """Record missing 'expected_answer' raises DatasetValidationError referencing record ID."""
    bad_file = tmp_path / "missing_expected.jsonl"
    incomplete_record = json.dumps(
        {
            "id": "r_010",
            "category": "reasoning",
            "prompt": "What is 5 + 5?",
        }
    )
    _write_lines(bad_file, [incomplete_record])

    with pytest.raises(DatasetValidationError) as exc_info:
        DatasetLoader.load_file(bad_file)

    assert exc_info.value.line_number == 1
    assert exc_info.value.record_id == "r_010"
    assert "expected_answer" in str(exc_info.value)


def test_duplicate_id_within_file_reports_both_lines(tmp_path: Path) -> None:
    """Duplicate record IDs in a JSONL file raise DatasetValidationError with line numbers."""
    dup_file = tmp_path / "duplicates.jsonl"
    rec1 = json.dumps(
        {"id": "dup_01", "category": "coding", "prompt": "Q1", "expected_answer": "A1"}
    )
    rec2 = json.dumps(
        {"id": "dup_02", "category": "coding", "prompt": "Q2", "expected_answer": "A2"}
    )
    rec3 = json.dumps(
        {"id": "dup_01", "category": "coding", "prompt": "Q3", "expected_answer": "A3"}
    )
    _write_lines(dup_file, [rec1, rec2, rec3])

    with pytest.raises(DatasetValidationError) as exc_info:
        DatasetLoader.load_file(dup_file)

    assert exc_info.value.line_number == 3
    assert exc_info.value.record_id == "dup_01"
    assert "first defined on line 1" in str(exc_info.value)


def test_blank_line_inside_jsonl_is_not_silently_ignored(tmp_path: Path) -> None:
    """Blank line between records raises DatasetValidationError rather than being ignored."""
    blank_line_file = tmp_path / "blank_line.jsonl"
    rec1 = json.dumps(
        {"id": "r_01", "category": "coding", "prompt": "Q1", "expected_answer": "A1"}
    )
    _write_lines(blank_line_file, [rec1, "", rec1])

    with pytest.raises(DatasetValidationError) as exc_info:
        DatasetLoader.load_file(blank_line_file)

    assert exc_info.value.line_number == 2


def test_invalid_difficulty_in_metadata_raises_error(tmp_path: Path) -> None:
    """Unsupported difficulty value in metadata raises DatasetValidationError."""
    bad_meta = tmp_path / "bad_meta.jsonl"
    rec = json.dumps(
        {
            "id": "r_01",
            "category": "reasoning",
            "prompt": "Q1",
            "expected_answer": "A1",
            "metadata": {"difficulty": "impossible"},
        }
    )
    _write_lines(bad_meta, [rec])

    with pytest.raises(DatasetValidationError, match="difficulty"):
        DatasetLoader.load_file(bad_meta)


def test_load_directory_combines_files_and_detects_cross_file_duplicates(
    tmp_path: Path,
) -> None:
    """DatasetLoader.load_directory loads multiple files and detects cross-file duplicate IDs."""
    sub_a = tmp_path / "cat_a"
    sub_b = tmp_path / "cat_b"
    sub_a.mkdir()
    sub_b.mkdir()

    _write_lines(
        sub_a / "a.jsonl",
        [
            json.dumps(
                {"id": "a_1", "category": "a", "prompt": "P1", "expected_answer": "E1"}
            )
        ],
    )
    _write_lines(
        sub_b / "b.jsonl",
        [
            json.dumps(
                {"id": "b_1", "category": "b", "prompt": "P2", "expected_answer": "E2"}
            )
        ],
    )

    combined = DatasetLoader.load_directory(tmp_path)
    assert len(combined) == 2
    assert isinstance(combined.records[0], DatasetRecord)

    # Now introduce a duplicate ID in b.jsonl
    _write_lines(
        sub_b / "b.jsonl",
        [
            json.dumps(
                {"id": "a_1", "category": "b", "prompt": "P2", "expected_answer": "E2"}
            )
        ],
    )
    with pytest.raises(DatasetValidationError, match="Duplicate record ID 'a_1'"):
        DatasetLoader.load_directory(tmp_path)
