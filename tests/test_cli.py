"""End-to-end unit tests for the OpenBench AI command-line interface."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from openbench.cli import main


def test_cli_version_flag(capsys: pytest.CaptureFixture[str]) -> None:
    """`openbench --version` prints version and exits with code 0."""
    with pytest.raises(SystemExit) as exc_info:
        main(["--version"])
    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "openbench 0.2.0" in captured.out


def test_cli_no_arguments_returns_exit_code_2(capsys: pytest.CaptureFixture[str]) -> None:
    """Running `openbench` without subcommands prints help and returns exit code 2."""
    code = main([])
    assert code == 2
    captured = capsys.readouterr()
    assert "usage: openbench" in captured.out


def test_cli_validate_dataset_success_and_failure(tmp_path: Path) -> None:
    """`openbench validate-dataset` returns 0 on valid dataset and 1 on invalid dataset."""
    repo_root = Path(__file__).resolve().parent.parent
    valid_ds = repo_root / "datasets" / "reasoning" / "sample.jsonl"

    assert main(["validate-dataset", str(valid_ds)]) == 0

    invalid_ds = tmp_path / "bad.jsonl"
    invalid_ds.write_text("{corrupt json\n", encoding="utf-8")
    assert main(["validate-dataset", str(invalid_ds)]) == 1


def test_cli_evaluate_and_compare_workflow(tmp_path: Path) -> None:
    """`openbench evaluate` and `openbench compare` generate valid JSON and Markdown reports."""
    repo_root = Path(__file__).resolve().parent.parent
    dataset_path = repo_root / "datasets" / "reasoning" / "sample.jsonl"
    responses_a = repo_root / "examples" / "sample_responses.json"

    responses_b = tmp_path / "responses_b.json"
    responses_b.write_text(
        json.dumps(
            {
                "model_name": "ModelB",
                "model_version": "0.5",
                "responses": [
                    {"id": "reasoning_001", "response": "180"},
                    {"id": "reasoning_002", "response": "wrong"},
                ],
            }
        ),
        encoding="utf-8",
    )

    out_dir = tmp_path / "results"
    code_eval_a = main(
        [
            "evaluate",
            "--dataset",
            str(dataset_path),
            "--responses",
            str(responses_a),
            "--metrics",
            "exact_match",
            "accuracy",
            "keyword_coverage",
            "--output",
            str(out_dir),
        ]
    )
    assert code_eval_a == 0
    model_a_json = out_dir / "alphabaseline-7b.json"
    assert model_a_json.exists()
    assert (out_dir / "alphabaseline-7b.md").exists()

    code_eval_b = main(
        [
            "evaluate",
            "--dataset",
            str(dataset_path),
            "--responses",
            str(responses_b),
            "--output",
            str(out_dir),
        ]
    )
    assert code_eval_b == 0
    model_b_json = out_dir / "modelb.json"
    assert model_b_json.exists()

    cmp_md = out_dir / "comparison.md"
    code_cmp = main(
        [
            "compare",
            "--results",
            str(model_a_json),
            "--results",
            str(model_b_json),
            "--output",
            str(cmp_md),
        ]
    )
    assert code_cmp == 0
    assert cmp_md.exists()
    assert (out_dir / "comparison.json").exists()


def test_cli_compare_fewer_than_two_files_fails(tmp_path: Path) -> None:
    """`openbench compare` with only 1 result file returns exit code 2."""
    dummy = tmp_path / "one.json"
    dummy.write_text("{}", encoding="utf-8")
    code = main(["compare", "--results", str(dummy), "--output", str(tmp_path / "out.md")])
    assert code == 2
