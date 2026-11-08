"""End-to-end tests for v0.3.0 CLI commands and backward compatibility with v0.1.0 and v0.2.0."""

from __future__ import annotations

import json
from pathlib import Path

from openbench.cli import main


def test_v0_3_0_cli_run_runs_providers_and_cache_commands(tmp_path: Path) -> None:
    """Verify `openbench run`, `runs list|show|verify|compare`, `providers list|check`, and `cache clear|stats`."""
    repo_root = Path(__file__).resolve().parent.parent
    ds_path = repo_root / "datasets" / "mathematics" / "sample.jsonl"
    exp_dir = tmp_path / "experiments"
    cache_dir = tmp_path / "cache"
    out_dir = tmp_path / "reports"

    # 1. providers list & check mock
    assert main(["providers", "list"]) == 0
    assert main(["providers", "check", "mock"]) == 0

    # 2. openbench run with mock provider
    code_run1 = main(
        [
            "run",
            "--dataset",
            str(ds_path),
            "--provider",
            "mock",
            "--model",
            "mock-alpha",
            "--metrics",
            "exact_match",
            "accuracy",
            "--experiments-dir",
            str(exp_dir),
            "--cache-dir",
            str(cache_dir),
            "--output",
            str(out_dir),
        ]
    )
    assert code_run1 == 0

    # 3. Second run with another mock model for run comparison
    code_run2 = main(
        [
            "run",
            "--dataset",
            str(ds_path),
            "--provider",
            "mock",
            "--model",
            "mock-beta",
            "--metrics",
            "exact_match",
            "accuracy",
            "--experiments-dir",
            str(exp_dir),
            "--cache-dir",
            str(cache_dir),
        ]
    )
    assert code_run2 == 0

    # 4. runs list, show, verify, compare
    assert main(["runs", "list", "--experiments-dir", str(exp_dir)]) == 0
    run_folders = sorted(d.name for d in exp_dir.iterdir() if d.is_dir())
    assert len(run_folders) == 2

    run_id_1, run_id_2 = run_folders[0], run_folders[1]
    assert main(["runs", "show", run_id_1, "--experiments-dir", str(exp_dir)]) == 0
    assert main(["runs", "verify", run_id_1, "--experiments-dir", str(exp_dir)]) == 0

    cmp_out = tmp_path / "runs_cmp.md"
    assert (
        main(
            [
                "runs",
                "compare",
                run_id_1,
                run_id_2,
                "--experiments-dir",
                str(exp_dir),
                "--output",
                str(cmp_out),
            ]
        )
        == 0
    )
    assert cmp_out.exists()

    # 5. cache stats & clear
    assert main(["cache", "stats", "--cache-dir", str(cache_dir)]) == 0
    assert main(["cache", "clear", "--cache-dir", str(cache_dir)]) == 0


def test_v0_3_0_config_driven_repeated_experiment(tmp_path: Path) -> None:
    """`openbench run --config configs/repeated_benchmark.yaml` executes repeated runs and writes variance analysis."""
    repo_root = Path(__file__).resolve().parent.parent
    cfg_path = repo_root / "configs" / "repeated_benchmark.yaml"
    exp_dir = tmp_path / "rep_experiments"

    code = main(
        [
            "run",
            "--config",
            str(cfg_path),
            "--experiments-dir",
            str(exp_dir),
        ]
    )
    assert code == 0
    run_dirs = sorted(d for d in exp_dir.iterdir() if d.is_dir())
    assert len(run_dirs) == 3
    var_files = list(exp_dir.rglob("repeated_runs_analysis.json"))
    assert len(var_files) == 1
    var_data = json.loads(var_files[0].read_text(encoding="utf-8"))
    assert var_data["sample_size_n"] == 3
