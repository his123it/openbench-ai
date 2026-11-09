"""Compatibility guards (plan X1, F2, section 12.1 items 3-5).

* The 70 public names and the 19 CLI commands (flags, defaults, choices) of v0.3.1 are all
  preserved (inventory generated from commit 8d3295a, tests/data/v031_public_surface.json);
  v0.4.0 only adds names, commands and flags.
* The 10 dataset files (200 records) match their v0.3.1 SHA-256 digests.
* Statistical analysis never modifies a v0.3.1 run directory.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import openbench
from _cli_inventory import cli_inventory
from openbench.cli import build_parser

ROOT = Path(__file__).resolve().parents[1]
SURFACE = json.loads((ROOT / "tests" / "data" / "v031_public_surface.json").read_text())

DATASET_SHA256 = {
    "coding": "4cc03a55c3170dd7bd6695b6ae1dabfa772dc684c830e804ca40d966ea99a24c",
    "debugging": "4ecb92d6909559edba530ce32b9bdfbc27b54c1a501f8c741c79fc3e62cd7709",
    "factual_knowledge": "ee6e05e98f6e388c13c9d016a27bab33ead05602298eb55ff0f5810d3bb3cd88",
    "instruction_following": "7cfcf65415739b359f7f35f426e0f4695c0ffeba717fad1077877a27aa7428c8",
    "mathematics": "75afa827ed18c3b509fb6fd7fbc87dd6643f6e87758a13bd12bf2b0abdcdd542",
    "multilingual": "00933c9a534884167c004a44307841dbccff95685392c389b2b0bebded7d380d",
    "reasoning": "9f7352bdca4018106dd38de18baedd8f64c9da85bc4bb91b22dd861f7f194e0c",
    "safety": "39c493fcae9aa9a3fca5d4c4545f8ce004a4d015e168aa5776e63912d69bc0e3",
    "structured_output": "e562d3c9ad65b2c686ba2c3b66aee2b7108d0276bab138137b450479c94fa411",
    "summarization": "91fb76f03a47affdbbf7694ccb30508f667ea246cf568374d6f961a39ea981ea",
}


def test_public_names_preserved() -> None:
    """v0.4.0 is additive: all 70 v0.3.1 names stay, in their original order, first."""
    assert len(SURFACE["public_names"]) == 70
    assert list(openbench.__all__)[:70] == SURFACE["public_names"]
    for name in SURFACE["public_names"]:
        assert hasattr(openbench, name), name


def test_cli_surface_preserved() -> None:
    """Every v0.3.1 command and flag (dest, options, required, nargs, default, choices) is kept."""
    current = cli_inventory(build_parser())
    for path, actions in SURFACE["cli"].items():
        assert path in current, path
        if path == "<root>":
            assert current[path] == actions
            continue
        by_dest = {a["dest"]: a for a in current[path]}
        for action in actions:
            assert by_dest.get(action["dest"]) == action, (path, action["dest"])
    old_leaves = [p for p in SURFACE["cli"] if p != "<root>" and not any(
        o.startswith(p + " ") for o in SURFACE["cli"]
    )]
    assert len(old_leaves) == 19


def test_dataset_files_match_v031_digests() -> None:
    records = 0
    for category, digest in DATASET_SHA256.items():
        path = ROOT / "datasets" / category / "sample.jsonl"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, category
        records += sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    assert records == 200
    assert sorted(p.parent.name for p in (ROOT / "datasets").glob("*/sample.jsonl")) == sorted(
        DATASET_SHA256
    )


def _snapshot(directory: Path) -> dict[str, tuple[str, int, int]]:
    out = {}
    for path in sorted(directory.rglob("*")):
        if path.is_file():
            stat = path.stat()
            out[str(path.relative_to(directory))] = (
                hashlib.sha256(path.read_bytes()).hexdigest(),
                stat.st_size,
                stat.st_mtime_ns,
            )
    return out


def test_statistics_never_modify_v031_run_directory(tmp_path, monkeypatch) -> None:
    from openbench import (
        DatasetLoader,
        EvaluationConfig,
        ExecutionConfig,
        ExperimentManager,
        MockModelProvider,
    )
    from openbench.stats import (
        binary_paired_counts,
        bootstrap_mean_ci,
        mcnemar_exact,
        sign_flip_test,
        tost_paired_binary_exact,
    )

    monkeypatch.chdir(tmp_path)
    dataset = DatasetLoader.load(str(ROOT / "datasets" / "reasoning" / "sample.jsonl"))
    manager = ExperimentManager(str(tmp_path / "experiments"))
    for model, use_expected in (("model-a", True), ("model-b", False)):
        manager.execute_and_record(
            dataset=dataset,
            provider=MockModelProvider(model_name=model, use_expected_answer=use_expected),
            evaluation_config=EvaluationConfig(metric_names=["exact_match", "keyword_coverage"]),
            execution_config=ExecutionConfig(use_cache=False),
        )
    experiments = tmp_path / "experiments"
    before = _snapshot(experiments)
    assert before, "expected v0.3.1 run artifacts"

    evaluations = sorted(experiments.rglob("evaluation.json"))
    assert len(evaluations) == 2
    scores = []
    for path in evaluations:
        data = json.loads(path.read_text(encoding="utf-8"))
        items = sorted(data["individual_results"], key=lambda r: r["id"])
        scores.append(items)
    em = [[r["metrics"]["exact_match"]["score"] for r in items] for items in scores]
    kc = [[r["metrics"]["keyword_coverage"]["score"] for r in items] for items in scores]
    counts = binary_paired_counts(em[0], em[1])
    mcnemar_exact(counts.b, counts.c)
    tost_paired_binary_exact(counts.b, counts.c, counts.n, 0.2, compute_interval=False)
    diffs = [b - a for a, b in zip(kc[0], kc[1], strict=True)]
    sign_flip_test(diffs)
    bootstrap_mean_ci(diffs, resamples=200)

    assert _snapshot(experiments) == before
    assert sorted(os.listdir(tmp_path)) == ["experiments"]
