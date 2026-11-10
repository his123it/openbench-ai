"""Repeat groups, run lineage, seed status and group resume (plan B1, B3, B4, F2)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from openbench import (
    DatasetLoader,
    EvaluationConfig,
    ExecutionConfig,
    ExperimentManager,
    GenerationConfig,
    MockModelProvider,
    OllamaAdapter,
    ResponseCache,
)
from openbench.cli import main
from openbench.exceptions import ConfigurationError
from openbench.lineage import RunLineage, is_repeat_group_id, legacy_lineage, seed_status

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "datasets" / "reasoning" / "sample.jsonl"
CFG = EvaluationConfig(metric_names=["exact_match", "keyword_coverage"])
NO_CACHE = ExecutionConfig(use_cache=False)


def _ds():
    return DatasetLoader.load(str(DATASET))


def test_group_lineage_ids_and_index(tmp_path):
    manager = ExperimentManager(tmp_path / "exp")
    group = manager.execute_repeated(_ds(), MockModelProvider(), 3, generation_base_seed=10,
                                     evaluation_config=CFG, execution_config=NO_CACHE)
    assert is_repeat_group_id(group.repeat_group_id)
    run_ids = [r.run_id for r in group.records]
    assert len(set(run_ids)) == 3
    for i, rec in enumerate(group.records):
        lin = rec.lineage
        assert lin["schema_version"] == "0.4.0"
        assert lin["repeat_group_id"] == group.repeat_group_id
        assert (lin["repetition_index"], lin["repetition_count"]) == (i, 3)
        assert (lin["generation_base_seed"], lin["repetition_seed"]) == (10, 10 + i)
        assert lin["seed_status"] == "unsupported_by_adapter"
        assert lin["seed_passed_to_provider"] is False
        assert lin["selected_record_ids"] == sorted(r.id for r in _ds().records)
        assert len(lin["record_content_sha256"]) == 20
        assert lin["experiment_id"] == group.repeat_group_id
        assert manager.verify_run(rec.run_id).valid
    # Only run directories are created as sub-directories; the index is a root-level file.
    subdirs = sorted(p.name for p in (tmp_path / "exp").iterdir() if p.is_dir())
    assert subdirs == sorted(run_ids)
    index = json.loads((tmp_path / "exp" / f"{group.repeat_group_id}.json").read_text())
    assert index["status"] == "complete"
    assert [m["run_id"] for m in index["members"]] == run_ids
    assert manager.verify_group(group.repeat_group_id)["valid"]
    assert [r.run_id for r in manager.list_runs(repeat_group_id=group.repeat_group_id)] == sorted(run_ids)


def test_single_run_lineage_defaults(tmp_path):
    manager = ExperimentManager(tmp_path / "exp")
    rec, _, _ = manager.execute_and_record(_ds(), MockModelProvider(), evaluation_config=CFG,
                                           execution_config=NO_CACHE)
    lin = rec.lineage
    assert lin["repeat_group_id"] is None
    assert (lin["repetition_index"], lin["repetition_count"]) == (0, 1)
    assert lin["seed_status"] == "not_supplied"
    repro = json.loads((Path(rec.run_dir) / "reproducibility.json").read_text())
    assert repro["schema_version"] == "0.4.0"
    assert repro["lineage"] == lin
    assert not list((tmp_path / "exp").glob("grp_*.json"))


def test_lineage_tampering_detected(tmp_path):
    manager = ExperimentManager(tmp_path / "exp")
    group = manager.execute_repeated(_ds(), MockModelProvider(), 2, evaluation_config=CFG,
                                     execution_config=NO_CACHE)
    run_dir = Path(group.records[1].run_dir)
    repro = json.loads((run_dir / "reproducibility.json").read_text())
    repro["lineage"]["repetition_index"] = 0
    (run_dir / "reproducibility.json").write_text(json.dumps(repro))
    check = manager.verify_run(group.records[1].run_id)
    assert not check.valid
    assert any("repetition_index" in m for m in check.inconsistent_metadata)


def test_seed_status_values():
    assert seed_status(None, True, 5) == "not_supplied"
    assert seed_status(3, False, 0) == "unsupported_by_adapter"
    assert seed_status(3, True, 4) == "passed_to_provider"
    assert seed_status(3, True, 0) == "no_provider_requests"
    assert legacy_lineage()["seed_status"] == "unknown"


def test_ollama_seed_passed_to_provider_with_mock_transport(tmp_path):
    seen = []

    def transport(url, payload, timeout):
        seen.append(payload["options"].get("seed"))
        return {"response": "4", "prompt_eval_count": 3, "eval_count": 1}

    adapter = OllamaAdapter(model_name="llama3.2", transport_fn=transport,
                            generation_config=GenerationConfig(seed=None))
    manager = ExperimentManager(tmp_path / "exp")
    ds = DatasetLoader.load(str(DATASET), ids=["reasoning_001", "reasoning_002"])
    cache = ResponseCache(tmp_path / "cache")
    group = manager.execute_repeated(ds, adapter, 2, generation_base_seed=40, evaluation_config=CFG,
                                     execution_config=ExecutionConfig(use_cache=True), cache=cache)
    assert sorted(seen) == [40, 40, 41, 41]
    assert [r.lineage["seed_status"] for r in group.records] == ["passed_to_provider"] * 2
    assert all(r.lineage["seed_passed_to_provider"] for r in group.records)
    # Re-running a repetition fully from its cache forwards nothing.
    rec, _, _ = manager.execute_and_record(
        ds, adapter, evaluation_config=CFG, cache=ResponseCache(tmp_path / "cache"),
        execution_config=ExecutionConfig(use_cache=True),
        lineage=RunLineage.for_repetition(group.repeat_group_id, 0, 2, generation_base_seed=40),
    )
    assert rec.lineage["seed_status"] == "no_provider_requests"
    assert adapter.generation_config.seed is None  # the caller's adapter is never mutated


def test_no_determinism_claim_in_lineage(tmp_path):
    manager = ExperimentManager(tmp_path / "exp")
    rec, _, _ = manager.execute_and_record(_ds(), MockModelProvider(), evaluation_config=CFG,
                                           execution_config=NO_CACHE)
    text = json.dumps(rec.lineage).lower()
    assert "does not guarantee deterministic" in text
    assert "is deterministic" not in text


def test_group_resume_finishes_partial_and_missing_repetitions(tmp_path):
    manager = ExperimentManager(tmp_path / "exp")
    failing = MockModelProvider(fail_on_ids=["reasoning_005"])
    group = manager.execute_repeated(_ds(), failing, 3, generation_base_seed=1, evaluation_config=CFG,
                                     execution_config=ExecutionConfig(use_cache=False))
    # Simulate an interrupted group: repetition 2 never ran.
    shutil.rmtree(group.records[2].run_dir)
    healthy = MockModelProvider()
    result = manager.resume_group(group.repeat_group_id, provider=healthy)
    assert result.is_complete
    assert [r.repetition_index for r in result.records] == [0, 1, 2]
    # Two partial members need one item each; the missing repetition needs all 20.
    assert sum(healthy.call_counts.values()) == 2 + 20
    assert result.records[0].run_id == group.records[0].run_id
    assert result.records[2].lineage["repetition_seed"] == 3
    assert manager.verify_group(group.repeat_group_id)["valid"]
    # Resuming again requests nothing.
    again = MockModelProvider()
    manager.resume_group(group.repeat_group_id, provider=again)
    assert sum(again.call_counts.values()) == 0


def test_group_resume_refuses_duplicate_indices(tmp_path):
    manager = ExperimentManager(tmp_path / "exp")
    group = manager.execute_repeated(_ds(), MockModelProvider(), 2, evaluation_config=CFG,
                                     execution_config=NO_CACHE)
    clone = Path(group.records[0].run_dir).with_name("run_duplicate000")
    shutil.copytree(group.records[0].run_dir, clone)
    cfg = json.loads((clone / "config.json").read_text())
    cfg["run_id"] = "run_duplicate000"
    (clone / "config.json").write_text(json.dumps(cfg))
    with pytest.raises(ConfigurationError, match="duplicate repetition indices"):
        manager.resume_group(group.repeat_group_id, provider=MockModelProvider())
    assert not manager.verify_group(group.repeat_group_id)["valid"]


def test_group_index_rebuilt_from_lineage_when_stale_or_missing(tmp_path):
    manager = ExperimentManager(tmp_path / "exp")
    group = manager.execute_repeated(_ds(), MockModelProvider(), 2, evaluation_config=CFG,
                                     execution_config=NO_CACHE)
    path = tmp_path / "exp" / f"{group.repeat_group_id}.json"
    stale = json.loads(path.read_text())
    stale["members"] = [{"repetition_index": 0, "run_id": "run_bogus"}]
    path.write_text(json.dumps(stale))
    index = manager.rebuild_group_index(group.repeat_group_id)
    assert {m["run_id"] for m in index["members"]} == {r.run_id for r in group.records}
    assert any("lineage" in w for w in index["rebuild_warnings"])
    path.unlink()
    assert manager.verify_group(group.repeat_group_id)["valid"]
    assert manager.rebuild_group_index(group.repeat_group_id)["member_count"] == 2


def test_execute_repeated_rejects_bad_arguments(tmp_path):
    manager = ExperimentManager(tmp_path / "exp")
    for bad in (0, -1, True, 2.0):
        with pytest.raises(ConfigurationError):
            manager.execute_repeated(_ds(), MockModelProvider(), bad)
    with pytest.raises(ConfigurationError):
        manager.execute_repeated(_ds(), MockModelProvider(), 2, repeat_group_id="grp_../../x")
    group = manager.execute_repeated(_ds(), MockModelProvider(), 2, evaluation_config=CFG,
                                     execution_config=NO_CACHE)
    with pytest.raises(ConfigurationError, match="already exists"):
        manager.execute_repeated(_ds(), MockModelProvider(), 2, repeat_group_id=group.repeat_group_id)


def test_run_lineage_validation():
    with pytest.raises(ConfigurationError):
        RunLineage(repetition_index=1)
    with pytest.raises(ConfigurationError):
        RunLineage(repeat_group_id="grp_a", repetition_index=2, repetition_count=2)
    with pytest.raises(ConfigurationError):
        RunLineage(repeat_group_id="../etc", repetition_count=2)
    with pytest.raises(ConfigurationError):
        RunLineage(repeat_group_id="grp_a", repetition_index=1, repetition_count=2,
                   generation_base_seed=5, repetition_seed=5)
    with pytest.raises(ConfigurationError):
        RunLineage(experiment_id="has space")


def test_unsafe_run_ids_never_resolve_outside_root(tmp_path):
    manager = ExperimentManager(tmp_path / "exp")
    for bad in ("run_/../../etc", "../x", "run_..", "run_a/b"):
        with pytest.raises(KeyError):
            manager.verify_run(bad)
        with pytest.raises(KeyError):
            manager.load_evaluation(bad)


def test_resume_keeps_id_filter_and_lineage(tmp_path):
    manager = ExperimentManager(tmp_path / "exp")
    ids = tmp_path / "ids.txt"
    ids.write_text("reasoning_001\nreasoning_002\nreasoning_003\n")
    cfg = EvaluationConfig(metric_names=["exact_match"], ids_file=str(ids))
    rec, _, _ = manager.execute_and_record(_ds(), MockModelProvider(fail_on_ids=["reasoning_002"]),
                                           evaluation_config=cfg, execution_config=NO_CACHE)
    assert rec.total_items == 3 and rec.status == "partial"
    ids.unlink()  # the lineage keeps the exact IDs, so the file is no longer needed
    resumed, _, _ = manager.resume_run(rec.run_id, provider=MockModelProvider())
    assert resumed.total_items == 3 and resumed.status == "completed"
    assert resumed.lineage["selected_record_ids"] == ["reasoning_001", "reasoning_002", "reasoning_003"]


def test_cli_repetitions_seed_and_group_commands(tmp_path, capsys):
    exp = tmp_path / "exp"
    assert main(["run", "--dataset", str(DATASET), "--provider", "mock", "--model", "m",
                 "--repetitions", "3", "--seed", "5", "--no-cache", "--experiments-dir", str(exp)]) == 0
    out = capsys.readouterr().out
    group_id = out.split("Repeat group '")[1].split("'")[0]
    assert "Generation seed status: unsupported_by_adapter" in out
    assert (exp / f"{group_id}.json").exists()
    assert len(list(exp.rglob("repeated_runs_analysis.json"))) == 1  # legacy file kept (D19)
    assert main(["runs", "list", "--group", group_id, "--experiments-dir", str(exp), "--format", "json"]) == 0
    listed = json.loads(capsys.readouterr().out)
    assert {r["lineage"]["repetition_seed"] for r in listed} == {5, 6, 7}
    assert main(["runs", "show", group_id, "--experiments-dir", str(exp)]) == 0
    assert "3 completed / 3 planned" in capsys.readouterr().out
    assert main(["runs", "verify", group_id, "--experiments-dir", str(exp)]) == 0
    assert main(["runs", "resume", group_id, "--experiments-dir", str(exp),
                 "--cache-dir", str(tmp_path / "c")]) == 0
    assert main(["runs", "show", "grp_doesnotexist", "--experiments-dir", str(exp)]) == 1
    assert main(["run", "--dataset", str(DATASET), "--provider", "mock", "--model", "m",
                 "--repetitions", "0", "--experiments-dir", str(exp)]) == 2


def test_legacy_v030_run_maps_to_unknown_seed_status(tmp_path):
    fixture = ROOT / "tests" / "data" / "v031_run_fixture"
    if not fixture.exists():
        pytest.skip("v0.3.1 run fixture not present")
    exp = tmp_path / "exp"
    shutil.copytree(fixture, exp)
    manager = ExperimentManager(exp)
    runs = manager.list_runs()
    assert runs and all(r.lineage["seed_status"] == "unknown" for r in runs)
    assert all(r.lineage["legacy_run"] for r in runs)


def test_show_verify_and_audit_of_groups_are_read_only(tmp_path, capsys):
    from _v040_helpers import snapshot

    exp = tmp_path / "exp"
    assert main(["run", "--dataset", str(DATASET), "--provider", "mock", "--model", "m", "--repetitions", "2",
                 "--no-cache", "--experiments-dir", str(exp)]) == 0
    group_id = capsys.readouterr().out.split("Repeat group '")[1].split("'")[0]
    index = exp / f"{group_id}.json"
    stale = json.loads(index.read_text())
    stale["members"] = []
    index.write_text(json.dumps(stale))
    before = snapshot(exp)
    assert main(["runs", "show", group_id, "--experiments-dir", str(exp)]) == 0
    assert main(["runs", "verify", group_id, "--experiments-dir", str(exp), "--quality-checks"]) == 0
    assert main(["analyze", "--runs", group_id, "--experiments-dir", str(exp), "--resamples", "100"]) == 0
    assert snapshot(exp) == before
