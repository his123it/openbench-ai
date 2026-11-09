"""Reliability: missing files, empty and malformed inputs, partial and interrupted runs,
retry exhaustion, timeouts, cache corruption, incomplete run folders, degenerate samples
and large inputs (roadmap Stage 6 C)."""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import pytest

from _v040_helpers import correct_answers, evaluate, write_dataset
from openbench import (
    DatasetLoader,
    EvaluationConfig,
    ExecutionConfig,
    ExperimentManager,
    MockModelProvider,
    ResponseCache,
    RetryPolicy,
)
from openbench.cli import main
from openbench.exceptions import OpenBenchError
from openbench.paired import ComparisonArm, ComparisonPolicy, PairedComparisonEngine

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "datasets" / "reasoning" / "sample.jsonl"
FAST = ComparisonPolicy(resamples=200, permutations=999)


def test_missing_files_exit_cleanly(tmp_path, capsys):
    assert main(["validate-dataset", str(tmp_path / "nope.jsonl")]) == 1
    assert main(["validate-dataset", str(tmp_path / "nope.jsonl"), "--quality-checks"]) == 1
    assert main(["analyze", "--results", str(tmp_path / "a.json"), str(tmp_path / "b.json")]) == 1
    assert main(["preferences", "validate", "--key", str(tmp_path / "k.json"), "--annotations", str(tmp_path / "a.jsonl")]) == 1
    assert "Traceback" not in capsys.readouterr().err


def test_empty_dataset(tmp_path):
    empty = tmp_path / "empty.jsonl"
    empty.write_text("")
    assert main(["validate-dataset", str(empty)]) == 1
    assert main(["validate-dataset", str(empty), "--quality-checks"]) == 1  # DQ013 error: no records


def test_retry_exhaustion_and_timeouts_recorded(tmp_path):
    ds = DatasetLoader.load(str(DATASET))
    manager = ExperimentManager(tmp_path / "exp")
    provider = MockModelProvider(transient_failures_by_id={"reasoning_001": 5}, timeout_on_ids=["reasoning_002"])
    cfg = ExecutionConfig(use_cache=False, retry_policy=RetryPolicy(max_attempts=2, initial_backoff_seconds=0.0))
    group = manager.execute_repeated(ds, provider, 2, evaluation_config=EvaluationConfig(metric_names=["exact_match"]), execution_config=cfg)
    for rec in group.records:
        assert rec.status == "partial"
        assert set(rec.failure_info["failed_record_ids"]) == {"reasoning_001", "reasoning_002"}
        assert rec.failure_info["timeout_count"] == 1
    from openbench.stats import decompose_variability

    var = decompose_variability(group.evaluations, ["exact_match"], resamples=200)
    assert var["failures"]["items_failing_in_every_repetition"] == ["reasoning_001", "reasoning_002"]


def test_interrupted_group_leaves_resumable_state(tmp_path):
    ds = DatasetLoader.load(str(DATASET))
    manager = ExperimentManager(tmp_path / "exp")

    class Interrupting(MockModelProvider):
        calls = 0

        def generate_one(self, record):
            Interrupting.calls += 1
            if Interrupting.calls == 30:
                raise KeyboardInterrupt
            return super().generate_one(record)

    with pytest.raises(KeyboardInterrupt):
        manager.execute_repeated(ds, Interrupting(), 3, evaluation_config=EvaluationConfig(metric_names=["exact_match"]),
                                 execution_config=ExecutionConfig(use_cache=False))
    index_files = list((tmp_path / "exp").glob("grp_*.json"))
    assert len(index_files) == 1
    group_id = index_files[0].stem
    index = json.loads(index_files[0].read_text())
    assert index["status"] == "incomplete" and index["planned_repetitions"] == 3 and index["member_count"] == 1
    healthy = MockModelProvider()
    result = manager.resume_group(group_id, provider=healthy)
    assert result.is_complete and sum(healthy.call_counts.values()) == 40


def test_cache_corruption_reported_clearly(tmp_path, capsys):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / "cache_entries.jsonl").write_text('{"cache_key": "x"}\n{broken\n')
    code = main(["run", "--dataset", str(DATASET), "--provider", "mock", "--model", "m",
                 "--cache-dir", str(cache_dir), "--experiments-dir", str(tmp_path / "exp")])
    assert code == 1
    assert "Corrupted cache file" in capsys.readouterr().err
    with pytest.raises(OpenBenchError):
        ResponseCache(cache_dir).stats()


def test_incomplete_run_folder(tmp_path, capsys):
    ds = DatasetLoader.load(str(DATASET))
    manager = ExperimentManager(tmp_path / "exp")
    rec, _, _ = manager.execute_and_record(ds, MockModelProvider(), execution_config=ExecutionConfig(use_cache=False))
    other, _, _ = manager.execute_and_record(ds, MockModelProvider(model_name="o"), execution_config=ExecutionConfig(use_cache=False))
    (Path(rec.run_dir) / "evaluation.json").unlink()
    assert not manager.verify_run(rec.run_id).valid
    assert {r.run_id for r in manager.list_runs()} == {rec.run_id, other.run_id}
    assert main(["analyze", "--runs", rec.run_id, other.run_id, "--experiments-dir", str(tmp_path / "exp")]) == 1
    (Path(other.run_dir) / "config.json").write_text("[]")
    assert {r.run_id for r in manager.list_runs()} == {rec.run_id}  # unreadable run skipped, not a crash


def test_new_run_ids_never_collide(tmp_path, monkeypatch):
    import uuid

    manager = ExperimentManager(tmp_path / "exp")
    (tmp_path / "exp" / "run_aaaaaaaaaaaa").mkdir(parents=True)
    seq = iter([uuid.UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"), uuid.UUID("bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb")])
    monkeypatch.setattr(uuid, "uuid4", lambda: next(seq))
    assert manager._new_run_id() == "run_bbbbbbbbbbbb"


@pytest.mark.parametrize("answers_b", [
    {"missing": tuple(range(1, 9))},  # n = 0 after exclusions
    {"missing": tuple(range(2, 9))},  # n = 1
    {},                               # all ties
])
def test_degenerate_samples_produce_nulls_not_crashes(tmp_path, answers_b):
    ds = write_dataset(tmp_path / "ds.jsonl", 8)
    a = evaluate(ds, correct_answers(8), "a", tmp_path / "a.json")
    b = evaluate(ds, correct_answers(8, **answers_b), "b", tmp_path / "b.json")
    out = PairedComparisonEngine().compare(
        [ComparisonArm.from_evaluation_file(a), ComparisonArm.from_evaluation_file(b)],
        ["exact_match", "keyword_coverage"],
        ComparisonPolicy(resamples=200, permutations=999, equivalence_margin=0.1),
    ).to_dict()
    json.dumps(out, allow_nan=False)
    for comp in out["comparisons"]:
        if comp["sample_count"] < 2:
            assert comp["p_value"] is None and comp["unavailable"]["p_value"] == "insufficient_items"
            assert comp["equivalence"]["familywise_result"] == "not_established"
        else:
            assert comp["point_estimate"] == 0.0
            assert any(w["code"] in {"degenerate_zero_variance", "all_differences_zero", "no_discordant_pairs"}
                       for w in comp["confidence_interval"]["warnings"] + (comp["test"] or {}).get("warnings", []))


def test_large_dataset_audit_and_comparison_are_bounded(tmp_path):
    n = 3000
    rows = [{"id": f"r{i:05d}", "category": f"c{i % 7}", "prompt": f"Question {i}: what is {i} times {i % 13}?",
             "expected_answer": str(i * (i % 13))} for i in range(n)]
    path = tmp_path / "big.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    from openbench.quality import DatasetAuditor

    start = time.perf_counter()
    report = DatasetAuditor(use_review_notes=False).audit(path).to_dict()
    assert time.perf_counter() - start < 60
    assert report["subject"]["record_count"] == n and report["summary"]["errors"] == 0


def test_corrupted_group_index_is_rebuilt(tmp_path):
    ds = DatasetLoader.load(str(DATASET))
    manager = ExperimentManager(tmp_path / "exp")
    group = manager.execute_repeated(ds, MockModelProvider(), 2, evaluation_config=EvaluationConfig(metric_names=["exact_match"]),
                                     execution_config=ExecutionConfig(use_cache=False))
    (tmp_path / "exp" / f"{group.repeat_group_id}.json").write_text("{corrupt")
    assert manager.load_group(group.repeat_group_id)["index"]["member_count"] == 2
    shutil.rmtree(tmp_path / "exp" / group.records[0].run_id)
    assert manager.rebuild_group_index(group.repeat_group_id)["missing_repetition_indices"] == [0]


def test_corrupted_cache_can_be_cleared_and_is_not_half_loaded(tmp_path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    good = ResponseCache(cache_dir)
    from openbench.evaluator import ResponseItem

    good.put("p", "m", "1", "d", "1", "r1", "prompt", ResponseItem(id="r1", response="x"))
    with (cache_dir / "cache_entries.jsonl").open("a") as fh:
        fh.write("{broken\n")
    fresh = ResponseCache(cache_dir)
    with pytest.raises(OpenBenchError):
        fresh.stats()
    with pytest.raises(OpenBenchError):
        fresh.stats()  # still failing: never a silently truncated cache
    assert main(["cache", "clear", "--cache-dir", str(cache_dir)]) == 0
    assert ResponseCache(cache_dir).stats()["stored_entries"] == 0
