"""Repetition-aware cache isolation (plan B2, section 3.3; v0.3.1 finding F1)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from openbench import (
    DatasetLoader,
    EvaluationConfig,
    ExecutionConfig,
    ExperimentManager,
    MockModelProvider,
    ResponseCache,
)
from openbench.evaluator import ResponseItem
from openbench.exceptions import OpenBenchError

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "datasets" / "reasoning" / "sample.jsonl"

#: Key computed with the v0.3.1 implementation (commit 8d3295a) for the same inputs.
V031_GOLDEN_KEY = "b666e442bc0489ddce4e125e9ab030632de3f4af203eaf931b8899708b423af9"


class CountingProvider(MockModelProvider):
    """Returns a different answer on every call (simulates sampling noise)."""

    def generate_one(self, record):
        item = super().generate_one(record)
        n = self.call_counts[record.id]
        return ResponseItem(id=item.id, response=f"{record.id}-call-{n}", latency_ms=1.0)


def _setup(tmp_path):
    ds = DatasetLoader.load(str(DATASET))
    manager = ExperimentManager(tmp_path / "exp")
    cache = ResponseCache(tmp_path / "cache", enabled=True)
    cfg = EvaluationConfig(metric_names=["exact_match"])
    return ds, manager, cache, cfg


def test_v031_key_unchanged_without_repetition_context():
    key = ResponseCache.compute_cache_key(
        "mock", "m", "1.0", "reasoning", "0.2.0", "reasoning_001", "What is 2+2?",
        {"temperature": 0.0, "max_tokens": 64, "api_key": "sk-abcdefghijklmnop"},
    )[0]
    assert key == V031_GOLDEN_KEY


def test_repetition_context_changes_key_by_group_and_index():
    args = ("mock", "m", "1.0", "d", "1", "r1", "p", {})
    k0 = ResponseCache.compute_cache_key(*args)[0]
    k_g1_0 = ResponseCache.compute_cache_key(*args, repetition_context=("grp_a", 0))[0]
    k_g1_1 = ResponseCache.compute_cache_key(*args, repetition_context=("grp_a", 1))[0]
    k_g2_0 = ResponseCache.compute_cache_key(*args, repetition_context=("grp_b", 0))[0]
    assert len({k0, k_g1_0, k_g1_1, k_g2_0}) == 4
    assert ResponseCache.compute_cache_key(*args, repetition_context=("grp_a", 0))[0] == k_g1_0


@pytest.mark.parametrize("ctx", [("", 0), ("grp_a", -1), ("grp_a", True), ("grp_a",), "grp_a"])
def test_invalid_repetition_context_rejected(ctx):
    with pytest.raises(OpenBenchError):
        ResponseCache.compute_cache_key("m", "m", "1", "d", "1", "r", "p", {}, repetition_context=ctx)


def test_a_repetitions_never_share_cached_responses(tmp_path):
    ds, manager, cache, cfg = _setup(tmp_path)
    provider = CountingProvider(model_name="noisy")
    group = manager.execute_repeated(ds, provider, 3, evaluation_config=cfg, cache=cache,
                                     execution_config=ExecutionConfig(use_cache=True))
    assert sum(provider.call_counts.values()) == 3 * len(ds)
    by_rep = [{i.id: i.model_response for i in ev.individual_results} for ev in group.evaluations]
    for rid in by_rep[0]:
        assert len({rep[rid] for rep in by_rep}) == 3, rid


def test_b_independent_groups_with_identical_configuration_both_call_provider(tmp_path):
    ds, manager, cache, cfg = _setup(tmp_path)
    provider = CountingProvider(model_name="noisy")
    exec_cfg = ExecutionConfig(use_cache=True)
    manager.execute_repeated(ds, provider, 2, evaluation_config=cfg, cache=cache, execution_config=exec_cfg)
    calls_first = sum(provider.call_counts.values())
    manager.execute_repeated(ds, provider, 2, evaluation_config=cfg, cache=cache, execution_config=exec_cfg)
    assert calls_first == 2 * len(ds)
    assert sum(provider.call_counts.values()) == 4 * len(ds)


def test_c_resume_reuses_only_its_own_repetition_entries(tmp_path):
    ds, manager, cache, cfg = _setup(tmp_path)
    failing = CountingProvider(model_name="noisy", fail_on_ids=["reasoning_003"])
    exec_cfg = ExecutionConfig(use_cache=True)
    group = manager.execute_repeated(ds, failing, 2, evaluation_config=cfg, cache=cache, execution_config=exec_cfg)
    assert all(r.status == "partial" for r in group.records)
    # Remove the preserved responses of repetition 1 so its resume must use the cache only.
    healthy = CountingProvider(model_name="noisy")
    result = manager.resume_group(group.repeat_group_id, provider=healthy, cache=ResponseCache(tmp_path / "cache"))
    assert result.is_complete
    # Only the two failed items (one per repetition) were requested again.
    assert sum(healthy.call_counts.values()) == 2
    # Each repetition's previously answered items keep their own responses.
    for before, after in zip(group.evaluations, result.evaluations, strict=True):
        b = {i.id: i.model_response for i in before.individual_results if i.model_response}
        a = {i.id: i.model_response for i in after.individual_results}
        assert all(a[k] == v for k, v in b.items())


def test_c2_cache_hits_are_confined_to_the_same_group_and_index(tmp_path):
    ds, manager, cache, cfg = _setup(tmp_path)
    provider = CountingProvider(model_name="noisy")
    group = manager.execute_repeated(ds, provider, 2, evaluation_config=cfg, cache=cache,
                                     execution_config=ExecutionConfig(use_cache=True))
    entries = [json.loads(line) for line in (tmp_path / "cache" / "cache_entries.jsonl").read_text().splitlines()]
    assert len(entries) == 2 * len(ds)
    assert {(e["repeat_group_id"], e["repetition_index"]) for e in entries} == {
        (group.repeat_group_id, 0), (group.repeat_group_id, 1)
    }
    # Re-running repetition 0 of the same group from scratch hits only (G, 0) entries.
    from openbench.lineage import RunLineage

    rec, ev, perf = manager.execute_and_record(
        ds, provider, evaluation_config=cfg, cache=ResponseCache(tmp_path / "cache"),
        execution_config=ExecutionConfig(use_cache=True),
        lineage=RunLineage.for_repetition(group.repeat_group_id, 0, 2),
    )
    assert perf.cached_requests == len(ds)
    first = {i.id: i.model_response for i in group.evaluations[0].individual_results}
    assert {i.id: i.model_response for i in ev.individual_results} == first


def test_single_runs_keep_v031_cache_sharing_and_entry_format(tmp_path):
    ds, manager, cache, cfg = _setup(tmp_path)
    provider = CountingProvider(model_name="noisy")
    exec_cfg = ExecutionConfig(use_cache=True)
    manager.execute_and_record(ds, provider, evaluation_config=cfg, cache=cache, execution_config=exec_cfg)
    _, _, perf = manager.execute_and_record(ds, provider, evaluation_config=cfg, cache=cache, execution_config=exec_cfg)
    assert perf.cached_requests == len(ds)  # documented v0.3.1 behaviour for single runs
    entry = json.loads((tmp_path / "cache" / "cache_entries.jsonl").read_text().splitlines()[0])
    assert "repeat_group_id" not in entry and "repetition_index" not in entry


def test_e_no_cache_performs_no_reads_or_writes(tmp_path):
    ds, manager, _, cfg = _setup(tmp_path)
    cache = ResponseCache(tmp_path / "nocache", enabled=False)
    provider = CountingProvider(model_name="noisy")
    manager.execute_repeated(ds, provider, 2, evaluation_config=cfg, cache=cache,
                             execution_config=ExecutionConfig(use_cache=False))
    assert not (tmp_path / "nocache").exists()
    assert cache.hits == 0
    assert sum(provider.call_counts.values()) == 2 * len(ds)


def test_cli_repeated_run_does_not_reuse_cache(tmp_path, capsys):
    from openbench.cli import main

    code = main([
        "run", "--dataset", str(DATASET), "--provider", "mock", "--model", "m",
        "--repetitions", "2", "--experiments-dir", str(tmp_path / "exp"),
        "--cache-dir", str(tmp_path / "cache"),
    ])
    assert code == 0
    entries = (tmp_path / "cache" / "cache_entries.jsonl").read_text().splitlines()
    assert len(entries) == 2 * 20
    out = capsys.readouterr().out
    assert "Repeat group 'grp_" in out
