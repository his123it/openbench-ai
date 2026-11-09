"""Cross-feature end-to-end workflow (roadmap Stage 6 A).

Validate and audit a dataset -> run two models -> repeat one -> interrupt and resume the
repeat group -> compare stored runs -> create blind preference tasks -> import and
analyse annotations -> audit runs -> render reports -> verify artifacts. Provider calls
are counted to show that nothing is requested twice and that analysis, reporting,
preference and audit steps never call a provider.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from _v040_helpers import snapshot, strip_volatile
from openbench import (
    DatasetLoader,
    EvaluationConfig,
    ExecutionConfig,
    ExperimentManager,
    MockModelProvider,
    ResponseCache,
)
from openbench.cli import main

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "datasets" / "reasoning" / "sample.jsonl"
CALLS: Counter = Counter()


class CountingMock(MockModelProvider):
    def generate_one(self, record):
        CALLS[(self.model_name, record.id)] += 1
        return super().generate_one(record)


def test_full_workflow(tmp_path, capsys, monkeypatch):
    CALLS.clear()
    exp, cache_dir, out = tmp_path / "exp", tmp_path / "cache", tmp_path / "out"
    d = ["--experiments-dir", str(exp)]

    # 1. Validate and audit the dataset (read-only).
    ds_before = snapshot(DATASET.parent)
    assert main(["validate-dataset", str(DATASET), "--quality-checks", "--output", str(out / "dataset_audit.md"), "--format", "markdown"]) == 0
    assert snapshot(DATASET.parent) == ds_before

    # 2-4. Repeated runs of model A, interrupted (two items fail), then resumed.
    ds = DatasetLoader.load(str(DATASET))
    manager = ExperimentManager(exp)
    cfg = EvaluationConfig(metric_names=["exact_match", "keyword_coverage", "f1"])
    exec_cfg = ExecutionConfig(use_cache=True)
    flaky = CountingMock(model_name="model-a", fail_on_ids=["reasoning_003", "reasoning_017"], stochastic_error_rate=0.3, stochastic_seed=5)
    group = manager.execute_repeated(ds, flaky, 3, generation_base_seed=100, evaluation_config=cfg,
                                     execution_config=exec_cfg, cache=ResponseCache(cache_dir), experiment_id="exp-workflow")
    assert [r.status for r in group.records] == ["partial"] * 3
    calls_after_first = sum(CALLS.values())
    healthy = CountingMock(model_name="model-a", stochastic_error_rate=0.3, stochastic_seed=5)
    resumed = manager.resume_group(group.repeat_group_id, provider=healthy, cache=ResponseCache(cache_dir))
    assert resumed.is_complete
    assert sum(CALLS.values()) - calls_after_first == 2 * 3  # only the failed items, once per repetition
    assert all(r.lineage["experiment_id"] == "exp-workflow" for r in resumed.records)
    assert [r.lineage["repetition_seed"] for r in resumed.records] == [100, 101, 102]

    # Model B: a single run through the CLI.
    assert main(["run", "--dataset", str(DATASET), "--provider", "mock", "--model", "model-b", "--metrics",
                 "exact_match", "keyword_coverage", "f1", "--cache-dir", str(cache_dir), *d]) == 0
    run_b = capsys.readouterr().out.split("Run '")[1].split("'")[0]

    # From here on no provider may be constructed or called.
    calls_before_analysis = sum(CALLS.values())

    def boom(*a, **k):
        raise AssertionError("provider used during analysis")

    monkeypatch.setattr(MockModelProvider, "__init__", boom)
    exp_before = snapshot(exp)

    # 5. Compare stored runs (group vs single run), in all formats; deterministic.
    for fmt in ("json", "markdown", "csv"):
        assert main(["analyze", "--runs", group.repeat_group_id, run_b, *d, "--baseline", run_b, "--resamples", "500",
                     "--permutations", "999", "--equivalence-margin", "0.2", "--format", fmt, "--output", str(out / f"cmp.{fmt}")]) == 0
    assert main(["analyze", "--runs", run_b, group.repeat_group_id, *d, "--baseline", run_b, "--resamples", "500",
                 "--permutations", "999", "--equivalence-margin", "0.2", "--output", str(out / "cmp2.json")]) == 0
    cmp1 = json.loads((out / "cmp.json").read_text())
    cmp2 = json.loads((out / "cmp2.json").read_text())
    assert strip_volatile(cmp1["comparisons"]) == strip_volatile(cmp2["comparisons"])
    assert cmp1["analysis_mode"] == "inferential" and cmp1["provider_calls"] == 0
    assert all(c["sample_count"] == 20 for c in cmp1["comparisons"])
    assert group.repeat_group_id in cmp1["variability"]

    # 6-8. Blind preference tasks from one repetition vs model B, annotations, analysis.
    assert main(["preferences", "create", "--runs", resumed.records[0].run_id, run_b, *d, "--seed", "9",
                 "--output", str(out / "prefs"), "--key-output", str(tmp_path / "private" / "key.json")]) == 0
    key = json.loads((tmp_path / "private" / "key.json").read_text())
    blind = (out / "prefs" / "tasks.jsonl").read_text()
    assert "model-a" not in blind and "model-b" not in blind and run_b not in blind
    annotations = tmp_path / "ann.jsonl"
    rows = []
    for i, t in enumerate(key["tasks"]):
        for annotator, outcome in (("rater_1", "response_1"), ("rater_2", "tie" if i % 3 else "response_2")):
            rows.append({"annotation_id": f"{annotator}-{i}", "task_id": t["task_id"], "annotator_id": annotator, "outcome": outcome,
                         "timestamp": f"2026-11-10T10:{i:02d}:00Z"})
    annotations.write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert main(["preferences", "validate", "--key", str(tmp_path / "private" / "key.json"), "--annotations", str(annotations)]) == 0
    for fmt in ("json", "markdown", "csv"):
        assert main(["preferences", "analyze", "--key", str(tmp_path / "private" / "key.json"), "--annotations", str(annotations),
                     "--resamples", "300", "--elo", "--format", fmt, "--output", str(out / f"prefs.{fmt}")]) == 0
    pref = json.loads((out / "prefs.json").read_text())
    assert pref["annotation_count"] == 2 * len(key["tasks"]) and pref["annotator_count"] == 2

    # 9. Audit runs and the group.
    assert main(["runs", "verify", group.repeat_group_id, *d, "--quality-checks", "--audit-output", str(out / "group_audit.json")]) == 0
    audit = json.loads((out / "group_audit.json").read_text())
    assert audit["summary"]["errors"] == 0 and "EQ004" not in audit["summary"]["by_check"]

    # 10-11. Reports exist and are strict JSON; runs untouched; no provider calls since step 5.
    for path in out.rglob("*.json"):
        json.loads(path.read_text(), parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c)))
    assert snapshot(exp) == exp_before
    assert sum(CALLS.values()) == calls_before_analysis
    assert manager.verify_group(group.repeat_group_id)["valid"]
    assert manager.verify_run(run_b).valid
