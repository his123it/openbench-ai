"""Dataset and experiment quality audits (plan E1, E3, section 7)."""

from __future__ import annotations

import json
import random
import shutil
from pathlib import Path

import pytest

from _v040_helpers import snapshot
from openbench import (
    DatasetLoader,
    EvaluationConfig,
    ExecutionConfig,
    ExperimentManager,
    MockModelProvider,
)
from openbench.cli import main
from openbench.quality import (
    DatasetAuditor,
    ExperimentAuditor,
    jaccard,
    normalise_prompt,
    shingles,
)

ROOT = Path(__file__).resolve().parents[1]


def _rec(rid, prompt="What is the capital of France?", answer="Paris", category="geo", **meta):
    return {"id": rid, "category": category, "prompt": prompt, "expected_answer": answer, "metadata": meta}


def _write(path: Path, rows) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(r if isinstance(r, str) else json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    return path


def _audit(path, **kw):
    kw.setdefault("use_review_notes", False)
    return DatasetAuditor(**kw).audit(path).to_dict()


def _checks(report):
    return [f["check_id"] for f in report["findings"]]


def _many(n, start=0):
    return [_rec(f"r{i:03d}", prompt=f"Unique question number {i} about topic {i * 7919 % 1000}?", answer=f"a{i}") for i in range(start, start + n)]


def test_clean_dataset_has_no_findings(tmp_path):
    rep = _audit(_write(tmp_path / "d.jsonl", _many(40)))
    assert rep["findings"] == [] and rep["summary"]["errors"] == 0


@pytest.mark.parametrize(("rows", "check", "severity"), [
    ([_rec("a"), _rec("a", prompt="Other?")], "DQ001", "error"),
    ([_rec("a"), _rec("b", prompt="  what is THE capital   of france? ")], "DQ002", "warning"),
    ([_rec("a", prompt="")], "DQ004", "error"),
    ([_rec("a", prompt=5)], "DQ004", "error"),
    ([{"id": "a", "category": "c", "prompt": "Why?"}], "DQ005", "error"),
    ([_rec("a", answer=7)], "DQ005", "error"),
    ([_rec("a", answer="  ")], "DQ006", "warning"),
    ([_rec("a", answer="TBD")], "DQ006", "warning"),
    ([_rec("a", answer="N/A")], "DQ006", "warning"),
    ([_rec("a", answer="Paris", keywords=["Paris", "France"])], "DQ007", "warning"),
    ([_rec("a", expected_schema="not an object")], "DQ008", "error"),
    ([_rec("a", version="0.1.0"), _rec("b", prompt="Q2?", version="0.2.0")], "DQ009a", "warning"),
    ([_rec("a", version="0.1.0"), _rec("b", prompt="Q2?")], "DQ009b", "info"),
    ([_rec("a", difficulty="extreme")], "DQ009c", "warning"),
    (["{not json"], "DQ012", "error"),
    (["[1, 2]"], "DQ012", "error"),
    ([{"category": "c", "prompt": "p", "expected_answer": "x"}], "DQ012", "error"),
    ([_rec("a", category="")], "DQ012", "error"),
])
def test_each_dataset_check_fires(tmp_path, rows, check, severity):
    rows = rows + _many(40, start=100)
    rep = _audit(_write(tmp_path / "d.jsonl", rows))
    hits = [f for f in rep["findings"] if f["check_id"] == check]
    assert hits, _checks(rep)
    assert hits[0]["severity"] == severity
    assert hits[0]["remediation"] or check == "DQ012"


def test_short_answers_are_not_placeholders(tmp_path):
    rep = _audit(_write(tmp_path / "d.jsonl", [_rec("a", answer="2"), _rec("b", prompt="Q?", answer="no")] + _many(40)))
    assert "DQ006" not in _checks(rep)


def test_category_vs_directory_and_cross_file_duplicate_ids(tmp_path):
    _write(tmp_path / "ds" / "geo" / "sample.jsonl", [_rec("a")] + _many(20))
    _write(tmp_path / "ds" / "math" / "sample.jsonl", [_rec("a", prompt="2+2?", category="geo")] + _many(20, start=500))
    rep = _audit(tmp_path / "ds")
    checks = _checks(rep)
    assert "DQ001" in checks and "DQ009d" in checks


def test_normalisation_nfkc_casefold_and_accents_kept():
    assert normalise_prompt("  ＡＢＣ\tSTRASSE  ") == normalise_prompt("abc strasse")
    assert normalise_prompt("Straße") == normalise_prompt("STRASSE")
    assert normalise_prompt("Gödel") != normalise_prompt("Godel")
    assert normalise_prompt("a, b") != normalise_prompt("a b")  # punctuation kept


def test_near_duplicates_threshold_boundary_and_short_prompts(tmp_path):
    base = "The quick brown fox jumps over the lazy dog near the river bank today"
    variant = base.replace("today", "now")
    sim = jaccard(shingles(normalise_prompt(base)), shingles(normalise_prompt(variant)))
    rows = [_rec("a", prompt=base), _rec("b", prompt=variant), _rec("c", prompt="Hi?"), _rec("d", prompt="Hi?")] + _many(40)
    at = _audit(_write(tmp_path / "d.jsonl", rows), near_duplicate_threshold=sim)
    above = _audit(_write(tmp_path / "e.jsonl", rows), near_duplicate_threshold=min(1.0, sim + 1e-6))
    dq3 = [f for f in at["findings"] if f["check_id"] == "DQ003"]
    assert dq3 and dq3[0]["record_ids"] == ["a", "b"] and "heuristic" in dq3[0]["message"]
    assert "DQ003" not in _checks(above)
    assert at["summary"]["short_prompts_not_compared"] == 2
    assert "DQ002" in _checks(at)  # short exact duplicates still caught


def test_prefix_filtering_matches_brute_force():
    rng = random.Random(3)
    words = ["alpha", "beta", "gamma", "delta", "eps", "zeta", "eta", "theta"]
    prompts = {f"p{i:02d}": " ".join(rng.choice(words) for _ in range(rng.randrange(3, 9))) for i in range(60)}
    sets = {k: shingles(normalise_prompt(v)) for k, v in prompts.items() if len(normalise_prompt(v)) >= 5}
    auditor = DatasetAuditor(near_duplicate_threshold=0.5, use_review_notes=False)
    fast = auditor._similar_pairs(sets)
    keys = sorted(sets)
    brute = {(a, b): jaccard(sets[a], sets[b]) for i, a in enumerate(keys) for b in keys[i + 1:]}
    assert {p for p, s in brute.items() if s >= 0.3} == set(fast)
    for p, s in fast.items():
        assert s == pytest.approx(brute[p])


def test_near_duplicate_skip_is_reported_not_silent(tmp_path):
    rep = _audit(_write(tmp_path / "d.jsonl", _many(40)), max_records_for_pairwise=10)
    skipped = [f for f in rep["findings"] if f["check_id"] == "DQ014"]
    assert skipped and skipped[0]["severity"] == "info"
    assert rep["summary"]["near_duplicate_check"] == "skipped"


def test_long_prompts_are_skipped_with_info(tmp_path, monkeypatch):
    import openbench.quality as q

    monkeypatch.setattr(q, "MAX_PROMPT_CHARS_FOR_SHINGLES", 50)
    rows = [_rec("long", prompt="x" * 200)] + _many(40)
    rep = _audit(_write(tmp_path / "d.jsonl", rows))
    assert any(f["check_id"] == "DQ014" and f["record_ids"] == ["long"] for f in rep["findings"])


def test_small_dataset_findings(tmp_path):
    assert [f["severity"] for f in _audit(_write(tmp_path / "a.jsonl", _many(5)))["findings"]] == ["warning"]
    assert [f["severity"] for f in _audit(_write(tmp_path / "b.jsonl", _many(20)))["findings"]] == ["info"]


def test_metric_configuration_checks():
    rep = DatasetAuditor(metrics=["exact_match", "no_such_metric"], metric_options={"exact_match": {"bogus": 1}, "f1": {}}, use_review_notes=False).audit(ROOT / "datasets" / "coding" / "sample.jsonl").to_dict()
    dq8 = [f for f in rep["findings"] if f["check_id"] == "DQ008"]
    assert len(dq8) == 3 and all(f["severity"] == "error" for f in dq8)


def test_audit_never_modifies_files_and_is_deterministic(tmp_path):
    path = _write(tmp_path / "d.jsonl", [_rec("a"), _rec("a")] + _many(30))
    before = snapshot(tmp_path)
    one = _audit(path)
    two = _audit(path)
    assert snapshot(tmp_path) == before
    for r in (one, two):
        r.pop("audited_at")
    assert one == two


def test_review_notes_only_apply_to_matching_content(tmp_path):
    rows = [json.loads(line) for line in (ROOT / "datasets" / "coding" / "sample.jsonl").read_text().splitlines()]
    edited = [dict(r) for r in rows]
    for r in edited:
        if r["id"] == "coding_003":
            r["prompt"] = r["prompt"] + " "
    path = _write(tmp_path / "coding" / "sample.jsonl", edited)
    rep = DatasetAuditor().audit(path).to_dict()
    f = next(f for f in rep["findings"] if f["record_ids"] == ["coding_003"])
    assert f["check_id"] == "DQ007" and f["classification"] is None  # content changed: note not applied
    other = next(f for f in rep["findings"] if f["record_ids"] == ["coding_007"])
    assert other["classification"] == "documented_historical_exception"


# -- experiment audits ------------------------------------------------------------------


def _fixture_runs(tmp_path) -> ExperimentManager:
    exp = tmp_path / "exp"
    shutil.copytree(ROOT / "tests" / "data" / "v031_run_fixture", exp)
    return ExperimentManager(exp)


def test_experiment_audit_legacy_partial_run(tmp_path, monkeypatch):
    monkeypatch.chdir(ROOT)
    manager = _fixture_runs(tmp_path)
    before = snapshot(manager.root)
    rep = ExperimentAuditor(manager).audit_run("run_v031fixtureb").to_dict()
    checks = _checks(rep)
    assert {"EQ001", "EQ002", "EQ009", "EQ005"} <= set(checks)
    assert "EQ003" not in checks
    assert snapshot(manager.root) == before


def test_experiment_audit_detects_tampering_and_invalid_metrics(tmp_path):
    manager = _fixture_runs(tmp_path)
    path = manager.root / "run_v031fixturea" / "evaluation.json"
    data = json.loads(path.read_text())
    data["individual_results"][0]["metrics"]["exact_match"]["score"] = 7.0
    path.write_text(json.dumps(data))
    rep = ExperimentAuditor(manager).audit_run("run_v031fixturea").to_dict()
    assert {"EQ003", "EQ008"} <= set(_checks(rep))
    assert rep["summary"]["errors"] >= 2


def test_experiment_audit_unreadable_config(tmp_path):
    manager = _fixture_runs(tmp_path)
    (manager.root / "run_v031fixturea" / "config.json").write_text("{broken")
    rep = ExperimentAuditor(manager).audit_run("run_v031fixturea").to_dict()
    assert _checks(rep) == ["EQ011"]


def test_group_audit_incomplete_and_inconsistent(tmp_path):
    ds = DatasetLoader.load(str(ROOT / "datasets" / "reasoning" / "sample.jsonl"))
    manager = ExperimentManager(tmp_path / "exp")
    group = manager.execute_repeated(ds, MockModelProvider(), 3, evaluation_config=EvaluationConfig(metric_names=["exact_match"]), execution_config=ExecutionConfig(use_cache=False))
    shutil.rmtree(group.records[2].run_dir)
    rep = ExperimentAuditor(manager).audit_group(group.repeat_group_id).to_dict()
    assert "EQ004" in _checks(rep) and "EQ010" not in _checks(rep)
    clone = Path(group.records[0].run_dir).with_name("run_clone0000")
    shutil.copytree(group.records[0].run_dir, clone)
    cfg = json.loads((clone / "config.json").read_text())
    cfg["run_id"] = "run_clone0000"
    (clone / "config.json").write_text(json.dumps(cfg))
    rep2 = ExperimentAuditor(manager).audit_group(group.repeat_group_id).to_dict()
    assert "EQ010" in _checks(rep2)


def test_external_evaluation_unknown_ids(tmp_path):
    from openbench import Evaluator, ModelResponseSet, ResponseItem

    ds = DatasetLoader.load(str(ROOT / "datasets" / "reasoning" / "sample.jsonl"))
    rs = ModelResponseSet("ext", "1", [ResponseItem(id="reasoning_001", response="x"), ResponseItem(id="zzz_999", response="y")])
    ev = Evaluator(config=EvaluationConfig(metric_names=["exact_match"])).evaluate(ds, rs)
    manager = ExperimentManager(tmp_path / "exp")
    rec = manager.register_external_evaluation(ds, rs, ev)
    checks = _checks(ExperimentAuditor(manager).audit_run(rec.run_id).to_dict())
    assert {"EQ001", "EQ007", "EQ012"} <= set(checks)


def test_cli_quality_checks(tmp_path, capsys, monkeypatch):
    bad = _write(tmp_path / "bad.jsonl", [_rec("a"), _rec("a")] + _many(30))
    assert main(["validate-dataset", str(bad), "--quality-checks", "--output", str(tmp_path / "a.json")]) == 1
    assert json.loads((tmp_path / "a.json").read_text())["summary"]["errors"] == 1
    good = _write(tmp_path / "good.jsonl", [_rec("a", answer="x", keywords=["y"])] + _many(30))
    assert main(["validate-dataset", str(good), "--quality-checks"]) == 0  # warnings only: exit 0 (D18)
    monkeypatch.chdir(ROOT)
    manager = _fixture_runs(tmp_path)
    assert main(["runs", "verify", "run_v031fixturea", "--experiments-dir", str(manager.root), "--quality-checks"]) == 0
    assert main(["runs", "verify", "run_v031fixtureb", "--experiments-dir", str(manager.root), "--quality-checks",
                 "--format", "json", "--audit-output", str(tmp_path / "eq.json")]) == 0
    assert json.loads((tmp_path / "eq.json").read_text())["kind"] == "experiment_audit"
