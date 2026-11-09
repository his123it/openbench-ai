"""Human preference evaluation: blinding, import validation, statistics and agreement
(plan D1-D5, section 6)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from openbench import preferences as prefs
from openbench.cli import main
from openbench.exceptions import ConfigurationError

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "datasets" / "reasoning" / "sample.jsonl"
BLIND_FIELDS = {"schema_version", "task_id", "prompt", "response_1", "response_2", "instructions", "rubric", "reference"}


def _source(sid, name, version, provider, n=6, wrong=(), invalid=(), text=None):
    items = {}
    for i in range(1, n + 1):
        rid = f"rec_{i:02d}"
        items[rid] = {
            "prompt": f"Prompt {i}?",
            "response": (text or {}).get(rid, f"{'bad' if i in wrong else 'good'} answer {i}"),
            "reference": f"reference {i}",
            "category": "c1" if i % 2 else "c2",
            "difficulty": "easy" if i <= 3 else "hard",
            "valid": i not in invalid,
        }
    return prefs.PreferenceSource(sid, name, version, provider, items)


A = _source("run_aaaa1111", "alpha-model-x", "2025.06", "ollama")
B = _source("run_bbbb2222", "beta-model-y", "rev-b", "huggingface_transformers", wrong=(2, 4))


def _build(seed=0, **kw):
    return prefs.PreferenceTaskBuilder(seed, **kw).build(A, B)


# -- blinding -------------------------------------------------------------------


def test_tasks_contain_no_identity_fields_or_strings():
    ts = _build(seed=7)
    identity = ["alpha-model-x", "beta-model-y", "2025.06", "rev-b", "ollama", "huggingface", "run_aaaa1111", "run_bbbb2222"]
    for task in ts.tasks:
        assert set(task) <= BLIND_FIELDS
        assert "reference" not in task
        blob = json.dumps(task).lower()
        for s in identity:
            assert s.lower() not in blob
        assert "rec_" not in blob  # record IDs stay in the key
        assert "AB" not in json.dumps(task) and "BA" not in json.dumps(task)
    assert ts.leakage_warnings == []


def test_key_holds_identity_and_order_reproducible_from_seed():
    one, two = _build(seed=11), _build(seed=11)
    assert one.tasks == two.tasks and one.key == two.key
    orders = [t["display_order"] for t in one.key["tasks"]]
    assert set(orders) <= {"AB", "BA"}
    other = _build(seed=12)
    assert [t["display_order"] for t in other.key["tasks"]] != orders or other.key["tasks"][0]["task_id"] != one.key["tasks"][0]["task_id"]
    assert one.key["models"]["A"]["model_name"] == "alpha-model-x"
    for task, entry in zip(one.tasks, one.key["tasks"], strict=True):
        expected_first = A.items[entry["record_id"]]["response"] if entry["display_order"] == "AB" else B.items[entry["record_id"]]["response"]
        assert task["response_1"] == expected_first
        assert task["task_id"] == entry["task_id"]


def test_eligibility_excludes_invalid_and_counts():
    b = _source("run_b", "m-b", "1", "p", invalid=(3,))
    ts = prefs.PreferenceTaskBuilder(0).build(A, b)
    assert len(ts.tasks) == 5
    assert ts.key["excluded_counts"]["missing_or_invalid_in_b"] == 1


def test_leakage_audit_warns_but_never_edits_outputs():
    leaky = _source("run_cccc3333", "gamma-model", "v9", "local_lookup", text={"rec_01": "As gamma-model I think yes"})
    ts = prefs.PreferenceTaskBuilder(0).build(A, leaky)
    assert ts.leakage_warnings and ts.leakage_warnings[0]["task_ids"]
    assert any("As gamma-model I think yes" in (t["response_1"], t["response_2"]) for t in ts.tasks)
    assert "gamma-model" not in json.dumps(ts.leakage_warnings)


def test_show_reference_rubric_and_files(tmp_path):
    rubric_file = tmp_path / "rubric.json"
    rubric_file.write_text(json.dumps({"criteria": [{"name": "helpfulness", "min": 1, "max": 5}], "instructions": "Be fair."}))
    rubric, instructions = prefs.load_rubric(rubric_file)
    ts = prefs.PreferenceTaskBuilder(3, rubric, instructions, show_reference=True).build(A, B)
    assert all(t["reference"].startswith("reference") and t["instructions"] == "Be fair." for t in ts.tasks)
    written = prefs.write_task_set(ts, tmp_path / "out", write_csv=True)
    assert Path(written["key"]).name == "tasks.key.json"
    tasks_text = Path(written["tasks"]).read_text()
    assert "alpha-model-x" not in tasks_text and "display_order" not in tasks_text
    key = prefs.load_key(written["key"])
    import hashlib

    assert key["tasks_jsonl_sha256"] == hashlib.sha256(tasks_text.encode()).hexdigest()
    with pytest.raises(ConfigurationError, match="overwrite"):
        prefs.write_task_set(ts, tmp_path / "out")
    with pytest.raises(ConfigurationError, match="separate"):
        prefs.write_task_set(ts, tmp_path / "o2", key_path=tmp_path / "o2" / "tasks.jsonl")


@pytest.mark.parametrize("bad", [
    {"criteria": []}, {"criteria": [{"name": "x", "min": 5, "max": 1}]},
    {"criteria": [{"name": "x", "min": 1, "max": 5, "extra": 1}]}, {"criteria": [{"name": "a b", "min": 1, "max": 2}]},
    {"criteria": [{"name": "x", "min": 1, "max": 2}], "evil": "__import__('os')"},
])
def test_invalid_rubrics_rejected(tmp_path, bad):
    f = tmp_path / "r.json"
    f.write_text(json.dumps(bad))
    with pytest.raises(ConfigurationError):
        prefs.load_rubric(f)


def test_csv_injection_neutralised():
    assert prefs.csv_safe("=HYPERLINK(1)") == "'=HYPERLINK(1)"
    for prefix in ("+", "-", "@", "\t", "\r"):
        assert prefs.csv_safe(prefix + "x").startswith("'")
    assert prefs.csv_safe("plain") == "plain"
    evil = _source("run_e", "m", "1", "p", text={"rec_01": "=cmd|' /C calc'!A0"})
    csv_text = prefs.PreferenceTaskBuilder(0).build(A, evil).tasks_csv()
    assert "'=cmd" in csv_text


# -- import validation ----------------------------------------------------------


def _key(seed=0):
    return _build(seed).key


def _ann(task_id, annotator="ann1", outcome="response_1", aid=None, **extra):
    return {"annotation_id": aid or f"{annotator}-{task_id}", "task_id": task_id, "annotator_id": annotator, "outcome": outcome, **extra}


def _write(tmp_path, rows, name="ann.jsonl"):
    path = tmp_path / name
    path.write_text("\n".join(r if isinstance(r, str) else json.dumps(r) for r in rows) + "\n")
    return path


def test_valid_annotations_load(tmp_path):
    key = _key()
    tids = [t["task_id"] for t in key["tasks"]]
    rows = [_ann(t, "ann1", "tie", rationale="ok", confidence=0.5, timestamp="2026-11-01T10:00:00Z") for t in tids]
    rep = prefs.AnnotationImporter(key).load([_write(tmp_path, rows)])
    assert rep.valid and len(rep.annotations) == len(tids)


@pytest.mark.parametrize(("row", "code"), [
    ({"extra": 1}, "unknown_fields"),
    ({"outcome": "model_a"}, "invalid_outcome"),
    ({"outcome": None}, "missing_outcome"),
    ({"task_id": "deadbeefdeadbeef"}, "unknown_task_id"),
    ({"annotator_id": "bad id!"}, "invalid_annotator_id"),
    ({"rationale": "x" * 5001}, "rationale_too_long"),
    ({"confidence": 2}, "invalid_confidence"),
    ({"timestamp": "yesterday"}, "invalid_timestamp"),
    ({"metadata": {"a": {"nested": 1}}}, "invalid_metadata"),
    ({"schema_version": "0.3.0"}, "schema_version_mismatch"),
    ({"rubric_scores": {"helpfulness": 3}}, "unknown_rubric_criterion"),
])
def test_invalid_annotation_fields_are_errors(tmp_path, row, code):
    key = _key()
    base = _ann(key["tasks"][0]["task_id"])
    base.update(row)
    if row.get("outcome", "x") is None:
        base["outcome"] = None
    rep = prefs.AnnotationImporter(key).load([_write(tmp_path, [base])])
    assert not rep.valid
    assert code in {e["code"] for e in rep.errors}


def test_duplicates_malformed_lines_roster_and_email_warning(tmp_path):
    key = _key()
    t0, t1 = key["tasks"][0]["task_id"], key["tasks"][1]["task_id"]
    rows = [_ann(t0, aid="x1"), _ann(t0, aid="x2"), _ann(t1, aid="x1"), "{not json", "[1, 2]",
            _ann(t1, "user@example.com", aid="x9")]
    rep = prefs.AnnotationImporter(key, annotator_roster=["ann1"]).load([_write(tmp_path, rows)])
    codes = {e["code"] for e in rep.errors}
    assert {"duplicate_annotation", "duplicate_annotation_id", "parse_error", "not_an_object"} <= codes
    assert "invalid_annotator_id" in codes  # '@' is not allowed in pseudonymous IDs


def test_unknown_annotator_against_roster(tmp_path):
    key = _key()
    rep = prefs.AnnotationImporter(key, annotator_roster=["someone_else"]).load([_write(tmp_path, [_ann(key["tasks"][0]["task_id"])])])
    assert "unknown_annotator" in {e["code"] for e in rep.errors}


def test_rubric_scores_range_and_csv_import(tmp_path):
    rubric = [prefs.RubricCriterion("helpfulness", 1, 5)]
    ts = prefs.PreferenceTaskBuilder(0, rubric).build(A, B)
    t = [x["task_id"] for x in ts.key["tasks"]]
    csv_path = tmp_path / "ann.csv"
    csv_path.write_text(
        "annotation_id,task_id,annotator_id,outcome,rubric:helpfulness,confidence,metadata\n"
        f"a1,{t[0]},ann1,response_2,4,0.9,\"{{\"\"batch\"\": 1}}\"\n"
        f"a2,{t[1]},ann1,neither,7,,\n"
    )
    rep = prefs.AnnotationImporter(ts.key).load([csv_path])
    assert [e["code"] for e in rep.errors] == ["rubric_score_out_of_range"]
    assert rep.annotations[0].rubric_scores == {"helpfulness": 4.0} and rep.annotations[0].metadata == {"batch": 1}
    exported = prefs.export_annotations_csv(rep.annotations, ["helpfulness"])
    assert exported.splitlines()[0].endswith("rubric:helpfulness")


def test_import_size_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(prefs, "MAX_IMPORT_BYTES", 10)
    with pytest.raises(ConfigurationError, match="limit"):
        prefs.AnnotationImporter(_key()).load([_write(tmp_path, [_ann("x" * 16)])])


def test_imported_text_is_never_executed(tmp_path):
    key = _key()
    payload = "__import__('os').system('touch HACKED')"
    rep = prefs.AnnotationImporter(key).load([_write(tmp_path, [_ann(key["tasks"][0]["task_id"], rationale=payload)])])
    assert rep.valid and rep.annotations[0].rationale == payload
    assert not (tmp_path / "HACKED").exists() and not Path("HACKED").exists()


# -- statistics -----------------------------------------------------------------


def _analyze(key, rows, **kw):
    anns = [prefs.PreferenceAnnotation(**r) for r in rows]
    return prefs.PreferenceAnalyzer([key], resamples=500, **kw).analyze(anns, elo=True)


def _model_outcome_to_display(key, task_id, model_outcome):
    order = next(t["display_order"] for t in key["tasks"] if t["task_id"] == task_id)
    if model_outcome in ("tie", "neither"):
        return model_outcome
    return {"AB": {"A": "response_1", "B": "response_2"}, "BA": {"A": "response_2", "B": "response_1"}}[order][model_outcome]


def test_task_weighted_win_rate_neither_and_unannotated():
    key = _key(seed=5)
    t = [x["task_id"] for x in key["tasks"]]
    plan = [  # (task, annotator, model-space outcome)
        (t[0], "a1", "A"), (t[0], "a2", "A"), (t[0], "a3", "B"),  # task value 2/3
        (t[1], "a1", "tie"),                                       # 0.5
        (t[2], "a1", "neither"), (t[2], "a2", "neither"),           # only neither: excluded
        (t[3], "a2", "B"),                                         # 0
    ]
    rows = [dict(annotation_id=f"n{i}", task_id=tid, annotator_id=ann, outcome=_model_outcome_to_display(key, tid, o))
            for i, (tid, ann, o) in enumerate(plan)]
    out = _analyze(key, rows)
    overall = out["pairs"][0]["overall"]
    assert overall["win_rate_a_task_weighted"] == pytest.approx((2 / 3 + 0.5 + 0) / 3)
    assert overall["win_rate_a_annotation_weighted"] == pytest.approx((1 + 1 + 0 + 0.5 + 0) / 5)
    assert overall["tasks_only_neither"] == 1
    assert overall["outcome_counts"] == {"A": 2, "B": 2, "tie": 1, "neither": 2}
    assert overall["neither_rate"] == pytest.approx(2 / 7)
    assert out["unannotated_tasks"] == len(t) - 4
    assert overall["win_rate_a_confidence_interval"]["sample_count"] == 3  # unit = task
    json.dumps(out, allow_nan=False)


def test_annotator_imbalance_and_position_bias():
    key = _key()
    t = [x["task_id"] for x in key["tasks"]]
    rows = [dict(annotation_id=f"i{i}", task_id=tid, annotator_id="a1", outcome="response_1") for i, tid in enumerate(t)]
    rows.append(dict(annotation_id="z", task_id=t[0], annotator_id="a2", outcome="response_1"))
    out = _analyze(key, rows)
    assert any(w["code"] == "annotator_imbalance" for w in out["warnings"])
    assert out["position_bias"]["share_choosing_response_1"] == 1.0
    assert out["position_bias"]["warnings"][0]["code"] == "position_bias"


def test_elo_three_updates_hand_computed():
    key = _key()
    t = [x["task_id"] for x in key["tasks"]]
    seq = [("A", "2026-01-01T00:00:01Z"), ("tie", "2026-01-01T00:00:02Z"), ("B", "2026-01-01T00:00:03Z")]
    rows = [dict(annotation_id=f"e{i}", task_id=t[i], annotator_id="a1", outcome=_model_outcome_to_display(key, t[i], o), timestamp=ts)
            for i, (o, ts) in enumerate(seq)]
    out = _analyze(key, rows)
    ra = rb = 1000.0
    for o, _ in seq:
        ea = 1 / (1 + 10 ** ((rb - ra) / 400))
        sa = {"A": 1.0, "tie": 0.5, "B": 0.0}[o]
        ra, rb = ra + 32 * (sa - ea), rb + 32 * ((1 - sa) - (1 - ea))
    assert out["elo"]["ratings"]["run_aaaa1111"] == pytest.approx(ra)
    assert out["elo"]["ratings"]["run_bbbb2222"] == pytest.approx(rb)
    assert "not absolute measures" in out["elo"]["limitations"]
    assert out["elo"]["warnings"][0]["code"] == "single_ordering"


# -- agreement ------------------------------------------------------------------


def test_cohens_kappa_hand_case():
    assert prefs.cohens_kappa(list("aabbab"), list("abbbaa")) == pytest.approx(1 / 3)
    assert prefs.cohens_kappa(["a"], ["a"]) is None
    assert prefs.cohens_kappa(["a", "a"], ["a", "a"]) is None


def test_fleiss_kappa_fleiss_1971_worked_example():
    table = [
        [0, 0, 0, 0, 14], [0, 2, 6, 4, 2], [0, 0, 3, 5, 6], [0, 3, 9, 2, 0], [2, 2, 8, 1, 1],
        [7, 7, 0, 0, 0], [3, 2, 6, 3, 0], [2, 5, 3, 2, 2], [6, 5, 2, 1, 0], [0, 2, 2, 3, 7],
    ]
    assert prefs.fleiss_kappa(table) == pytest.approx(0.20993, abs=1e-5)
    with pytest.raises(ValueError):
        prefs.fleiss_kappa([[1, 1], [2, 1]])


def test_agreement_same_percent_in_both_spaces():
    key = _key(seed=9)
    t = [x["task_id"] for x in key["tasks"]]
    outcomes = {"a1": ["response_1", "response_2", "tie", "response_1", "response_2", "neither"],
                "a2": ["response_1", "response_1", "tie", "response_2", "response_2", "neither"]}
    rows = [dict(annotation_id=f"{a}{i}", task_id=t[i], annotator_id=a, outcome=o)
            for a, outs in outcomes.items() for i, o in enumerate(outs)]
    out = _analyze(key, rows)
    disp, model = out["agreement"]["display_space"], out["agreement"]["model_space"]
    assert disp["percent_agreement"] == model["percent_agreement"] == pytest.approx(4 / 6)
    assert disp["cohens_kappa_pairs"][0]["shared_tasks"] == 6
    assert disp["fleiss_ratings_per_task"] == 2 and disp["fleiss_tasks_used"] == 6
    assert math.isfinite(disp["cohens_kappa_pairs"][0]["cohens_kappa"])


# -- CLI ------------------------------------------------------------------------


def test_cli_create_validate_analyze(tmp_path, capsys):
    exp = tmp_path / "exp"
    ids = []
    for model, extra in (("model-one", []), ("model-two", ["--metrics", "exact_match"])):
        assert main(["run", "--dataset", str(DATASET), "--provider", "mock", "--model", model, "--no-cache",
                     "--experiments-dir", str(exp), *extra]) == 0
        ids.append(capsys.readouterr().out.split("Run '")[1].split("'")[0])
    out_dir = tmp_path / "tasks"
    assert main(["preferences", "create", "--runs", *ids, "--experiments-dir", str(exp), "--seed", "4",
                 "--output", str(out_dir), "--csv"]) == 0
    tasks = [json.loads(line) for line in (out_dir / "tasks.jsonl").read_text().splitlines()]
    assert len(tasks) == 20
    text = (out_dir / "tasks.jsonl").read_text() + (out_dir / "tasks.csv").read_text()
    for s in ("model-one", "model-two", *ids):
        assert s not in text
    key = json.loads((out_dir / "tasks.key.json").read_text())
    anns = tmp_path / "ann.jsonl"
    anns.write_text("".join(json.dumps(_ann(t["task_id"], a, "response_1", aid=f"{a}-{i}")) + "\n"
                            for i, t in enumerate(key["tasks"]) for a in ("r1", "r2")))
    assert main(["preferences", "validate", "--key", str(out_dir / "tasks.key.json"), "--annotations", str(anns)]) == 0
    assert main(["preferences", "analyze", "--key", str(out_dir / "tasks.key.json"), "--annotations", str(anns),
                 "--resamples", "300", "--elo", "--output", str(tmp_path / "pa.json")]) == 0
    result = json.loads((tmp_path / "pa.json").read_text())
    assert result["annotation_count"] == 40 and result["elo"] is not None
    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps({"annotation_id": "x", "task_id": "nope", "annotator_id": "r1", "outcome": "A"}) + "\n")
    assert main(["preferences", "validate", "--key", str(out_dir / "tasks.key.json"), "--annotations", str(bad)]) == 1
    assert main(["preferences", "analyze", "--key", str(out_dir / "tasks.key.json"), "--annotations", str(bad)]) == 1
    assert main(["preferences", "create", "--runs", *ids, "--experiments-dir", str(exp), "--output", str(out_dir)]) == 1
