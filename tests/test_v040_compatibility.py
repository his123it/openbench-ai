"""v0.4.0 compatibility contract (plan X1, F2).

* Every v0.3.1 public name keeps its signature: v0.4.0 only appends parameters that have
  defaults, and keeps every public member (frozen inventory from commit 8d3295a).
* `openbench evaluate` output equals the frozen v0.3.1 output (masked IDs, timestamps
  and the version string).
* v0.3.1 run directories go through every read path without any file being modified,
  created or removed.
* The `--ids-file` v0.1.0 reproduction still gives 0.9167 / 1.0 / 0.9167.
"""

from __future__ import annotations

import inspect
import json
import shutil
from pathlib import Path

import openbench
from _golden import masked_json_digest, masked_md_digest
from _v040_helpers import snapshot
from openbench.cli import main

ROOT = Path(__file__).resolve().parents[1]
SIGNATURES = json.loads((ROOT / "tests" / "data" / "v031_signatures.json").read_text())
GOLDEN = json.loads((ROOT / "tests" / "data" / "v031_evaluate_golden.json").read_text())


def _params(obj):
    try:
        sig = inspect.signature(obj)
    except (TypeError, ValueError):
        return None
    return [[p.name, str(p.kind), repr(p.default) if p.default is not inspect.Parameter.empty else "<required>"] for p in sig.parameters.values()]


def _assert_extends(old, new, where):
    if old is None:
        return
    assert new is not None, where
    assert new[: len(old)] == old, where
    for name, _kind, default in new[len(old):]:
        assert default != "<required>", f"{where}: new parameter {name} must have a default"


def test_public_signatures_only_extended():
    for name, entry in SIGNATURES.items():
        obj = getattr(openbench, name)
        assert type(obj).__name__ == entry["type"], name
        if "signature" in entry:
            _assert_extends(entry["signature"], _params(obj), name)
        if "init" in entry:
            _assert_extends(entry["init"], _params(obj), f"{name}()")
        for member, old in entry.get("members", {}).items():
            current = inspect.getattr_static(obj, member, None)
            assert current is not None, f"{name}.{member} removed"
            if old == "property":
                assert isinstance(current, property), f"{name}.{member}"
                continue
            if isinstance(current, (staticmethod, classmethod)):
                current = current.__func__
            _assert_extends(old, _params(current), f"{name}.{member}")


def test_new_public_names_are_appended_after_the_70():
    assert len(openbench.__all__) > 70
    assert set(openbench.__all__[:70]) == set(SIGNATURES)
    for name in openbench.__all__:
        assert hasattr(openbench, name), name


def test_evaluate_output_matches_v031_golden(tmp_path, monkeypatch):
    monkeypatch.chdir(ROOT)
    out = tmp_path / "eval"
    assert main([*GOLDEN["command"], "--output", str(out)]) == 0
    payload = (out / "alphabaseline-7b.json").read_text(encoding="utf-8")
    assert masked_json_digest(payload) == GOLDEN["json_sha256"]
    assert masked_md_digest((out / "alphabaseline-7b.md").read_text(encoding="utf-8")) == GOLDEN["markdown_sha256"]
    scores = {k: v["score"] for k, v in json.loads(payload)["aggregate_results"].items()}
    assert scores == GOLDEN["aggregate_scores"]


def test_v031_run_directories_through_every_read_path(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(ROOT)
    exp = tmp_path / "exp"
    shutil.copytree(ROOT / "tests" / "data" / "v031_run_fixture", exp)
    before = snapshot(exp)
    d = ["--experiments-dir", str(exp)]
    assert main(["runs", "list", *d, "--format", "json"]) == 0
    listed = json.loads(capsys.readouterr().out)
    assert {r["run_id"] for r in listed} == {"run_v031fixturea", "run_v031fixtureb"}
    assert all(r["lineage"]["seed_status"] == "unknown" for r in listed)
    assert main(["runs", "show", "run_v031fixtureb", *d]) == 0
    assert main(["runs", "verify", "run_v031fixturea", *d]) == 0
    assert main(["runs", "verify", "run_v031fixtureb", *d, "--quality-checks"]) == 0
    assert main(["runs", "compare", "run_v031fixturea", "run_v031fixtureb", *d]) == 0
    manager = openbench.ExperimentManager(exp)
    assert manager.load_evaluation("run_v031fixturea").model_name == "fixture-model-a"
    assert main(["analyze", "--runs", "run_v031fixturea", "run_v031fixtureb", *d, "--resamples", "200",
                 "--permutations", "999", "--output", str(tmp_path / "a.json")]) == 0
    assert main(["preferences", "create", "--runs", "run_v031fixturea", "run_v031fixtureb", *d,
                 "--output", str(tmp_path / "prefs")]) == 0
    capsys.readouterr()
    assert snapshot(exp) == before  # no file modified, created or removed


def test_v031_integrity_verifier_accepts_both_schema_versions():
    from openbench.integrity import SUPPORTED_PROVENANCE_SCHEMA_VERSIONS

    assert SUPPORTED_PROVENANCE_SCHEMA_VERSIONS == {"0.3.0", "0.4.0"}


def test_ids_file_v010_reproduction(tmp_path, monkeypatch):
    """The 12 v0.1.0 reasoning records (no metadata.version) with the v0.1.0 responses."""
    monkeypatch.chdir(ROOT)
    ids = tmp_path / "v0_1_0_ids.txt"
    lines = (ROOT / "datasets" / "reasoning" / "sample.jsonl").read_text(encoding="utf-8").splitlines()
    v010 = [json.loads(line)["id"] for line in lines if line.strip() and "version" not in json.loads(line).get("metadata", {})]
    assert len(v010) == 12
    ids.write_text("\n".join(v010) + "\n", encoding="utf-8")
    out = tmp_path / "r.json"
    assert main(["evaluate", "--dataset", "datasets", "--responses", "examples/sample_responses.json",
                 "--ids-file", str(ids), "--output", str(out)]) == 0
    scores = {k: round(v["score"], 4) for k, v in json.loads(out.read_text())["aggregate_results"].items()}
    assert scores == {"exact_match": 0.9167, "accuracy": 1.0, "keyword_coverage": 0.9167}
