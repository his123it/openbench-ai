"""v0.4.0 security review tests (plan X2; roadmap Stage 6 B)."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from openbench import (
    DatasetLoader,
    EvaluationConfig,
    ExecutionConfig,
    ExperimentManager,
    GenerationConfig,
    MockModelProvider,
    ResponseCache,
)
from openbench.cli import main

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "datasets" / "reasoning" / "sample.jsonl"
SECRETS = ["sk-REALSECRETabcdefgh1234", "hf_SECRETTOKENabcdefgh", "SUPERSECRETPASSWORD99"]


def _all_text(*roots: Path) -> str:
    chunks = []
    for root in roots:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                chunks.append(path.read_text(encoding="utf-8", errors="replace"))
    return "\n".join(chunks)


class LeakyProvider(MockModelProvider):
    def generate_one(self, record):
        if record.id == "reasoning_002":
            raise RuntimeError(f"upstream said Authorization: Bearer {SECRETS[0]} and password={SECRETS[2]}")
        return super().generate_one(record)


def test_secrets_never_persisted_in_runs_caches_logs_or_reports(tmp_path, capsys):
    gen = GenerationConfig(extra_options={"api_key": SECRETS[0], "hf_token": SECRETS[1], "note": f"password={SECRETS[2]}"})
    provider = LeakyProvider(model_name="leaky", generation_config=gen)
    manager = ExperimentManager(tmp_path / "exp")
    cache = ResponseCache(tmp_path / "cache")
    ds = DatasetLoader.load(str(DATASET))
    group = manager.execute_repeated(ds, provider, 2, generation_base_seed=1, evaluation_config=EvaluationConfig(metric_names=["exact_match"]),
                                     execution_config=ExecutionConfig(use_cache=True), cache=cache)
    single, _, _ = manager.execute_and_record(ds, MockModelProvider(model_name="clean"), evaluation_config=EvaluationConfig(metric_names=["exact_match"]),
                                              execution_config=ExecutionConfig(use_cache=False))
    out = tmp_path / "reports"
    for fmt in ("json", "markdown", "csv"):
        assert main(["analyze", "--runs", group.repeat_group_id, single.run_id, "--experiments-dir", str(tmp_path / "exp"),
                     "--resamples", "200", "--permutations", "999", "--format", fmt, "--output", str(out / f"a.{fmt}")]) == 0
    assert main(["runs", "verify", group.repeat_group_id, "--experiments-dir", str(tmp_path / "exp"), "--quality-checks",
                 "--audit-output", str(out / "audit.json")]) in (0, 1)
    assert main(["preferences", "create", "--runs", group.records[0].run_id, single.run_id, "--experiments-dir",
                 str(tmp_path / "exp"), "--output", str(out / "prefs")]) == 0
    text = _all_text(tmp_path / "exp", tmp_path / "cache", out) + capsys.readouterr().out
    for secret in SECRETS:
        assert secret not in text, secret
    assert "[REDACTED" in text


def test_cli_error_messages_are_scrubbed(tmp_path, capsys):
    secret = "tok3nVALUEabc123456"
    code = main(["analyze", "--results", f"/nonexistent/a.json?token={secret}", "/nonexistent/b.json"])
    err = capsys.readouterr().err
    assert code == 1 and secret not in err and "[REDACTED]" in err


@pytest.mark.parametrize("bad", ["../../etc/passwd", "run_/../../x", "run_a\\b", "run_a\x00b"])
def test_path_traversal_run_ids_rejected(tmp_path, bad, capsys):
    (tmp_path / "exp").mkdir()
    assert main(["runs", "show", bad, "--experiments-dir", str(tmp_path / "exp")]) == 1
    assert main(["runs", "verify", bad, "--experiments-dir", str(tmp_path / "exp")]) == 1


def test_group_ids_and_output_paths_validated(tmp_path):
    from openbench.exceptions import ConfigurationError
    from openbench.lineage import validate_repeat_group_id
    from openbench.security import validate_safe_output_path

    for bad in ("grp_../../x", "grp_", "grp_a/b", "../grp_a"):
        with pytest.raises(ConfigurationError):
            validate_repeat_group_id(bad)
    with pytest.raises(ConfigurationError):
        validate_safe_output_path("out\x00.json")
    assert main(["analyze", "--results", "a.json", "b.json", "--output", "x\x00y"]) == 1


def test_no_unsafe_deserialisation_or_code_execution_in_library():
    pattern = re.compile(r"\b(pickle|marshal|shelve|eval\(|exec\(|yaml\.load\(|yaml\.unsafe_load|subprocess|os\.system|__import__)")
    hits = []
    for path in sorted((ROOT / "src" / "openbench").rglob("*.py")):
        for no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            code = line.split("#", 1)[0]
            if pattern.search(code):
                hits.append(f"{path.name}:{no}: {line.strip()}")
    assert hits == [], hits


def test_import_does_not_open_network_connections():
    probe = (
        "import socket\n"
        "def deny(*a, **k):\n    raise RuntimeError('network access at import time')\n"
        "socket.socket.connect = deny\nsocket.create_connection = deny\n"
        "import openbench, openbench.paired, openbench.preferences, openbench.quality, openbench.stats\n"
        "print('ok')\n"
    )
    res = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, cwd=ROOT,
                         env={"PYTHONPATH": str(ROOT / "src"), "PATH": ""}, timeout=60)
    assert res.returncode == 0 and res.stdout.strip() == "ok", res.stderr


def test_malformed_untrusted_inputs_fail_cleanly(tmp_path, capsys):
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{not json")
    wrong_types = tmp_path / "types.json"
    wrong_types.write_text(json.dumps({"project_version": "x", "evaluation_timestamp": "x", "dataset": {}, "model": {"model_name": "m", "model_version": "1"}, "aggregate_results": {}, "individual_results": 5}))
    not_object = tmp_path / "list.json"
    not_object.write_text("[1, 2, 3]")
    for path in (bad_json, wrong_types, not_object):
        assert main(["analyze", "--results", str(path), str(path)]) == 1
    err = capsys.readouterr().err
    assert "Traceback" not in err and "Unexpected Error" not in err


def test_blind_tasks_never_contain_key_material(tmp_path):
    exp = tmp_path / "exp"
    for model in ("secret-model-one", "secret-model-two"):
        assert main(["run", "--dataset", str(DATASET), "--provider", "mock", "--model", model, "--no-cache",
                     "--experiments-dir", str(exp)]) == 0
    runs = sorted(p.name for p in exp.iterdir() if p.is_dir())
    assert main(["preferences", "create", "--runs", *runs, "--experiments-dir", str(exp), "--output", str(tmp_path / "t"),
                 "--key-output", str(tmp_path / "private" / "key.json"), "--csv"]) == 0
    assert sorted(p.name for p in (tmp_path / "t").iterdir()) == ["tasks.csv", "tasks.jsonl"]
    blind = (tmp_path / "t" / "tasks.jsonl").read_text() + (tmp_path / "t" / "tasks.csv").read_text()
    key = json.loads((tmp_path / "private" / "key.json").read_text())
    for s in ("secret-model-one", "secret-model-two", *runs, "display_order", "record_id"):
        assert s not in blind
    for t in key["tasks"]:
        assert t["record_id"] not in blind


def test_cache_entries_store_redacted_generation_config(tmp_path):
    cache = ResponseCache(tmp_path / "c")
    from openbench.evaluator import ResponseItem

    cache.put("p", "m", "1", "d", "1", "r1", "prompt", ResponseItem(id="r1", response="x"),
              generation_config={"api_key": SECRETS[0]}, repetition_context=("grp_abc", 0))
    assert SECRETS[0] not in (tmp_path / "c" / "cache_entries.jsonl").read_text()


def test_every_module_imports_standalone_without_cycles():
    modules = sorted(
        "openbench." + ".".join(p.relative_to(ROOT / "src" / "openbench").with_suffix("").parts)
        for p in (ROOT / "src" / "openbench").rglob("*.py")
        if p.name != "__init__.py"
    )
    probe = "import importlib, sys\nfor m in sys.argv[1:]:\n    importlib.import_module(m)\nprint('ok')\n"
    assert len(modules) > 30
    for module in modules:
        res = subprocess.run([sys.executable, "-c", probe, module], capture_output=True, text=True,
                             env={"PYTHONPATH": str(ROOT / "src"), "PATH": ""}, timeout=60)
        assert res.returncode == 0, (module, res.stderr[-500:])
