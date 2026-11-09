"""Comparison compatibility policy and exploratory mode (plan C3, section 5)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from _v040_helpers import correct_answers, evaluate, write_dataset
from openbench import (
    DatasetLoader,
    EvaluationConfig,
    ExecutionConfig,
    ExperimentManager,
    MockModelProvider,
)
from openbench.cli import main
from openbench.exceptions import ConfigurationError, IncompatibleComparisonError
from openbench.paired import ComparisonArm, ComparisonPolicy, PairedComparisonEngine

FAST = {"resamples": 300, "permutations": 999}


def _pair(tmp_path, edit=None, name_b="m-b"):
    ds = write_dataset(tmp_path / "ds.jsonl", 8)
    a = evaluate(ds, correct_answers(8), "m-a", tmp_path / "a.json")
    b = evaluate(ds, correct_answers(8, wrong=(2,)), name_b, tmp_path / "b.json")
    if edit:
        payload = json.loads(b.read_text())
        edit(payload)
        b.write_text(json.dumps(payload))
    return a, b


def _run(paths, **policy):
    arms = [ComparisonArm.from_evaluation_file(p) for p in paths]
    return PairedComparisonEngine().compare(arms, ["exact_match", "keyword_coverage"], ComparisonPolicy(**{**FAST, **policy}))


def _codes(result, level):
    return {e["code"] for e in result.to_dict()["pairs"][0]["compatibility"][level]}


def _hard_codes(paths):
    with pytest.raises(IncompatibleComparisonError) as info:
        _run(paths)
    return {e["code"] for rep in info.value.report for e in rep.hard}


def test_compatible_pair_is_inferential(tmp_path):
    out = _run(_pair(tmp_path))
    assert out.analysis_mode == "inferential"
    assert _codes(out, "hard_incompatibilities") == set()


def test_hard_dataset_checksum_differs_without_record_hashes(tmp_path):
    ds = write_dataset(tmp_path / "ds.jsonl", 8)
    ds2 = write_dataset(tmp_path / "ds2.jsonl", 8, tweak={"item_003": "different"})
    a = evaluate(ds, correct_answers(8), "m-a", tmp_path / "a.json")
    b = evaluate(ds2, correct_answers(8), "m-b", tmp_path / "b.json")
    assert "dataset_checksum_differs" in _hard_codes([a, b])


def test_hard_record_content_differs_with_run_lineage(tmp_path):
    ds = DatasetLoader.load(str(write_dataset(tmp_path / "ds.jsonl", 8)))
    ds2 = DatasetLoader.load(str(write_dataset(tmp_path / "ds2.jsonl", 8, tweak={"item_003": "different"})))
    m = ExperimentManager(tmp_path / "exp")
    cfg = EvaluationConfig(metric_names=["exact_match", "keyword_coverage"])
    r1, _, _ = m.execute_and_record(ds, MockModelProvider(model_name="a"), evaluation_config=cfg, execution_config=ExecutionConfig(use_cache=False))
    r2, _, _ = m.execute_and_record(ds2, MockModelProvider(model_name="b"), evaluation_config=cfg, execution_config=ExecutionConfig(use_cache=False))
    arms = [ComparisonArm.from_experiments(m, r1.run_id), ComparisonArm.from_experiments(m, r2.run_id)]
    with pytest.raises(IncompatibleComparisonError) as info:
        PairedComparisonEngine().compare(arms, ["exact_match"], ComparisonPolicy(**FAST))
    hard = info.value.report[0].hard
    assert hard[0]["code"] == "record_content_differs"
    assert hard[0]["details"]["record_ids"] == ["item_003"]


def test_hard_metric_version_differs(tmp_path):
    def edit(p):
        p["aggregate_results"]["exact_match"]["metric_version"] = "9.9.9"
        for m in p["metrics_used"]:
            if m["name"] == "exact_match":
                m["version"] = "9.9.9"
    assert "metric_version_differs" in _hard_codes(_pair(tmp_path, edit))


def test_hard_metric_options_differ(tmp_path):
    def edit(p):
        p["configuration"]["metric_options"] = {"keyword_coverage": {"threshold": 0.5}}
    assert "metric_options_differ" in _hard_codes(_pair(tmp_path, edit))


def test_hard_normalization_differs(tmp_path):
    def edit(p):
        p["configuration"]["normalization"]["case_sensitive"] = True
    assert "normalization_differs" in _hard_codes(_pair(tmp_path, edit))


def test_hard_metric_missing_in_arm(tmp_path):
    def edit(p):
        del p["aggregate_results"]["keyword_coverage"]
    assert "metric_not_in_arm" in _hard_codes(_pair(tmp_path, edit))


def test_warning_dataset_version_label_differs_content_verified(tmp_path):
    def edit(p):
        p["dataset"]["version"] = "0.9.0"
    out = _run(_pair(tmp_path, edit))
    assert "dataset_version_label_differs" in _codes(out, "warnings")


def test_warning_record_id_sets_differ_with_run_lineage(tmp_path):
    ds = DatasetLoader.load(str(write_dataset(tmp_path / "ds.jsonl", 8)))
    m = ExperimentManager(tmp_path / "exp")
    full = EvaluationConfig(metric_names=["exact_match"])
    subset = EvaluationConfig(metric_names=["exact_match"], selected_ids=["item_001", "item_002", "item_003"])
    r1, _, _ = m.execute_and_record(ds, MockModelProvider(model_name="a"), evaluation_config=full, execution_config=ExecutionConfig(use_cache=False))
    r2, _, _ = m.execute_and_record(ds, MockModelProvider(model_name="b"), evaluation_config=subset, execution_config=ExecutionConfig(use_cache=False))
    arms = [ComparisonArm.from_experiments(m, r1.run_id), ComparisonArm.from_experiments(m, r2.run_id)]
    out = PairedComparisonEngine().compare(arms, ["exact_match"], ComparisonPolicy(**FAST))
    assert "record_id_sets_differ" in _codes(out, "warnings")
    comp = out.comparisons[0]
    assert comp["sample_count"] == 3
    assert comp["excluded_breakdown"]["not_in_all_arms"] == 5


def test_warning_generation_configuration_lists_every_key(tmp_path):
    def edit(p):
        p["model"]["inference_config"].update({"temperature": 0.7, "max_tokens": 99})
    out = _run(_pair(tmp_path, edit))
    entry = next(e for e in out.to_dict()["pairs"][0]["compatibility"]["warnings"] if e["code"] == "generation_configuration_differs")
    assert set(entry["details"]["differences"]) == {"temperature", "max_tokens"}


def test_generation_config_secrets_are_redacted_in_report(tmp_path):
    def edit(p):
        p["model"]["inference_config"]["extra_options"] = {"api_key": "sk-abcdefghijklmnop"}
    out = _run(_pair(tmp_path, edit))
    assert "sk-abcdefghijklmnop" not in json.dumps(out.to_dict())


def test_warning_mock_and_info_provider_differs(tmp_path):
    def edit(p):
        p["model"]["model_provider"] = "mock"
        p["model"]["inference_config"]["is_mock"] = True
    out = _run(_pair(tmp_path, edit))
    assert "mock_provider" in _codes(out, "warnings")
    assert "provider_differs" in _codes(out, "info")


def test_exploratory_requires_reason():
    with pytest.raises(ConfigurationError):
        ComparisonPolicy(exploratory=True)
    with pytest.raises(ConfigurationError):
        ComparisonPolicy(exploratory=True, exploratory_reason="   ")
    with pytest.raises(ConfigurationError):
        ComparisonPolicy(exploratory_reason="why")


def test_exploratory_mode_nulls_every_inferential_field(tmp_path):
    def edit(p):
        p["configuration"]["normalization"]["case_sensitive"] = True
    paths = _pair(tmp_path, edit)
    out = _run(paths, exploratory=True, exploratory_reason="checking a normalisation change", equivalence_margin=0.1, min_practical_difference=0.1).to_dict()
    assert out["analysis_mode"] == "exploratory"
    assert out["banner"].startswith("EXPLORATORY")
    assert out["exploratory_reason"] == "checking a normalisation change"
    for comp in out["comparisons"]:
        for key in ("confidence_interval", "p_value", "adjusted_p_value", "equivalence", "practical_classification"):
            assert comp[key] is None
            assert comp["unavailable"][key] == "exploratory_incompatible_inputs"
        assert comp["point_estimate"] is not None
        assert "wins_losses_ties" in comp
    assert out["pairs"][0]["compatibility"]["hard_incompatibilities"]


def test_cli_hard_incompatibility_exits_1_and_lists_all(tmp_path, capsys):
    def edit(p):
        p["configuration"]["normalization"]["case_sensitive"] = True
        p["aggregate_results"]["exact_match"]["metric_version"] = "9.9.9"
        for m in p["metrics_used"]:
            if m["name"] == "exact_match":
                m["version"] = "9.9.9"
    a, b = _pair(tmp_path, edit)
    code = main(["analyze", "--results", str(a), str(b), "--resamples", "200", "--output", str(tmp_path / "o.json")])
    err = capsys.readouterr().err
    assert code == 1
    assert "normalization_differs" in err and "metric_version_differs" in err
    assert not (tmp_path / "o.json").exists()
    code = main(["analyze", "--results", str(a), str(b), "--resamples", "200", "--exploratory",
                 "--exploratory-reason", "look only", "--output", str(tmp_path / "o.json")])
    assert code == 0
    assert json.loads((tmp_path / "o.json").read_text())["analysis_mode"] == "exploratory"
    assert main(["analyze", "--results", str(a), str(b), "--exploratory"]) == 2
    assert main(["analyze", "--results", str(a), str(b), "--exploratory-reason", "x"]) == 2


def test_cli_analyze_runs_group_variability_and_comparison(tmp_path, capsys):
    exp = tmp_path / "exp"
    ds = Path(__file__).resolve().parents[1] / "datasets" / "reasoning" / "sample.jsonl"
    assert main(["run", "--dataset", str(ds), "--provider", "mock", "--model", "m1", "--repetitions", "2",
                 "--no-cache", "--experiments-dir", str(exp)]) == 0
    out = capsys.readouterr().out
    group_id = out.split("Repeat group '")[1].split("'")[0]
    assert main(["run", "--dataset", str(ds), "--provider", "mock", "--model", "m2", "--no-cache",
                 "--experiments-dir", str(exp)]) == 0
    run_id = capsys.readouterr().out.split("Run '")[1].split("'")[0]
    assert main(["analyze", "--runs", group_id, "--experiments-dir", str(exp), "--resamples", "200",
                 "--output", str(tmp_path / "var.json")]) == 0
    var = json.loads((tmp_path / "var.json").read_text())
    assert var["analysis"] == "repeated_run_variability" and var["repetitions"] == 2
    assert main(["analyze", "--runs", group_id, run_id, "--experiments-dir", str(exp), "--resamples", "200",
                 "--permutations", "999", "--output", str(tmp_path / "cmp.json")]) == 0
    cmp_out = json.loads((tmp_path / "cmp.json").read_text())
    assert cmp_out["analysis"] == "paired_comparison"
    assert group_id in cmp_out["variability"]
    assert main(["analyze", "--runs", run_id, "--experiments-dir", str(exp)]) == 2
    assert main(["analyze", "--runs", "run_missing00", run_id, "--experiments-dir", str(exp)]) == 1
