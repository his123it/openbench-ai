"""Paired comparison engine: stable-ID alignment, hand-checked statistics, missing data,
breakdowns, failure rates, determinism and no provider calls (plan C1, C2, C4)."""

from __future__ import annotations

import json

import pytest

from _v040_helpers import correct_answers, evaluate, strip_volatile, write_dataset
from openbench import EvaluationConfig, ExecutionConfig, ExperimentManager, MockModelProvider
from openbench.paired import ComparisonArm, ComparisonPolicy, PairedComparisonEngine
from openbench.stats import adjust_p_values, mcnemar_exact

FAST = {"resamples": 400, "permutations": 999}


@pytest.fixture
def files(tmp_path):
    ds = write_dataset(tmp_path / "ds.jsonl", 8)
    a = evaluate(ds, correct_answers(8, wrong=(2, 5)), "model-a", tmp_path / "a_base.json")
    b = evaluate(ds, correct_answers(8, wrong=(3,)), "model-b", tmp_path / "b_chal.json")
    a_s = evaluate(ds, correct_answers(8, wrong=(2, 5)), "model-a", tmp_path / "s" / "a_base.json", shuffle=True)
    b_s = evaluate(ds, correct_answers(8, wrong=(3,)), "model-b", tmp_path / "s" / "b_chal.json", shuffle=True)
    return ds, a, b, a_s, b_s


def _compare(paths, metrics=("exact_match", "keyword_coverage"), **policy):
    arms = [ComparisonArm.from_evaluation_file(p) for p in paths]
    return PairedComparisonEngine().compare(arms, list(metrics), ComparisonPolicy(**{**FAST, **policy})).to_dict()


def _comp(out, metric):
    return next(c for c in out["comparisons"] if c["metric"]["name"] == metric)


def test_hand_fixture_binary(files):
    _, a, b, _, _ = files
    out = _compare([a, b])
    em = _comp(out, "exact_match")
    assert (em["baseline"], em["challenger"]) == ("a_base", "b_chal")
    assert em["sample_count"] == 8 and em["excluded_count"] == 0
    assert em["point_estimate"] == pytest.approx(0.125)
    assert em["arm_means"] == {"a_base": 0.75, "b_chal": 0.875}
    assert em["test"]["test"] == "mcnemar_exact"
    assert em["effect_size"]["binary"]["b_challenger_only_correct"] == 2
    assert em["effect_size"]["binary"]["c_baseline_only_correct"] == 1
    assert em["p_value"] == mcnemar_exact(2, 1).p_value == 1.0
    wlt = em["wins_losses_ties"]
    assert wlt["challenger_better_ids"] == ["item_002", "item_005"]
    assert wlt["baseline_better_ids"] == ["item_003"]
    assert wlt["ties"] == 5
    assert [d["id"] for d in em["disagreements"]] == ["item_002", "item_003", "item_005"]
    assert "No statistically significant difference" in em["interpretation"]
    assert "equivalent" not in em["interpretation"].lower()
    assert out["provider_calls"] == 0
    json.dumps(out, allow_nan=False)


def test_holm_family_over_pairs_and_metrics(files):
    _, a, b, _, _ = files
    out = _compare([a, b])
    raw = [c["p_value"] for c in out["comparisons"]]
    expected = adjust_p_values(raw, "holm").adjusted
    assert [c["adjusted_p_value"] for c in out["comparisons"]] == list(expected)
    assert out["families"]["difference"]["size"] == 2
    out_bh = _compare([a, b], correction="bh")
    assert out_bh["families"]["difference"]["warnings"][0]["code"] == "controls_fdr_not_fwer"


def test_reordered_files_and_arm_order_give_identical_results(files):
    _, a, b, a_s, b_s = files
    base = strip_volatile(_compare([a, b])["comparisons"])
    shuffled = strip_volatile(_compare([a_s, b_s])["comparisons"])
    swapped = strip_volatile(_compare([b, a])["comparisons"])
    assert base == shuffled == swapped


def test_deterministic_given_seed_and_seed_matters(files):
    _, a, b, _, _ = files
    one = strip_volatile(_compare([a, b], analysis_seed=3))
    two = strip_volatile(_compare([a, b], analysis_seed=3))
    assert one == two
    other = strip_volatile(_compare([a, b], analysis_seed=4))
    assert one["comparisons"][1]["confidence_interval"] != other["comparisons"][1]["confidence_interval"]


def test_missing_and_invalid_responses_complete_case_vs_failure(tmp_path):
    ds = write_dataset(tmp_path / "ds.jsonl", 8)
    a = evaluate(ds, correct_answers(8), "model-a", tmp_path / "a.json")
    c = evaluate(ds, correct_answers(8, missing=(4,), empty=(6,)), "model-c", tmp_path / "c.json")
    cc = _comp(_compare([a, c]), "exact_match")
    assert cc["sample_count"] == 6 and cc["excluded_count"] == 2
    assert cc["excluded_breakdown"]["missing_in"]["c"] == 1
    assert cc["excluded_breakdown"]["invalid_in"]["c"] == 1
    assert cc["point_estimate"] == 0.0  # missing items are never counted as wrong here
    mf = _comp(_compare([a, c], missing_policy="missing_as_failure"), "exact_match")
    assert mf["sample_count"] == 8
    assert mf["point_estimate"] == pytest.approx(-2 / 8)
    cov = cc["coverage"]["c"]
    assert (cov["with_valid_value"], cov["missing"], cov["invalid"]) == (6, 1, 1)
    out = _compare([a, c])
    fr = out["failure_rate_comparisons"][0]
    assert fr["failure_rate"] == {"a": 0.0, "c": 0.25}
    assert fr["items_failed_only_in_challenger"] == ["item_004", "item_006"]
    assert fr["test"]["test"] == "mcnemar_exact"


def test_breakdowns_descriptive_unless_stratum_tests(files):
    _, a, b, _, _ = files
    em = _comp(_compare([a, b]), "exact_match")
    cats = em["breakdowns"]["by_category"]
    assert set(cats) == {"alpha", "beta"}
    assert sum(v["sample_count"] for v in cats.values()) == 8
    assert all(v["descriptive_only"] and v["p_value"] is None for v in cats.values())
    assert set(em["breakdowns"]["by_difficulty"]) == {"easy", "medium", "hard"}
    out = _compare([a, b], stratum_tests=True)
    em2 = _comp(out, "exact_match")
    assert all(v["p_value"] is not None for v in em2["breakdowns"]["by_category"].values())
    assert out["families"]["difference"]["size"] > 2


def test_ids_file_restricts_and_counts_filtered_out(files):
    _, a, b, _, _ = files
    em = _comp(_compare([a, b], selected_ids=("item_001", "item_002", "item_099")), "exact_match")
    assert em["sample_count"] == 2
    assert em["excluded_breakdown"]["filtered_out"] == 6
    assert em["excluded_breakdown"]["requested_ids_not_found"] == 1


def test_continuous_metric_uses_sign_flip_and_unknown_gets_ci_only(files, tmp_path):
    _, a, b, _, _ = files
    kc = _comp(_compare([a, b]), "keyword_coverage")
    assert kc["test"]["test"] == "sign_flip"
    assert kc["test"]["validity"] == {
        "sharp_exchangeability_null": "exact", "mean_difference_null": "approximate"
    }


def test_equivalence_family_and_practical_classification(files):
    _, a, b, _, _ = files
    out = _compare([a, b], equivalence_margin=0.2, min_practical_difference=0.05)
    em = _comp(out, "exact_match")
    eq = em["equivalence"]
    assert eq["procedure"] == "exact_unconditional_score"
    assert eq["familywise_result"] in {"established", "not_established"}
    assert eq["interval"]["multiplicity_adjusted"] is False
    assert "not adjusted for the" in eq["interval"]["label"]
    kc = _comp(out, "keyword_coverage")
    assert kc["equivalence"]["result_unadjusted"] == "not_established"
    assert kc["equivalence"]["reason"] == "method_not_validated"
    assert em["practical_classification"] in {
        "exceeds_practical_threshold", "below_practical_threshold", "inconclusive_relative_to_threshold"
    }
    assert out["families"]["equivalence"]["correction"] == "holm"


def test_no_provider_is_constructed_or_called(tmp_path, monkeypatch):
    ds = write_dataset(tmp_path / "ds.jsonl", 8)
    manager = ExperimentManager(tmp_path / "exp")
    from openbench import DatasetLoader

    data = DatasetLoader.load(str(ds))
    cfg = EvaluationConfig(metric_names=["exact_match"])
    r1, _, _ = manager.execute_and_record(data, MockModelProvider(model_name="x"), evaluation_config=cfg, execution_config=ExecutionConfig(use_cache=False))
    r2, _, _ = manager.execute_and_record(data, MockModelProvider(model_name="y", response_map={"item_001": "no"}), evaluation_config=cfg, execution_config=ExecutionConfig(use_cache=False))

    def boom(*args, **kwargs):
        raise AssertionError("a provider must not be constructed during analysis")

    monkeypatch.setattr(MockModelProvider, "__init__", boom)
    monkeypatch.setattr(MockModelProvider, "generate_one", boom)
    arms = [ComparisonArm.from_experiments(manager, r1.run_id), ComparisonArm.from_experiments(manager, r2.run_id)]
    out = PairedComparisonEngine().compare(arms, ["exact_match"], ComparisonPolicy(**FAST)).to_dict()
    assert out["comparisons"][0]["sample_count"] == 8


def test_three_arms_baseline_and_all_pairs(tmp_path):
    ds = write_dataset(tmp_path / "ds.jsonl", 8)
    paths = [
        evaluate(ds, correct_answers(8, wrong=w), name, tmp_path / f"{name}.json")
        for name, w in (("m1", ()), ("m2", (1,)), ("m3", (1, 2)))
    ]
    all_pairs = _compare(paths, metrics=("exact_match",))
    assert len(all_pairs["comparisons"]) == 3
    vs_base = _compare(paths, metrics=("exact_match",), baseline="m2")
    assert {(c["baseline"], c["challenger"]) for c in vs_base["comparisons"]} == {("m2", "m1"), ("m2", "m3")}
    with pytest.raises(Exception, match="Baseline"):
        _compare(paths, metrics=("exact_match",), baseline="nope")


def test_repeat_group_arm_uses_item_means(tmp_path):
    from openbench import DatasetLoader

    ds = DatasetLoader.load(str(write_dataset(tmp_path / "ds.jsonl", 8)))
    manager = ExperimentManager(tmp_path / "exp")
    cfg = EvaluationConfig(metric_names=["exact_match"])
    group = manager.execute_repeated(ds, MockModelProvider(model_name="g", stochastic_error_rate=0.5, stochastic_seed=2), 4,
                                     evaluation_config=cfg, execution_config=ExecutionConfig(use_cache=False))
    single, _, _ = manager.execute_and_record(ds, MockModelProvider(model_name="s"), evaluation_config=cfg,
                                              execution_config=ExecutionConfig(use_cache=False))
    arms = [ComparisonArm.from_experiments(manager, group.repeat_group_id), ComparisonArm.from_experiments(manager, single.run_id)]
    out = PairedComparisonEngine().compare(arms, ["exact_match"], ComparisonPolicy(**FAST)).to_dict()
    comp = out["comparisons"][0]
    assert comp["test"]["test"] == "sign_flip"  # repeated binary -> per-item means
    assert comp["sample_count"] == 8  # unit is the item, never item x repetition
    assert {r["baseline_repetitions_used"] for r in comp["per_item"]} == {4}
    assert any(c["code"] == "repetition_counts_differ" for c in out["pairs"][0]["compatibility"]["warnings"])
