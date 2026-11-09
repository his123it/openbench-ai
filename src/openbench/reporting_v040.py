"""Renderers for v0.4.0 reports: paired comparison, repeated-run variability, human
preference analysis and quality audits, in JSON, Markdown and CSV.

* JSON is the source of truth (strict: no NaN or infinity).
* Markdown is rendered from the same payload, so numbers agree with the JSON (formatted to
  4 decimal places; p-values to 4 significant digits).
* CSV cells go through :func:`openbench.security.csv_safe` (formula-injection protection).

Wording rules: a non-significant difference is never described as equivalence; intervals
are labelled as not multiplicity-adjusted; exploratory reports carry a banner; unavailable
statistics are shown as ``n/a`` with their reason.
"""

from __future__ import annotations

import csv
import io
import json
from typing import Any

from openbench.exceptions import ConfigurationError
from openbench.reporting import escape_markdown_cell
from openbench.security import csv_safe

MD_DISPLAY_LIMIT = 50


def to_json(payload: dict[str, Any]) -> str:
    """Serialise strictly (no NaN or infinity)."""
    return json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n"


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def fmt_p(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.4g}"


def fmt_ci(ci: dict[str, Any] | None) -> str:
    if not ci or ci.get("lower") is None or ci.get("upper") is None:
        reason = (ci or {}).get("unavailable_reason")
        return f"n/a ({reason})" if reason else "n/a"
    return f"[{fmt(ci['lower'])}, {fmt(ci['upper'])}]"


def _cell(value: Any) -> str:
    return escape_markdown_cell(value)


def _table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    out = ["| " + " | ".join(_cell(h) for h in headers) + " |", "|" + "---|" * len(headers)]
    for row in rows:
        out.append("| " + " | ".join(_cell(c) for c in row) + " |")
    return out


def _csv(headers: list[str], rows: list[list[Any]]) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(headers)
    for row in rows:
        writer.writerow(["" if c is None else csv_safe(c) for c in row])
    return buf.getvalue()


def _provenance_md(prov: dict[str, Any] | None) -> list[str]:
    if not prov:
        return []
    lines = ["## Provenance", ""]
    lines += _table(
        ["Field", "Value"],
        [
            ["OpenBench version", prov.get("openbench_version")],
            ["Python", prov.get("python_version")],
            ["Platform", prov.get("platform")],
            ["Generated at", prov.get("generated_at")],
            ["Analysis seed", prov.get("analysis_seed")],
            ["Parameters SHA-256", prov.get("parameters_sha256")],
            ["Provider calls", prov.get("provider_calls")],
        ],
    )
    files = [(i.get("arm_id", ""), f["path"], f["sha256"]) for i in prov.get("inputs", []) for f in i.get("files", [])]
    if files:
        lines += ["", "Input files:", ""]
        lines += _table(["Arm", "File", "SHA-256"], [list(f) for f in files])
    lines += ["", f"_{prov.get('integrity_disclaimer', '')}_", ""]
    return lines


# ---------------------------------------------------------------------------
# Analysis (paired comparison and variability)
# ---------------------------------------------------------------------------


def render_analysis(payload: dict[str, Any], fmt_name: str) -> str:
    if fmt_name == "json":
        return to_json(payload)
    kind = payload.get("analysis")
    if fmt_name == "markdown":
        if kind == "paired_comparison":
            return paired_markdown(payload)
        if kind == "repeated_run_variability":
            return variability_markdown(payload)
    if fmt_name == "csv":
        if kind == "paired_comparison":
            return paired_csv(payload)
        if kind == "repeated_run_variability":
            return variability_csv(payload)
    raise ConfigurationError(f"Unsupported analysis format '{fmt_name}' for '{kind}'")


PAIRED_CSV_HEADERS = [
    "comparison_id", "metric", "metric_version", "score_kind", "baseline", "challenger",
    "sample_count", "excluded_count", "baseline_mean", "challenger_mean", "point_estimate",
    "ci_lower", "ci_upper", "confidence_level", "test", "p_value", "adjusted_p_value",
    "correction", "family_size", "challenger_better", "baseline_better", "ties",
    "practical_classification", "equivalence_margin", "p_tost", "adjusted_p_tost",
    "equivalence_unadjusted", "equivalence_familywise", "equivalence_reason", "analysis_mode",
    "missing_policy", "analysis_seed",
]


def paired_rows(payload: dict[str, Any]) -> list[list[Any]]:
    rows = []
    params = payload.get("parameters", {})
    for c in payload["comparisons"]:
        ci = c.get("confidence_interval") or {}
        eq = c.get("equivalence") or {}
        wlt = c["wins_losses_ties"]
        rows.append([
            c["comparison_id"], c["metric"]["name"], c["metric"]["version"], c["metric"]["score_kind"],
            c["baseline"], c["challenger"], c["sample_count"], c["excluded_count"],
            c["arm_means"][c["baseline"]], c["arm_means"][c["challenger"]], c["point_estimate"],
            ci.get("lower"), ci.get("upper"), ci.get("confidence_level"),
            (c.get("test") or {}).get("test"), c.get("p_value"), c.get("adjusted_p_value"),
            (c.get("family") or {}).get("correction"), (c.get("family") or {}).get("size"),
            wlt["challenger_better"], wlt["baseline_better"], wlt["ties"],
            c.get("practical_classification"), eq.get("margin"), eq.get("p_tost"), eq.get("adjusted_p_tost"),
            eq.get("result_unadjusted"), eq.get("familywise_result"), eq.get("reason"),
            payload["analysis_mode"], params.get("missing_policy"), payload.get("analysis_seed"),
        ])
    return rows


def paired_csv(payload: dict[str, Any]) -> str:
    return _csv(PAIRED_CSV_HEADERS, paired_rows(payload))


def paired_markdown(p: dict[str, Any]) -> str:
    params = p["parameters"]
    level = f"{round(params['confidence_level'] * 100, 3):g}%"
    lines = ["# OpenBench AI paired comparison", ""]
    if p["analysis_mode"] == "exploratory":
        lines += [f"> **{p['banner']}**", ">", f"> Reason given: {escape_markdown_cell(p.get('exploratory_reason'))}", "> Only descriptive values are shown; every inferential field is unavailable.", ""]
    lines += [
        f"- Analysis ID: `{p['analysis_id']}` (schema {p['schema_version']}, mode **{p['analysis_mode']}**)",
        f"- Missing-data policy: `{params['missing_policy']}`; multiplicity correction: `{params['correction']}`; alpha = {params['alpha']}",
        f"- Confidence level: {level}; bootstrap resamples: {params['resamples']}; permutations: {params['permutations']}; analysis seed: {p['analysis_seed']}",
        f"- Practical threshold: {fmt(params.get('min_practical_difference'))}; pre-specified equivalence margin: {fmt(params.get('equivalence_margin'))}",
        "- Difference = challenger - baseline (higher is better for every metric).",
        "",
        "## Arms",
        "",
    ]
    lines += _table(
        ["Arm", "Model", "Version", "Provider", "Source", "Repetitions", "Mock"],
        [[a["arm_id"], a["model_name"], a["model_version"], a["model_provider"], a["source"], a["repetitions"], fmt(a["is_mock_provider"])] for a in p["arms"]],
    )
    lines += ["", "## Compatibility", ""]
    for pair in p["pairs"]:
        comp = pair["compatibility"]
        lines.append(f"**{pair['baseline']} vs {pair['challenger']}**: {'compatible for inference' if comp['compatible_for_inference'] else 'NOT compatible for inference'}")
        for level_name, key in (("Hard incompatibility", "hard_incompatibilities"), ("Warning", "warnings"), ("Info", "info")):
            for e in comp[key]:
                lines.append(f"- {level_name} `{e['code']}`: {escape_markdown_cell(e['message'])}")
        lines.append("")
    lines += ["## Results", ""]
    rows = []
    for c in p["comparisons"]:
        rows.append([
            c["metric"]["name"], f"{c['baseline']} -> {c['challenger']}", c["sample_count"], c["excluded_count"],
            fmt(c["arm_means"][c["baseline"]]), fmt(c["arm_means"][c["challenger"]]), fmt(c["point_estimate"]),
            fmt_ci(c.get("confidence_interval")), (c.get("test") or {}).get("test") or "n/a",
            fmt_p(c.get("p_value")), fmt_p(c.get("adjusted_p_value")),
        ])
    lines += _table(["Metric", "Pair", "n", "Excluded", "Baseline mean", "Challenger mean", "Difference", f"{level} CI (not multiplicity-adjusted)", "Test", "p", "Adjusted p"], rows)
    lines.append("")
    for c in p["comparisons"]:
        lines += [f"### {c['metric']['name']}: {c['baseline']} -> {c['challenger']}", ""]
        lines.append(f"- {escape_markdown_cell(c.get('interpretation', ''))}")
        test = c.get("test") or {}
        if test.get("description"):
            lines.append(f"- Test: {escape_markdown_cell(test['description'])}")
        if test.get("permutation_distribution"):
            lines.append(f"- Permutation distribution: {test['permutation_distribution']}; validity: {test.get('validity')}")
        eff = c.get("effect_size") or {}
        cont = eff.get("continuous") or {}
        if cont:
            lines.append(f"- Effect sizes: mean difference {fmt(cont.get('mean_difference'))}, d_z {fmt(cont.get('d_z'))} (descriptive), probability of superiority {fmt(cont.get('probability_of_superiority'))}")
        if eff.get("binary"):
            b = eff["binary"]
            lines.append(f"- Discordant pairs: challenger-only correct {b['b_challenger_only_correct']}, baseline-only correct {b['c_baseline_only_correct']}; risk difference {fmt(b.get('risk_difference'))}")
        wlt = c["wins_losses_ties"]
        lines.append(f"- Wins/losses/ties (challenger vs baseline): {wlt['challenger_better']} / {wlt['baseline_better']} / {wlt['ties']}")
        eb = c["excluded_breakdown"]
        lines.append(f"- Exclusions: missing {eb['missing_in']}, invalid {eb['invalid_in']}, metric unavailable {eb['metric_unavailable_in']}, not in all arms {eb['not_in_all_arms']}, filtered out {eb['filtered_out']}")
        cov = c["coverage"]
        lines.append("- Coverage: " + "; ".join(f"{arm} {fmt(v['coverage_rate'])} ({v['with_valid_value']}/{v['items']})" for arm, v in cov.items()))
        if c.get("practical_classification") is not None:
            lines.append(f"- Practical importance (descriptive, not equivalence): `{c['practical_classification']}`")
        eq = c.get("equivalence")
        if eq:
            interval = eq.get("interval") or {}
            lines.append(f"- Equivalence (TOST, margin {fmt(eq.get('margin'))}): familywise **{eq.get('familywise_result')}** (Holm-adjusted p_TOST {fmt_p(eq.get('adjusted_p_tost'))}); unadjusted {eq.get('result_unadjusted')} (p_TOST {fmt_p(eq.get('p_tost'))}){'; reason: ' + str(eq['reason']) if eq.get('reason') else ''}")
            if interval:
                lines.append(f"  - {escape_markdown_cell(interval.get('label', ''))}: [{fmt(interval.get('lower'))}, {fmt(interval.get('upper'))}]")
            if eq.get("decision_disagreement"):
                lines.append(f"  - {eq['decision_disagreement']}")
        for w in c.get("warnings", []):
            lines.append(f"- Warning `{w['code']}`: {escape_markdown_cell(w['message'])}")
        for key, reason in sorted(c.get("unavailable", {}).items()):
            lines.append(f"- Unavailable `{key}`: {reason}")
        for dim in ("by_category", "by_difficulty"):
            bd = c.get("breakdowns", {}).get(dim) or {}
            if bd:
                lines += ["", f"{dim.replace('_', ' ').capitalize()} ({'tests included in the family' if params.get('stratum_tests') else 'descriptive'}):", ""]
                lines += _table(
                    ["Group", "n", "Baseline mean", "Challenger mean", "Difference", "CI", "p", "Adjusted p"],
                    [[k, v["sample_count"], fmt(v["baseline_mean"]), fmt(v["challenger_mean"]), fmt(v["point_estimate"]), fmt_ci(v.get("confidence_interval")), fmt_p(v.get("p_value")), fmt_p(v.get("adjusted_p_value"))] for k, v in bd.items()],
                )
        dis = c.get("disagreements", [])
        if dis:
            lines += ["", f"Largest disagreements ({min(len(dis), MD_DISPLAY_LIMIT)} of {len(dis)} shown):", ""]
            lines += _table(["Record", "Baseline", "Challenger", "Difference"], [[d["id"], fmt(d["baseline"]), fmt(d["challenger"]), fmt(d["difference"])] for d in dis[:MD_DISPLAY_LIMIT]])
        lines.append("")
    lines += ["## Failure-rate comparisons", ""]
    lines += _table(
        ["Pair", "n", "Baseline failure rate", "Challenger failure rate", "Difference", "CI", "p", "Adjusted p (Holm)"],
        [[f"{f['baseline']} -> {f['challenger']}", f["sample_count"], fmt(f["failure_rate"][f["baseline"]]), fmt(f["failure_rate"][f["challenger"]]), fmt(f["difference"]), fmt_ci(f.get("confidence_interval")), fmt_p(f.get("p_value")), fmt_p(f.get("adjusted_p_value"))] for f in p["failure_rate_comparisons"]],
    )
    fam = p.get("families") or {}
    if fam.get("difference"):
        d = fam["difference"]
        lines += ["", f"Difference-test family: {d['size']} member(s), correction `{d['correction']}`."]
        for w in d.get("warnings", []):
            lines.append(f"- Warning `{w['code']}`: {w['message']}")
    if fam.get("equivalence"):
        lines.append(f"Equivalence family: {fam['equivalence']['size']} member(s), correction `holm` (separate from the difference family).")
    variability = p.get("variability") or {}
    for arm_id, var in variability.items():
        lines += ["", f"## Repeated-run variability: {arm_id}", ""]
        lines += variability_markdown(var, heading=False).splitlines()
    lines += ["", "## Limitations", ""] + [f"- {item}" for item in p.get("limitations", [])]
    lines.append("")
    lines += _provenance_md(p.get("provenance"))
    return "\n".join(lines).rstrip() + "\n"


def variability_rows(v: dict[str, Any]) -> list[list[Any]]:
    rows = []
    for metric, m in v["metrics"].items():
        for e in m["per_item"]:
            rows.append([metric, e["id"], e["category"], e["difficulty"], e["repetitions_used"], e["mean"], e["sd"], e["min"], e["max"], e["changed"], " ".join("" if x is None else f"{x:g}" for x in e["repetition_values"]), " ".join(e["repetition_statuses"])])
    return rows


def variability_csv(v: dict[str, Any]) -> str:
    return _csv(["metric", "record_id", "category", "difficulty", "repetitions_used", "mean", "sd", "min", "max", "changed", "repetition_values", "repetition_statuses"], variability_rows(v))


def variability_markdown(v: dict[str, Any], heading: bool = True) -> str:
    lines = []
    if heading:
        lines += ["# OpenBench AI repeated-run variability", ""]
    lines += [f"- Repetitions: {v['repetitions']}; analysis seed: {v['analysis_seed']}; unit of analysis: {v['parameters']['unit_of_analysis']}", ""]
    rows = []
    for metric, m in v["metrics"].items():
        gen, item = m["generation_variability"], m["item_variation"]
        rd = m["run_distribution"]["headline_summary"]
        rows.append([
            metric, m["items_with_valid_values"], fmt(item["mean_of_item_means"]), fmt_ci(item["question_level_confidence_interval"]),
            fmt(item["item_spread_sd"]), fmt(gen["within_item_variance_mean"]), fmt(gen["items_with_any_change_rate"]),
            fmt(gen.get("mean_pairwise_agreement")), fmt(rd["mean"]), fmt(rd["sd"]), f"{fmt(rd['min'])} to {fmt(rd['max'])}",
        ])
    lines += _table(["Metric", "Items", "Mean of item means", "Question-level CI", "Item spread (SD)", "Within-item variance", "Items changing", "Pairwise agreement", "Run mean", "Run SD", "Run range"], rows)
    ri = v["response_identity"]
    lines += ["", f"- Exact response identity across repetitions: {fmt(ri['exact_response_identity_rate'])} ({ri['items_with_identical_responses']}/{ri['items_with_response_in_every_repetition']} items)"]
    fail = v["failures"]
    lines.append("- Failures per repetition (missing + invalid): " + ", ".join(f"#{r['repetition_index']}: {r['missing']} + {r['invalid']}" for r in fail["per_repetition"]))
    if fail["items_failing_in_some_repetitions"]:
        lines.append(f"- Items failing in some repetitions: {', '.join(fail['items_failing_in_some_repetitions'][:MD_DISPLAY_LIMIT])}")
    lat = v["latency_and_tokens"]["latency_ms"]
    if lat["available"]:
        lines.append(f"- Latency (ms), mean per repetition: {', '.join(fmt(x, 2) for x in lat['per_repetition_mean'])}")
    lines += ["", f"_{v['interpretation']}_", ""]
    lines += [f"- {item}" for item in v.get("limitations", [])]
    if heading:
        lines += [""] + _provenance_md(v.get("provenance"))
    return "\n".join(lines).rstrip() + "\n"


# ---------------------------------------------------------------------------
# Preferences
# ---------------------------------------------------------------------------


def _pref_rows(p: dict[str, Any]) -> list[list[Any]]:
    rows = []
    for pair in p["pairs"]:
        a, b = pair["model_a"]["source_id"], pair["model_b"]["source_id"]
        groups = [("overall", "all", pair["overall"])]
        groups += [("category", k, v) for k, v in pair["by_category"].items()]
        groups += [("difficulty", k, v) for k, v in pair["by_difficulty"].items()]
        for dim, label, s in groups:
            ci = s["win_rate_a_confidence_interval"]
            oc = s["outcome_counts"]
            rows.append([a, b, dim, label, s["tasks_in_win_rate"], s["annotations"], s["win_rate_a_task_weighted"], ci.get("lower"), ci.get("upper"), s["win_rate_a_annotation_weighted"], s["tie_rate"], s["neither_rate"], oc["A"], oc["B"], oc["tie"], oc["neither"]])
    return rows


PREF_HEADERS = ["model_a", "model_b", "breakdown", "group", "tasks_in_win_rate", "annotations", "win_rate_a_task_weighted", "ci_lower", "ci_upper", "win_rate_a_annotation_weighted", "tie_rate", "neither_rate", "a_wins", "b_wins", "ties", "neither"]


def render_preferences(payload: dict[str, Any], fmt_name: str) -> str:
    if fmt_name == "json":
        return to_json(payload)
    if fmt_name == "csv":
        return _csv(PREF_HEADERS, _pref_rows(payload))
    if fmt_name != "markdown":
        raise ConfigurationError(f"Unsupported preference report format '{fmt_name}'")
    p = payload
    lines = ["# OpenBench AI human preference analysis", "", "> Unblinded analysis: it joins annotations with the identity key. Do not share it with annotators who are still annotating.", ""]
    lines += [f"- Annotations: {p['annotation_count']}; annotators: {p['annotator_count']}; tasks: {p['task_count']}; unannotated tasks: {p['unannotated_tasks']}", f"- Outcome distribution (display space): {p['outcome_distribution_display_space']}", ""]
    rows = [[r[0], r[1], r[2], r[3], r[4], r[5], fmt(r[6]), f"[{fmt(r[7])}, {fmt(r[8])}]", fmt(r[9]), fmt(r[10]), fmt(r[11])] for r in _pref_rows(p)]
    lines += _table(["Model A", "Model B", "Breakdown", "Group", "Tasks", "Annotations", "A win rate (task-weighted)", "CI (bootstrap over tasks)", "A win rate (annotation-weighted)", "Tie rate", "Neither rate"], rows)
    pb = p["position_bias"]
    lines += ["", f"- Position bias: share of decisive choices for response_1 = {fmt(pb['share_choosing_response_1'])}, CI {fmt_ci(pb['confidence_interval'])}"]
    for w in pb.get("warnings", []) + p.get("warnings", []):
        lines.append(f"- Warning `{w['code']}`: {w['message']}")
    lines += ["", "## Agreement", ""]
    for space in ("display_space", "model_space"):
        a = p["agreement"][space]
        lines.append(f"- {space.replace('_', ' ')}: percent agreement {fmt(a['percent_agreement'])} over {a['annotator_pairs_compared']} pair(s); Fleiss' kappa {fmt(a['fleiss_kappa'])} ({a['fleiss_tasks_used']} task(s) with {a['fleiss_ratings_per_task']} ratings)")
        for pair in a["cohens_kappa_pairs"]:
            lines.append(f"  - Cohen's kappa {pair['annotators'][0]} vs {pair['annotators'][1]}: {fmt(pair['cohens_kappa'])} over {pair['shared_tasks']} shared task(s)")
    lines.append(f"- {p['agreement']['note']}")
    lines += ["", "## Annotators", ""]
    lines += _table(["Annotator", "Annotations", "Tasks", "A", "B", "Tie", "Neither", "A win rate"], [[k, v["annotations"], v["tasks"], v["outcome_counts"]["A"], v["outcome_counts"]["B"], v["outcome_counts"]["tie"], v["outcome_counts"]["neither"], fmt(v["win_rate_a_annotation_weighted"])] for k, v in p["per_annotator"].items()])
    if p.get("elo"):
        e = p["elo"]
        lines += ["", "## Elo (optional, relative)", ""]
        lines += _table(["Model", "Rating", "Comparisons"], [[m, fmt(r, 1), e["comparisons_per_model"][m]] for m, r in e["ratings"].items()])
        lines += ["", f"_{e['limitations']}_"]
    lines += ["", "## Methodology", ""] + [f"- {k.replace('_', ' ')}: {v}" for k, v in p["methodology"].items()]
    lines += [""] + _provenance_md(p.get("provenance"))
    return "\n".join(lines).rstrip() + "\n"


# ---------------------------------------------------------------------------
# Audits
# ---------------------------------------------------------------------------


AUDIT_HEADERS = ["check_id", "severity", "title", "message", "record_ids", "run_ids", "classification", "remediation"]


def render_audit(payload: dict[str, Any], fmt_name: str) -> str:
    if fmt_name == "json":
        return to_json(payload)
    rows = [[f["check_id"], f["severity"], f["title"], f["message"], " ".join(f["record_ids"]), " ".join(f["run_ids"]), f.get("classification"), f.get("remediation")] for f in payload["findings"]]
    if fmt_name == "csv":
        return _csv(AUDIT_HEADERS, rows)
    if fmt_name != "markdown":
        raise ConfigurationError(f"Unsupported audit report format '{fmt_name}'")
    s = payload["summary"]
    lines = [f"# OpenBench AI quality audit ({payload['kind']})", "", f"- Subject: {escape_markdown_cell(payload['subject'].get('path') or payload['subject'].get('run_id') or payload['subject'].get('repeat_group_id'))}", f"- Findings: {s['errors']} error(s), {s['warnings']} warning(s), {s['info']} info", f"- {payload['non_mutation']}", ""]
    lines += _table(["Check", "Severity", "Title", "Records / runs", "Classification", "Message"], [[r[0], r[1], r[2], (r[4] or r[5])[:200], r[6] or "", r[3]] for r in rows])
    lines += ["", "## Parameters", ""] + [f"- {k}: {v}" for k, v in payload["parameters"].items()]
    return "\n".join(lines).rstrip() + "\n"
