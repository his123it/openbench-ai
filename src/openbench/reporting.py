"""Report generation (JSON, Markdown, CSV, and Benchmark Reproducibility Card) for OpenBench AI."""

from __future__ import annotations

import csv
import io
import json
import platform
import sys
from pathlib import Path
from typing import Any

from openbench.comparison import ComparisonResult
from openbench.evaluator import EvaluationResult

_SECRET_SUBSTRINGS = ("key", "token", "secret", "password", "credential", "auth")


def _redact_secrets(config_dict: dict[str, Any]) -> dict[str, Any]:
    """Ensure API keys, tokens, or credentials are never serialized into reproducibility cards."""
    cleaned: dict[str, Any] = {}
    for k, v in config_dict.items():
        if any(sub in k.lower() for sub in _SECRET_SUBSTRINGS):
            cleaned[k] = "[REDACTED]"
        elif isinstance(v, dict):
            cleaned[k] = _redact_secrets(v)
        else:
            cleaned[k] = v
    return cleaned


class ReportGenerator:
    """Generates machine-readable JSON, CSV, and human-readable Markdown evaluation reports."""

    @staticmethod
    def to_json(result: EvaluationResult | ComparisonResult, indent: int = 2) -> str:
        """Serialize an EvaluationResult or ComparisonResult to a formatted JSON string."""
        payload = result.to_dict()
        if isinstance(result, EvaluationResult):
            payload["reproducibility_card"] = ReportGenerator.build_reproducibility_card(result)
        return json.dumps(payload, indent=indent, sort_keys=False) + "\n"

    @staticmethod
    def save_json(result: EvaluationResult | ComparisonResult, output_path: str | Path) -> Path:
        """Write JSON report to disk and return the resolved path."""
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(ReportGenerator.to_json(result), encoding="utf-8")
        return target

    @staticmethod
    def to_csv(
        result: EvaluationResult | ComparisonResult,
        mode: str = "individual",
    ) -> str:
        """Serialize an EvaluationResult or ComparisonResult to CSV format.

        For `EvaluationResult`:
        - `mode="individual"` (default): One row per dataset item with per-metric scores and pass flags.
        - `mode="aggregate"`: One row per metric with aggregate score, passed items, and missing items.
        For `ComparisonResult`:
        - One row per compared model with all metric scores.
        """
        buf = io.StringIO()
        writer = csv.writer(buf)

        if isinstance(result, ComparisonResult):
            headers = [
                "model_name",
                "model_version",
                "dataset_name",
                "evaluated_items",
                "total_dataset_items",
                "missing_responses_count",
                *result.metrics_compared,
            ]
            if result.weights_used is not None:
                headers.append("weighted_score")
            writer.writerow(headers)

            for entry in result.models:
                row: list[Any] = [
                    entry.model_name,
                    entry.model_version,
                    entry.dataset_name,
                    entry.evaluated_items,
                    entry.total_dataset_items,
                    entry.missing_responses_count,
                ]
                for m_name in result.metrics_compared:
                    sc = entry.metric_scores.get(m_name)
                    row.append(f"{sc:.6f}" if sc is not None else "")
                if result.weights_used is not None:
                    row.append(
                        f"{entry.weighted_score:.6f}"
                        if entry.weighted_score is not None
                        else ""
                    )
                writer.writerow(row)
            return buf.getvalue()

        if mode == "aggregate":
            writer.writerow(
                [
                    "evaluation_id",
                    "model_name",
                    "model_version",
                    "dataset_name",
                    "metric_name",
                    "metric_version",
                    "score",
                    "passed_items",
                    "evaluated_items",
                    "total_items",
                ]
            )
            ds_name = result.dataset_summary.get("name", "unknown")
            for m_name, agg in result.aggregate_results.items():
                writer.writerow(
                    [
                        result.evaluation_id,
                        result.model_name,
                        result.model_version,
                        ds_name,
                        m_name,
                        agg.metric_version,
                        f"{agg.score:.6f}",
                        agg.passed_items,
                        agg.evaluated_items,
                        agg.total_items,
                    ]
                )
            return buf.getvalue()

        # Default: individual item results CSV
        metric_names = list(result.aggregate_results.keys())
        headers = [
            "id",
            "category",
            "difficulty",
            "status",
            "is_valid_response",
            "latency_ms",
        ]
        for m_name in metric_names:
            headers.extend([f"{m_name}_score", f"{m_name}_passed"])
        headers.extend(["expected_answer", "model_response"])
        writer.writerow(headers)

        for item in result.individual_results:
            row = [
                item.id,
                item.category,
                item.difficulty,
                item.status,
                item.is_valid_response,
                f"{item.latency_ms:.3f}" if item.latency_ms is not None else "",
            ]
            for m_name in metric_names:
                m_res = item.metrics.get(m_name)
                if m_res is not None:
                    row.extend([f"{m_res.score:.6f}", m_res.passed])
                else:
                    row.extend(["", ""])
            row.extend([item.expected_answer, item.model_response or ""])
            writer.writerow(row)
        return buf.getvalue()

    @staticmethod
    def save_csv(
        result: EvaluationResult | ComparisonResult,
        output_path: str | Path,
        mode: str = "individual",
    ) -> Path:
        """Write CSV report to disk and return the resolved path."""
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(ReportGenerator.to_csv(result, mode=mode), encoding="utf-8")
        return target

    @staticmethod
    def build_reproducibility_card(result: EvaluationResult) -> dict[str, Any]:
        """Construct the structured Benchmark Reproducibility Card data dictionary.

        Preserves all v0.1.0 keys while adding v0.2.0 reproducibility fields
        (model_provider, dataset_identifier, dataset_checksum, python_version,
        operating_system, inference_configuration, metric_configurations, warnings).
        """
        ds = result.dataset_summary
        safe_inference_cfg = _redact_secrets(result.inference_config)
        safe_eval_cfg = _redact_secrets(result.config)

        return {
            "evaluation_id": result.evaluation_id,
            "model_name": result.model_name,
            "model_version": result.model_version,
            "model_provider": result.model_provider,
            "dataset_name": ds.get("name", "unknown"),
            "dataset_identifier": ds.get("name", "unknown"),
            "dataset_version": ds.get("version", "0.1.0"),
            "dataset_source": ds.get("source_path", "unknown"),
            "dataset_checksum": ds.get("checksum", "unknown"),
            "metrics": [
                {"name": m.get("name"), "version": m.get("version", "0.1.0")}
                for m in result.metrics_used
            ],
            "metric_configurations": list(result.metrics_used),
            "evaluation_timestamp": result.evaluation_timestamp,
            "python_version": platform.python_version(),
            "openbench_version": result.project_version,
            "operating_system": f"{platform.system()} {platform.release()}",
            "evaluation_configuration": safe_eval_cfg,
            "inference_configuration": safe_inference_cfg,
            "number_of_test_cases": result.total_dataset_items,
            "evaluated_test_cases": result.evaluated_items_count,
            "missing_responses": result.missing_responses_count,
            "warnings": list(result.warnings),
            "results": {
                m_name: round(agg.score, 6)
                for m_name, agg in result.aggregate_results.items()
            },
            "known_limitations": list(result.limitations),
        }

    @staticmethod
    def format_reproducibility_card_markdown(result: EvaluationResult) -> str:
        """Render the Benchmark Reproducibility Card as a self-contained Markdown section."""
        card = ReportGenerator.build_reproducibility_card(result)
        metrics_str = ", ".join(
            f"`{m['name']}` (v{m['version']})" for m in card["metrics"]
        )
        results_str = ", ".join(
            f"**{k}**: {v:.4f} ({v * 100:.2f}%)" for k, v in card["results"].items()
        )
        norm_cfg = card["evaluation_configuration"].get("normalization", {})
        norm_summary = (
            f"strip_whitespace={norm_cfg.get('strip_whitespace', True)}, "
            f"case_sensitive={norm_cfg.get('case_sensitive', False)}, "
            f"collapse_whitespace={norm_cfg.get('collapse_whitespace', True)}, "
            f"strip_punctuation={norm_cfg.get('strip_punctuation', False)}"
        )

        lines = [
            "## Benchmark Reproducibility Card",
            "",
            "| Field | Value |",
            "| :--- | :--- |",
            f"| **Evaluation ID** | `{card['evaluation_id']}` |",
            f"| **Model Name & Version** | `{card['model_name']}` (v`{card['model_version']}`) |",
            f"| **Model Provider** | `{card['model_provider']}` |",
            f"| **Dataset Name & Version** | `{card['dataset_name']}` (v`{card['dataset_version']}`) |",
            f"| **Dataset Source Path** | `{card['dataset_source']}` |",
            f"| **Dataset SHA-256 Checksum** | `{card['dataset_checksum']}` |",
            f"| **Metrics & Versions** | {metrics_str} |",
            f"| **Evaluation Timestamp (UTC)** | `{card['evaluation_timestamp']}` |",
            f"| **Python Environment** | Python `{card['python_version']}` on `{card['operating_system']}` |",
            f"| **OpenBench AI Version** | `v{card['openbench_version']}` |",
            f"| **Normalization Config** | `{norm_summary}` |",
            f"| **Number of Test Cases** | {card['number_of_test_cases']} total ({card['evaluated_test_cases']} evaluated, {card['missing_responses']} missing) |",
            f"| **Aggregate Results** | {results_str} |",
            "",
            "### Known Limitations",
        ]
        for lim in card["known_limitations"]:
            lines.append(f"- {lim}")
        return "\n".join(lines)

    @staticmethod
    def to_markdown(result: EvaluationResult) -> str:
        """Generate a comprehensive human-readable Markdown report for an EvaluationResult."""
        ds = result.dataset_summary
        lines: list[str] = [
            f"# OpenBench AI Evaluation Report: {result.model_name} (v{result.model_version})",
            "",
            "## Evaluation Summary",
            "",
            f"- **Evaluation ID**: `{result.evaluation_id}`",
            f"- **Model**: `{result.model_name}` (version `{result.model_version}`, provider `{result.model_provider}`)",
            f"- **Dataset**: `{ds.get('name', 'unknown')}` (version `{ds.get('version', '0.1.0')}`)",
            f"- **Evaluation Timestamp**: `{result.evaluation_timestamp}`",
            f"- **OpenBench AI Version**: `v{result.project_version}`",
            f"- **Test Cases Evaluated**: {result.evaluated_items_count} / {result.total_dataset_items} "
            f"(Missing responses: {result.missing_responses_count})",
            "",
            "## Dataset Details",
            "",
            f"- **Source Path**: `{ds.get('source_path', 'unknown')}`",
            f"- **SHA-256 Checksum**: `{ds.get('checksum', 'unknown')}`",
            f"- **Total Records**: {ds.get('total_records', 0)}",
            f"- **Categories**: {', '.join(f'`{k}` ({v})' for k, v in ds.get('categories', {}).items()) or 'N/A'}",
            f"- **Difficulty Breakdown**: {', '.join(f'`{k}` ({v})' for k, v in ds.get('difficulties', {}).items()) or 'N/A'}",
            "",
            "## Model Details",
            "",
            f"- **Model Name**: `{result.model_name}`",
            f"- **Model Version**: `{result.model_version}`",
            f"- **Model Provider**: `{result.model_provider}`",
            f"- **Evaluated Items**: {result.evaluated_items_count}",
            f"- **Missing Responses**: {result.missing_responses_count}",
            f"- **Unknown Response IDs Ignored**: {len(result.unknown_response_ids)}",
            "",
            "## Metric Results",
            "",
            "| Metric | Version | Aggregate Score | Percentage | Passed Items | Evaluated Items | Total Items |",
            "| :--- | :--- | ---: | ---: | ---: | ---: | ---: |",
        ]

        for m_name, agg in result.aggregate_results.items():
            pct = agg.score * 100.0
            lines.append(
                f"| `{m_name}` | `{agg.metric_version}` | {agg.score:.4f} | {pct:.2f}% | "
                f"{agg.passed_items} | {agg.evaluated_items} | {agg.total_items} |"
            )

        metric_cols = list(result.aggregate_results.keys())
        if result.category_breakdown:
            lines.extend(["", "## Category-Wise Breakdown", ""])
            header = "| Category | Evaluated / Total | Missing | " + " | ".join(
                f"`{m}`" for m in metric_cols
            ) + " |"
            sep = "| :--- | ---: | ---: | " + " | ".join("---:" for _ in metric_cols) + " |"
            lines.extend([header, sep])
            for cat_name, c_info in result.category_breakdown.items():
                m_cells = [
                    f"{c_info['metrics'].get(m, 0.0) * 100:.2f}% ({c_info['metrics'].get(m, 0.0):.4f})"
                    for m in metric_cols
                ]
                lines.append(
                    f"| `{cat_name}` | {c_info['evaluated_items']} / {c_info['total_items']} | "
                    f"{c_info['missing_items']} | " + " | ".join(m_cells) + " |"
                )

        if result.difficulty_breakdown:
            lines.extend(["", "## Difficulty-Wise Breakdown", ""])
            header = "| Difficulty | Evaluated / Total | Missing | " + " | ".join(
                f"`{m}`" for m in metric_cols
            ) + " |"
            sep = "| :--- | ---: | ---: | " + " | ".join("---:" for _ in metric_cols) + " |"
            lines.extend([header, sep])
            for diff_name, d_info in result.difficulty_breakdown.items():
                m_cells = [
                    f"{d_info['metrics'].get(m, 0.0) * 100:.2f}% ({d_info['metrics'].get(m, 0.0):.4f})"
                    for m in metric_cols
                ]
                lines.append(
                    f"| `{diff_name}` | {d_info['evaluated_items']} / {d_info['total_items']} | "
                    f"{d_info['missing_items']} | " + " | ".join(m_cells) + " |"
                )

        if result.warnings:
            lines.extend(["", "## Warnings", ""])
            for warn in result.warnings:
                lines.append(f"- **Warning**: {warn}")

        lines.extend(["", "## Limitations", ""])
        for lim in result.limitations:
            lines.append(f"- {lim}")

        lines.extend(["", ReportGenerator.format_reproducibility_card_markdown(result), ""])
        return "\n".join(lines)

    @staticmethod
    def comparison_to_markdown(comparison: ComparisonResult) -> str:
        """Generate a readable Markdown report for a multi-model ComparisonResult."""
        lines: list[str] = [
            f"# OpenBench AI Model Comparison Report: {comparison.dataset_name}",
            "",
            "## Comparison Summary",
            "",
            f"- **Dataset**: `{comparison.dataset_name}` ({comparison.total_dataset_items} test cases)",
            f"- **Models Compared**: {len(comparison.models)}",
            f"- **Metrics Evaluated**: {', '.join(f'`{m}`' for m in comparison.metrics_compared)}",
            f"- **Comparison Timestamp**: `{comparison.comparison_timestamp}`",
            "",
            "## Model Comparison Table",
            "",
        ]

        # Build dynamic columns for all compared metrics while preserving standard v0.1.0 columns
        extra_metrics = [
            m for m in comparison.metrics_compared
            if m not in {"exact_match", "accuracy", "keyword_coverage"}
        ]
        headers = [
            "Model",
            "Version",
            "Evaluated / Total",
            "Missing",
            "Exact Match",
            "Accuracy",
            "Keyword Coverage",
            *[f"`{m}`" for m in extra_metrics],
        ]
        if comparison.weights_used is not None:
            headers.append("User-Weighted Score")

        aligns = [":---", ":---", "---:", "---:", "---:", "---:", "---:", *["---:" for _ in extra_metrics]]
        if comparison.weights_used is not None:
            aligns.append("---:")

        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(aligns) + " |")

        for entry in comparison.models:
            em = (
                f"{entry.exact_match_score * 100:.2f}% ({entry.exact_match_score:.4f})"
                if entry.exact_match_score is not None
                else "N/A"
            )
            acc = (
                f"{entry.accuracy_score * 100:.2f}% ({entry.accuracy_score:.4f})"
                if entry.accuracy_score is not None
                else "N/A"
            )
            kw = (
                f"{entry.keyword_coverage_score * 100:.2f}% ({entry.keyword_coverage_score:.4f})"
                if entry.keyword_coverage_score is not None
                else "N/A"
            )
            row_cells = [
                f"`{entry.model_name}`",
                f"`{entry.model_version}`",
                f"{entry.evaluated_items} / {entry.total_dataset_items}",
                str(entry.missing_responses_count),
                em,
                acc,
                kw,
            ]
            for m in extra_metrics:
                sc = entry.metric_scores.get(m)
                row_cells.append(f"{sc * 100:.2f}% ({sc:.4f})" if sc is not None else "N/A")
            if comparison.weights_used is not None:
                ws = entry.weighted_score
                row_cells.append(f"{ws * 100:.2f}% ({ws:.4f})" if ws is not None else "N/A")

            lines.append("| " + " | ".join(row_cells) + " |")

        lines.extend(
            [
                "",
                "## Highest Score by Metric",
                "",
                "| Metric | Highest Score | Percentage | Margin Over Runner-Up | Leading Model(s) |",
                "| :--- | ---: | ---: | ---: | :--- |",
            ]
        )

        for m_name, leader in comparison.highest_by_metric.items():
            pct = leader.highest_score * 100.0
            models_str = ", ".join(f"`{m}`" for m in leader.leading_models)
            margin_str = f"+{leader.score_margin:.4f}" if not (len(leader.leading_models) > 1) else "Tie (0.0000)"
            lines.append(
                f"| `{m_name}` | {leader.highest_score:.4f} | {pct:.2f}% | {margin_str} | {models_str} |"
            )

        if comparison.relative_performance:
            lines.extend(["", "## Relative Performance Summary", ""])
            for summary_line in comparison.relative_performance:
                lines.append(f"- {summary_line}")

        if comparison.warnings:
            lines.extend(["", "## Warnings", ""])
            for warn in comparison.warnings:
                lines.append(f"- **Warning**: {warn}")

        lines.extend(
            [
                "",
                "## Methodology & Limitations",
                "",
                f"- {comparison.methodology_note}",
                "- Keyword coverage measures lexical overlap only and does not imply semantic correctness.",
                "- Reference-based deterministic metrics do not capture stylistic quality or open-ended reasoning depth.",
                "",
            ]
        )
        return "\n".join(lines)

    @staticmethod
    def save_markdown(
        result: EvaluationResult | ComparisonResult,
        output_path: str | Path,
    ) -> Path:
        """Write Markdown report to disk and return the resolved path."""
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(result, ComparisonResult):
            content = ReportGenerator.comparison_to_markdown(result)
        else:
            content = ReportGenerator.to_markdown(result)
        target.write_text(content, encoding="utf-8")
        return target
