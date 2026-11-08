"""Report generation (JSON, Markdown, and Benchmark Reproducibility Card) for OpenBench AI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from openbench.comparison import ComparisonResult
from openbench.evaluator import EvaluationResult


class ReportGenerator:
    """Generates machine-readable JSON and human-readable Markdown evaluation reports."""

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
    def build_reproducibility_card(result: EvaluationResult) -> dict[str, Any]:
        """Construct the structured Benchmark Reproducibility Card data dictionary."""
        return {
            "model_name": result.model_name,
            "model_version": result.model_version,
            "dataset_name": result.dataset_summary.get("name", "unknown"),
            "dataset_version": result.dataset_summary.get("version", "0.1.0"),
            "dataset_source": result.dataset_summary.get("source_path", "unknown"),
            "metrics": [
                {"name": m.get("name"), "version": m.get("version", "0.1.0")}
                for m in result.metrics_used
            ],
            "evaluation_timestamp": result.evaluation_timestamp,
            "openbench_version": result.project_version,
            "evaluation_configuration": dict(result.config),
            "number_of_test_cases": result.total_dataset_items,
            "evaluated_test_cases": result.evaluated_items_count,
            "missing_responses": result.missing_responses_count,
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
            f"| **Model Name & Version** | `{card['model_name']}` (v`{card['model_version']}`) |",
            f"| **Dataset Name & Version** | `{card['dataset_name']}` (v`{card['dataset_version']}`) |",
            f"| **Dataset Source Path** | `{card['dataset_source']}` |",
            f"| **Metrics & Versions** | {metrics_str} |",
            f"| **Evaluation Timestamp (UTC)** | `{card['evaluation_timestamp']}` |",
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
            f"- **Model**: `{result.model_name}` (version `{result.model_version}`)",
            f"- **Dataset**: `{ds.get('name', 'unknown')}` (version `{ds.get('version', '0.1.0')}`)",
            f"- **Evaluation Timestamp**: `{result.evaluation_timestamp}`",
            f"- **OpenBench AI Version**: `v{result.project_version}`",
            f"- **Test Cases Evaluated**: {result.evaluated_items_count} / {result.total_dataset_items} "
            f"(Missing responses: {result.missing_responses_count})",
            "",
            "## Dataset Details",
            "",
            f"- **Source Path**: `{ds.get('source_path', 'unknown')}`",
            f"- **Total Records**: {ds.get('total_records', 0)}",
            f"- **Categories**: {', '.join(f'`{k}` ({v})' for k, v in ds.get('categories', {}).items()) or 'N/A'}",
            f"- **Difficulty Breakdown**: {', '.join(f'`{k}` ({v})' for k, v in ds.get('difficulties', {}).items()) or 'N/A'}",
            "",
            "## Model Details",
            "",
            f"- **Model Name**: `{result.model_name}`",
            f"- **Model Version**: `{result.model_version}`",
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
            "| Model | Version | Evaluated / Total | Missing | Exact Match | Accuracy | Keyword Coverage |",
            "| :--- | :--- | ---: | ---: | ---: | ---: | ---: |",
        ]

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
            lines.append(
                f"| `{entry.model_name}` | `{entry.model_version}` | "
                f"{entry.evaluated_items} / {entry.total_dataset_items} | "
                f"{entry.missing_responses_count} | {em} | {acc} | {kw} |"
            )

        lines.extend(
            [
                "",
                "## Highest Score by Metric",
                "",
                "| Metric | Highest Score | Percentage | Leading Model(s) |",
                "| :--- | ---: | ---: | :--- |",
            ]
        )

        for m_name, leader in comparison.highest_by_metric.items():
            pct = leader.highest_score * 100.0
            models_str = ", ".join(f"`{m}`" for m in leader.leading_models)
            lines.append(
                f"| `{m_name}` | {leader.highest_score:.4f} | {pct:.2f}% | {models_str} |"
            )

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
