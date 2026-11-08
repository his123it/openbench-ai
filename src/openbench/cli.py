"""Command-Line Interface for OpenBench AI."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Sequence
from pathlib import Path

from openbench import __version__
from openbench.comparison import ModelComparator
from openbench.dataset import DatasetLoader
from openbench.evaluator import (
    DEFAULT_METRIC_NAMES,
    EvaluationConfig,
    EvaluationResult,
    Evaluator,
    ResponseLoader,
)
from openbench.exceptions import OpenBenchError
from openbench.reporting import ReportGenerator


def _slugify(value: str) -> str:
    """Create a filesystem-safe slug from a model name."""
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "_", value.strip()).strip("_").lower()
    return slug or "model"


def build_parser() -> argparse.ArgumentParser:
    """Construct the OpenBench AI CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="openbench",
        description="OpenBench AI — Modular, open-source LLM evaluation and benchmarking toolkit.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"openbench {__version__}",
        help="Show the OpenBench AI version and exit.",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available OpenBench commands")

    # 1. validate-dataset
    val_parser = subparsers.add_parser(
        "validate-dataset",
        help="Validate a JSONL dataset file or directory of JSONL files.",
    )
    val_parser.add_argument(
        "dataset_path",
        type=str,
        help="Path to a JSONL dataset file or directory.",
    )

    # 2. evaluate
    eval_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate one or more model response sets against a dataset.",
    )
    eval_parser.add_argument(
        "--dataset",
        required=True,
        type=str,
        help="Path to a JSONL dataset file or directory.",
    )
    eval_parser.add_argument(
        "--responses",
        required=True,
        type=str,
        help="Path to a JSON file containing model responses.",
    )
    eval_parser.add_argument(
        "--metrics",
        nargs="+",
        default=list(DEFAULT_METRIC_NAMES),
        help="Space-separated list of metrics to compute (default: exact_match accuracy keyword_coverage).",
    )
    eval_parser.add_argument(
        "--output",
        required=True,
        type=str,
        help="Output directory or file path (.json or .md) for evaluation reports.",
    )

    # 3. compare
    cmp_parser = subparsers.add_parser(
        "compare",
        help="Compare two or more JSON evaluation result files.",
    )
    cmp_parser.add_argument(
        "--results",
        action="append",
        required=True,
         nargs="+",
        help="Paths to evaluation result JSON files (provide at least two files).",
    )
    cmp_parser.add_argument(
        "--output",
        required=True,
        type=str,
        help="Output file path (.md or .json) or directory for the comparison report.",
    )

    return parser


def _handle_validate_dataset(dataset_path: str) -> int:
    """Execute the `validate-dataset` command."""
    dataset = DatasetLoader.load(dataset_path)
    summary = dataset.summary()
    categories_str = ", ".join(
        f"{cat}: {cnt}" for cat, cnt in summary["categories"].items()
    )
    print(
        f"Dataset valid: '{dataset.name}' ({len(dataset)} records; categories: {categories_str})"
    )
    return 0


def _handle_evaluate(
    dataset_path: str,
    responses_path: str,
    metric_names: list[str],
    output_arg: str,
) -> int:
    """Execute the `evaluate` command."""
    dataset = DatasetLoader.load(dataset_path)
    response_sets = ResponseLoader.load_all(responses_path)
    config = EvaluationConfig(metric_names=metric_names)
    evaluator = Evaluator(config=config)
    results = evaluator.evaluate_multiple(dataset, response_sets)

    out_path = Path(output_arg)
    is_single_file = out_path.suffix.lower() in {".json", ".md"} and len(results) == 1

    if is_single_file:
        res = results[0]
        if out_path.suffix.lower() == ".json":
            ReportGenerator.save_json(res, out_path)
            md_companion = out_path.with_suffix(".md")
            ReportGenerator.save_markdown(res, md_companion)
            print(f"Saved evaluation reports: {out_path}, {md_companion}")
        else:
            ReportGenerator.save_markdown(res, out_path)
            json_companion = out_path.with_suffix(".json")
            ReportGenerator.save_json(res, json_companion)
            print(f"Saved evaluation reports: {out_path}, {json_companion}")
        return 0

    # Directory output (standard for `openbench evaluate --output results/`)
    out_dir = out_path
    out_dir.mkdir(parents=True, exist_ok=True)

    written_files: list[str] = []
    for res in results:
        slug = _slugify(res.model_name)
        json_file = out_dir / f"{slug}.json"
        md_file = out_dir / f"{slug}.md"
        ReportGenerator.save_json(res, json_file)
        ReportGenerator.save_markdown(res, md_file)
        written_files.extend([str(json_file), str(md_file)])

    print(
        f"Evaluated {len(results)} model(s) on '{dataset.name}' ({len(dataset)} items). "
        f"Reports written to: {', '.join(written_files)}"
    )
    return 0


def _handle_compare(results_args: list[list[str]], output_arg: str) -> int:
    """Execute the `compare` command."""
    # Flatten `--results a.json --results b.json` or `--results a.json b.json`
    flat_paths: list[str] = [p for group in results_args for p in group]
    if len(flat_paths) < 2:
        print(
            "Error: At least two result files must be provided via --results for comparison.",
            file=sys.stderr,
        )
        return 2

    eval_results: list[EvaluationResult] = []
    for path_str in flat_paths:
        target = Path(path_str)
        if not target.exists():
            raise FileNotFoundError(f"Evaluation result file does not exist: {target}")
        raw_text = target.read_text(encoding="utf-8")
        try:
            payload = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            raise OpenBenchError(
                f"Malformed JSON in result file {target}: {exc.msg}"
            ) from exc
        eval_results.append(EvaluationResult.from_dict(payload))

    comparison = ModelComparator.compare(eval_results)
    out_path = Path(output_arg)

    if out_path.suffix.lower() == ".json":
        ReportGenerator.save_json(comparison, out_path)
        print(f"Comparison JSON report written to: {out_path}")
    elif out_path.suffix.lower() == ".md":
        ReportGenerator.save_markdown(comparison, out_path)
        json_companion = out_path.with_suffix(".json")
        ReportGenerator.save_json(comparison, json_companion)
        print(f"Comparison reports written to: {out_path}, {json_companion}")
    else:
        out_path.mkdir(parents=True, exist_ok=True)
        md_file = out_path / "comparison.md"
        json_file = out_path / "comparison.json"
        ReportGenerator.save_markdown(comparison, md_file)
        ReportGenerator.save_json(comparison, json_file)
        print(f"Comparison reports written to: {md_file}, {json_file}")

    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Main entry point for the `openbench` CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 2

    try:
        if args.command == "validate-dataset":
            return _handle_validate_dataset(args.dataset_path)
        if args.command == "evaluate":
            return _handle_evaluate(
                dataset_path=args.dataset,
                responses_path=args.responses,
                metric_names=args.metrics,
                output_arg=args.output,
            )
        if args.command == "compare":
            return _handle_compare(
                results_args=args.results,
                output_arg=args.output,
            )
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except OpenBenchError as exc:
        print(f"Validation/Evaluation Error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # pragma: no cover
        print(f"Unexpected Error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
