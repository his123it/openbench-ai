"""Command-Line Interface for OpenBench AI (v0.2.0)."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Sequence
from pathlib import Path

from openbench import __version__
from openbench.catalog import DatasetCatalog
from openbench.comparison import ComparisonResult, ModelComparator
from openbench.configuration import ProfileLoader
from openbench.dataset import DatasetLoader
from openbench.evaluator import (
    DEFAULT_METRIC_NAMES,
    EvaluationConfig,
    EvaluationResult,
    Evaluator,
    ResponseLoader,
)
from openbench.exceptions import OpenBenchError
from openbench.history import ResultHistoryStore
from openbench.reporting import ReportGenerator


def _slugify(value: str) -> str:
    """Create a filesystem-safe slug from a model name."""
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "_", value.strip()).strip("_").lower()
    return slug or "model"


def build_parser() -> argparse.ArgumentParser:
    """Construct the OpenBench AI CLI argument parser (preserving all v0.1.0 commands and adding v0.2.0 commands)."""
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

    # 1. validate-dataset (v0.1.0 preserved)
    val_parser = subparsers.add_parser(
        "validate-dataset",
        help="Validate a JSONL dataset file or directory of JSONL files.",
    )
    val_parser.add_argument(
        "dataset_path",
        type=str,
        help="Path to a JSONL dataset file or directory.",
    )

    # 2. evaluate (v0.1.0 preserved + v0.2.0 --config, --category, --difficulty, --history)
    eval_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate one or more model response sets against a dataset or YAML configuration profile.",
    )
    eval_parser.add_argument(
        "--dataset",
        required=False,
        type=str,
        help="Path to a JSONL dataset file or directory (required unless --config specifies dataset.path).",
    )
    eval_parser.add_argument(
        "--config",
        required=False,
        type=str,
        help="Path to a YAML evaluation configuration profile.",
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
        default=None,
        help="Space-separated list of metrics to compute (default: exact_match accuracy keyword_coverage, or profile metrics).",
    )
    eval_parser.add_argument(
        "--category",
        nargs="+",
        default=None,
        help="Optional category filter(s) to apply to the dataset.",
    )
    eval_parser.add_argument(
        "--difficulty",
        nargs="+",
        default=None,
        help="Optional difficulty filter(s) (easy, medium, hard).",
    )
    eval_parser.add_argument(
        "--history-file",
        required=False,
        default=None,
        type=str,
        help="Optional path to local JSONL history file to record evaluation summary.",
    )
    eval_parser.add_argument(
        "--output",
        required=True,
        type=str,
        help="Output directory or file path (.json, .md, or .csv) for evaluation reports.",
    )

    # 3. compare (v0.1.0 preserved)
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
        help="Output file path (.md, .json, or .csv) or directory for the comparison report.",
    )

    # 4. list-datasets (v0.2.0 new)
    list_ds_parser = subparsers.add_parser(
        "list-datasets",
        help="List all built-in or directory JSONL datasets in the catalog.",
    )
    list_ds_parser.add_argument(
        "--path",
        type=str,
        default=None,
        help="Optional root directory of datasets (defaults to built-in datasets/ directory).",
    )
    list_ds_parser.add_argument(
        "--category",
        type=str,
        default=None,
        help="Filter listed datasets by category name.",
    )
    list_ds_parser.add_argument(
        "--difficulty",
        type=str,
        default=None,
        help="Filter listed datasets by difficulty level.",
    )
    list_ds_parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format for dataset listing (default: text).",
    )

    # 5. inspect-dataset (v0.2.0 new)
    inspect_ds_parser = subparsers.add_parser(
        "inspect-dataset",
        help="Inspect metadata, schema breakdown, and sample records of a dataset.",
    )
    inspect_ds_parser.add_argument(
        "dataset_path",
        type=str,
        help="Path to a JSONL dataset file or directory.",
    )
    inspect_ds_parser.add_argument(
        "--category",
        type=str,
        default=None,
        help="Optional category filter.",
    )
    inspect_ds_parser.add_argument(
        "--difficulty",
        type=str,
        default=None,
        help="Optional difficulty filter.",
    )
    inspect_ds_parser.add_argument(
        "--samples",
        type=int,
        default=2,
        help="Number of sample records to display (default: 2).",
    )
    inspect_ds_parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format (default: text).",
    )

    # 6. dataset-stats (v0.2.0 new)
    stats_ds_parser = subparsers.add_parser(
        "dataset-stats",
        help="Display detailed statistics for a dataset file or directory.",
    )
    stats_ds_parser.add_argument(
        "dataset_path",
        type=str,
        help="Path to a JSONL dataset file or directory.",
    )
    stats_ds_parser.add_argument(
        "--category",
        type=str,
        default=None,
        help="Optional category filter.",
    )
    stats_ds_parser.add_argument(
        "--difficulty",
        type=str,
        default=None,
        help="Optional difficulty filter.",
    )
    stats_ds_parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format (default: text).",
    )

    # 7. history (v0.2.0 new: `openbench history list` & `openbench history show EVALUATION_ID`)
    history_parser = subparsers.add_parser(
        "history",
        help="List or inspect local historical evaluation runs.",
    )
    history_subparsers = history_parser.add_subparsers(
        dest="history_action",
        help="History subcommands (list, show)",
    )
    hist_list_parser = history_subparsers.add_parser(
        "list",
        help="List recorded historical evaluation runs.",
    )
    hist_list_parser.add_argument(
        "--store",
        type=str,
        default=None,
        help="Path to JSONL history file (default: results/history.jsonl).",
    )
    hist_list_parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Filter history by model name.",
    )
    hist_list_parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Filter history by dataset name.",
    )
    hist_list_parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format (default: text).",
    )

    hist_show_parser = history_subparsers.add_parser(
        "show",
        help="Show details for a specific evaluation run by EVALUATION_ID.",
    )
    hist_show_parser.add_argument(
        "evaluation_id",
        type=str,
        help="Unique evaluation ID (e.g. eval_...).",
    )
    hist_show_parser.add_argument(
        "--store",
        type=str,
        default=None,
        help="Path to JSONL history file (default: results/history.jsonl).",
    )
    hist_show_parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format (default: text).",
    )

    # 8. export (v0.2.0 new: `openbench export --format csv ...`)
    export_parser = subparsers.add_parser(
        "export",
        help="Export an existing JSON evaluation or comparison result into CSV or Markdown.",
    )
    export_parser.add_argument(
        "--input",
        required=True,
        type=str,
        help="Path to an existing JSON evaluation or comparison report file.",
    )
    export_parser.add_argument(
        "--format",
        required=True,
        choices=["csv", "markdown", "json"],
        help="Target export format (csv, markdown, json).",
    )
    export_parser.add_argument(
        "--mode",
        choices=["individual", "aggregate"],
        default="individual",
        help="For CSV export of single-model results, export 'individual' items or 'aggregate' metrics (default: individual).",
    )
    export_parser.add_argument(
        "--output",
        required=True,
        type=str,
        help="Output file path for the exported report.",
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


def _handle_evaluate(args: argparse.Namespace) -> int:
    """Execute the `evaluate` command (supporting both direct CLI flags and `--config` profiles)."""
    profile = None
    output_formats = ["json", "markdown"]

    if args.config:
        profile = ProfileLoader.load_file(args.config)
        output_formats = list(profile.output_formats)

    dataset_path = args.dataset or (profile.dataset_path if profile else None)
    if not dataset_path:
        print(
            "Error: Either --dataset or a --config profile specifying dataset.path is required.",
            file=sys.stderr,
        )
        return 2

    if profile is not None:
        config = profile.to_evaluation_config(override_metrics=args.metrics)
        if args.category:
            config.categories = list(args.category)
        if args.difficulty:
            config.difficulties = list(args.difficulty)
    else:
        metric_names = list(args.metrics) if args.metrics else list(DEFAULT_METRIC_NAMES)
        config = EvaluationConfig(
            metric_names=metric_names,
            categories=list(args.category) if args.category else None,
            difficulties=list(args.difficulty) if args.difficulty else None,
        )

    dataset = DatasetLoader.load(dataset_path)
    response_sets = ResponseLoader.load_all(args.responses)
    evaluator = Evaluator(config=config)
    results = evaluator.evaluate_multiple(dataset, response_sets)

    out_path = Path(args.output)
    is_single_file = out_path.suffix.lower() in {".json", ".md", ".csv"} and len(results) == 1

    history_store: ResultHistoryStore | None = None
    if args.history_file:
        history_store = ResultHistoryStore(args.history_file)
    elif profile and profile.record_history:
        base_hist_dir = out_path.parent if is_single_file else out_path
        history_store = ResultHistoryStore(base_hist_dir / "history.jsonl")

    if is_single_file:
        res = results[0]
        suffix = out_path.suffix.lower()
        if suffix == ".json":
            ReportGenerator.save_json(res, out_path)
            md_companion = out_path.with_suffix(".md")
            ReportGenerator.save_markdown(res, md_companion)
            if history_store:
                history_store.record(res, report_path=out_path)
            print(f"Saved evaluation reports: {out_path}, {md_companion}")
        elif suffix == ".md":
            ReportGenerator.save_markdown(res, out_path)
            json_companion = out_path.with_suffix(".json")
            ReportGenerator.save_json(res, json_companion)
            if history_store:
                history_store.record(res, report_path=json_companion)
            print(f"Saved evaluation reports: {out_path}, {json_companion}")
        else:  # .csv
            ReportGenerator.save_csv(res, out_path)
            json_companion = out_path.with_suffix(".json")
            ReportGenerator.save_json(res, json_companion)
            if history_store:
                history_store.record(res, report_path=json_companion)
            print(f"Saved evaluation reports: {out_path}, {json_companion}")
        return 0

    out_dir = out_path
    out_dir.mkdir(parents=True, exist_ok=True)

    written_files: list[str] = []
    for res in results:
        slug = _slugify(res.model_name)
        json_file = out_dir / f"{slug}.json"
        md_file = out_dir / f"{slug}.md"
        csv_file = out_dir / f"{slug}.csv"

        if "json" in output_formats:
            ReportGenerator.save_json(res, json_file)
            written_files.append(str(json_file))
        if "markdown" in output_formats:
            ReportGenerator.save_markdown(res, md_file)
            written_files.append(str(md_file))
        if "csv" in output_formats:
            ReportGenerator.save_csv(res, csv_file)
            written_files.append(str(csv_file))

        if history_store:
            history_store.record(res, report_path=json_file if "json" in output_formats else None)

    print(
        f"Evaluated {len(results)} model(s) on '{dataset.name}' ({len(results[0].individual_results)} items). "
        f"Reports written to: {', '.join(written_files)}"
    )
    return 0


def _handle_compare(results_args: list[list[str]], output_arg: str) -> int:
    """Execute the `compare` command."""
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
    elif out_path.suffix.lower() == ".csv":
        ReportGenerator.save_csv(comparison, out_path)
        print(f"Comparison CSV report written to: {out_path}")
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


def _handle_list_datasets(args: argparse.Namespace) -> int:
    """Execute the `list-datasets` command."""
    catalog = DatasetCatalog(args.path)
    entries = catalog.list_datasets(category=args.category, difficulty=args.difficulty)
    if args.format == "json":
        print(json.dumps([e.to_dict() for e in entries], indent=2))
        return 0

    total_records = sum(e.total_records for e in entries)
    print(f"Found {len(entries)} dataset(s) ({total_records} total records):")
    for entry in entries:
        cats = ", ".join(f"{k}:{v}" for k, v in entry.categories.items())
        diffs = ", ".join(f"{k}:{v}" for k, v in entry.difficulties.items())
        print(
            f"  - {entry.name} (v{entry.version}): {entry.total_records} records "
            f"[categories: {cats}] [difficulties: {diffs}] ({entry.relative_path})"
        )
    return 0


def _handle_inspect_dataset(args: argparse.Namespace) -> int:
    """Execute the `inspect-dataset` command."""
    info = DatasetCatalog.inspect_dataset(
        args.dataset_path,
        category=args.category,
        difficulty=args.difficulty,
        sample_limit=args.samples,
    )
    if args.format == "json":
        print(json.dumps(info, indent=2))
        return 0

    print(f"Dataset Inspection: {info['name']} (v{info['version']})")
    print(f"  Source Path   : {info['source_path']}")
    print(f"  SHA-256       : {info['checksum']}")
    print(f"  Total Records : {info['total_records']}")
    print(f"  Categories    : {info['categories']}")
    print(f"  Difficulties  : {info['difficulties']}")
    print(f"  Languages     : {info['languages']}")
    if info["sample_records"]:
        print(f"  Sample Records ({len(info['sample_records'])}):")
        for sample in info["sample_records"]:
            print(f"    * [{sample['id']}] ({sample['category']}): {sample['prompt']}")
    return 0


def _handle_dataset_stats(args: argparse.Namespace) -> int:
    """Execute the `dataset-stats` command."""
    stats = DatasetCatalog.compute_stats(
        args.dataset_path,
        category=args.category,
        difficulty=args.difficulty,
    )
    if args.format == "json":
        print(json.dumps(stats, indent=2))
        return 0

    print(f"Dataset Statistics: {stats['name']} (v{stats['version']})")
    print(f"  Files Count      : {stats['files_count']}")
    print(f"  Total Records    : {stats['total_records']}")
    print(f"  SHA-256 Checksum : {stats['checksum']}")
    print(f"  Categories       : {stats['categories']}")
    print(f"  Difficulties     : {stats['difficulties']}")
    print(f"  Languages        : {stats['languages']}")
    print(f"  Prompt Length    : {stats['prompt_char_length']}")
    print(f"  Answer Length    : {stats['expected_answer_char_length']}")
    return 0


def _handle_history(args: argparse.Namespace) -> int:
    """Execute `openbench history list` or `openbench history show`."""
    if not args.history_action:
        print("Error: Specify a history action ('list' or 'show').", file=sys.stderr)
        return 2

    store = ResultHistoryStore(args.store)
    if args.history_action == "list":
        entries = store.list_entries(model_name=args.model, dataset_name=args.dataset)
        if args.format == "json":
            print(json.dumps([e.to_dict() for e in entries], indent=2))
            return 0
        if not entries:
            print("No historical evaluation runs found.")
            return 0
        print(f"Recorded Evaluation Runs ({len(entries)}):")
        for e in entries:
            scores_str = ", ".join(f"{k}={v:.4f}" for k, v in e.result_summary.items())
            print(
                f"  - {e.evaluation_id} | {e.timestamp} | {e.model_name} (v{e.model_version}) "
                f"| dataset={e.dataset_name} | {scores_str}"
            )
        return 0

    if args.history_action == "show":
        try:
            entry = store.get_entry(args.evaluation_id)
        except KeyError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        if args.format == "json":
            print(json.dumps(entry.to_dict(), indent=2))
            return 0
        print(f"Evaluation Run: {entry.evaluation_id}")
        print(f"  Timestamp     : {entry.timestamp}")
        print(f"  Model         : {entry.model_name} (v{entry.model_version})")
        print(f"  Dataset       : {entry.dataset_name} (v{entry.dataset_version})")
        print(f"  Checksum      : {entry.dataset_checksum}")
        print(f"  Evaluated     : {entry.evaluated_items} / {entry.total_items} (missing: {entry.missing_responses_count})")
        print(f"  Scores        : {entry.result_summary}")
        if entry.report_path:
            print(f"  Report Path   : {entry.report_path}")
        return 0

    return 2


def _handle_export(args: argparse.Namespace) -> int:
    """Execute `openbench export --input ... --format csv|markdown|json --output ...`."""
    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(f"Input result file does not exist: {in_path}")

    raw_text = in_path.read_text(encoding="utf-8")
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise OpenBenchError(f"Malformed JSON in {in_path}: {exc.msg}") from exc

    out_path = Path(args.output)
    # Detect whether input JSON is an EvaluationResult or a ComparisonResult
    if "models" in payload and "highest_by_metric" in payload:
        # Reconstruct ComparisonResult or export directly
        from openbench.comparison import MetricLeader, ModelComparisonEntry

        entries = [
            ModelComparisonEntry(
                model_name=m["model_name"],
                model_version=m["model_version"],
                dataset_name=m.get("dataset_name", "unknown"),
                total_dataset_items=int(m.get("total_dataset_items", 0)),
                evaluated_items=int(m.get("evaluated_items", 0)),
                missing_responses_count=int(m.get("missing_responses_count", 0)),
                exact_match_score=m.get("exact_match"),
                accuracy_score=m.get("accuracy"),
                keyword_coverage_score=m.get("keyword_coverage"),
                metric_scores=dict(m.get("metric_scores", {})),
                weighted_score=m.get("weighted_score"),
            )
            for m in payload["models"]
        ]
        leaders = {
            k: MetricLeader(
                metric_name=v["metric_name"],
                highest_score=float(v["highest_score"]),
                leading_models=list(v["leading_models"]),
                runner_up_score=v.get("runner_up_score"),
                score_margin=float(v.get("score_margin", 0.0)),
            )
            for k, v in payload.get("highest_by_metric", {}).items()
        }
        target_obj: EvaluationResult | ComparisonResult = ComparisonResult(
            comparison_timestamp=str(payload.get("comparison_timestamp", "")),
            dataset_name=str(payload.get("dataset_name", "unknown")),
            total_dataset_items=int(payload.get("total_dataset_items", 0)),
            models=entries,
            metrics_compared=list(payload.get("metrics_compared", [])),
            highest_by_metric=leaders,
            warnings=list(payload.get("warnings", [])),
            methodology_note=str(payload.get("methodology_note", "")),
            weights_used=payload.get("weights_used"),
        )
    else:
        target_obj = EvaluationResult.from_dict(payload)

    if args.format == "csv":
        ReportGenerator.save_csv(target_obj, out_path, mode=args.mode)
    elif args.format == "markdown":
        ReportGenerator.save_markdown(target_obj, out_path)
    else:
        ReportGenerator.save_json(target_obj, out_path)

    print(f"Exported {args.format.upper()} report to: {out_path}")
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
            return _handle_evaluate(args)
        if args.command == "compare":
            return _handle_compare(
                results_args=args.results,
                output_arg=args.output,
            )
        if args.command == "list-datasets":
            return _handle_list_datasets(args)
        if args.command == "inspect-dataset":
            return _handle_inspect_dataset(args)
        if args.command == "dataset-stats":
            return _handle_dataset_stats(args)
        if args.command == "history":
            return _handle_history(args)
        if args.command == "export":
            return _handle_export(args)
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
