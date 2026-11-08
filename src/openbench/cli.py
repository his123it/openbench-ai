"""Command-Line Interface for OpenBench AI (v0.3.0)."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from openbench import __version__
from openbench.adapters import (
    DEFAULT_OLLAMA_ENDPOINT,
    GenerationConfig,
    create_provider,
    list_registered_providers,
)
from openbench.analytics import BenchmarkAnalytics
from openbench.cache import ResponseCache
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
from openbench.execution import ExecutionConfig, RetryPolicy
from openbench.experiments import ExperimentManager
from openbench.history import ResultHistoryStore
from openbench.reporting import ReportGenerator


def _slugify(value: str) -> str:
    """Create a filesystem-safe slug from a model name."""
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "_", value.strip()).strip("_").lower()
    return slug or "model"


def build_parser() -> argparse.ArgumentParser:
    """Construct the OpenBench AI CLI argument parser (preserving all v0.1.0 & v0.2.0 commands + v0.3.0 commands)."""
    parser = argparse.ArgumentParser(
        prog="openbench",
        description="OpenBench AI — Modular, open-source LLM experiment execution and benchmarking platform.",
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

    # 2. evaluate (v0.1.0 & v0.2.0 preserved)
    eval_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate one or more pre-generated model response files against a dataset or YAML profile.",
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
        help="Space-separated list of metrics to compute.",
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
        "--ids-file",
        required=False,
        default=None,
        type=str,
        help="Optional path to a file listing dataset record IDs to evaluate (like-for-like subset evaluation).",
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

    # 4. list-datasets (v0.2.0 preserved)
    list_ds_parser = subparsers.add_parser(
        "list-datasets",
        help="List all built-in or directory JSONL datasets in the catalog.",
    )
    list_ds_parser.add_argument("--path", type=str, default=None)
    list_ds_parser.add_argument("--category", type=str, default=None)
    list_ds_parser.add_argument("--difficulty", type=str, default=None)
    list_ds_parser.add_argument("--format", choices=["text", "json"], default="text")

    # 5. inspect-dataset (v0.2.0 preserved)
    inspect_ds_parser = subparsers.add_parser(
        "inspect-dataset",
        help="Inspect metadata, schema breakdown, and sample records of a dataset.",
    )
    inspect_ds_parser.add_argument("dataset_path", type=str)
    inspect_ds_parser.add_argument("--category", type=str, default=None)
    inspect_ds_parser.add_argument("--difficulty", type=str, default=None)
    inspect_ds_parser.add_argument("--samples", type=int, default=2)
    inspect_ds_parser.add_argument("--format", choices=["text", "json"], default="text")

    # 6. dataset-stats (v0.2.0 preserved)
    stats_ds_parser = subparsers.add_parser(
        "dataset-stats",
        help="Display detailed statistics for a dataset file or directory.",
    )
    stats_ds_parser.add_argument("dataset_path", type=str)
    stats_ds_parser.add_argument("--category", type=str, default=None)
    stats_ds_parser.add_argument("--difficulty", type=str, default=None)
    stats_ds_parser.add_argument("--format", choices=["text", "json"], default="text")

    # 7. history (v0.2.0 preserved)
    history_parser = subparsers.add_parser(
        "history",
        help="List or inspect local historical evaluation runs.",
    )
    history_subparsers = history_parser.add_subparsers(dest="history_action")
    hist_list_parser = history_subparsers.add_parser("list")
    hist_list_parser.add_argument("--store", type=str, default=None)
    hist_list_parser.add_argument("--model", type=str, default=None)
    hist_list_parser.add_argument("--dataset", type=str, default=None)
    hist_list_parser.add_argument("--format", choices=["text", "json"], default="text")

    hist_show_parser = history_subparsers.add_parser("show")
    hist_show_parser.add_argument("evaluation_id", type=str)
    hist_show_parser.add_argument("--store", type=str, default=None)
    hist_show_parser.add_argument("--format", choices=["text", "json"], default="text")

    # 8. export (v0.2.0 preserved)
    export_parser = subparsers.add_parser(
        "export",
        help="Export an existing JSON evaluation or comparison result into CSV or Markdown.",
    )
    export_parser.add_argument("--input", required=True, type=str)
    export_parser.add_argument("--format", required=True, choices=["csv", "markdown", "json"])
    export_parser.add_argument("--mode", choices=["individual", "aggregate"], default="individual")
    export_parser.add_argument("--output", required=True, type=str)

    # 9. run (v0.3.0 new: Execute a model provider on a dataset and record an experiment run)
    run_parser = subparsers.add_parser(
        "run",
        help="Execute a model provider against a dataset, evaluate metrics, and create a verifiable experiment run.",
    )
    run_parser.add_argument(
        "--config",
        required=False,
        type=str,
        help="Path to a YAML experiment configuration profile.",
    )
    run_parser.add_argument(
        "--dataset",
        required=False,
        type=str,
        help="Path to a JSONL dataset file or directory.",
    )
    run_parser.add_argument(
        "--provider",
        required=False,
        type=str,
        help="Model provider identifier (e.g. ollama, huggingface, mock).",
    )
    run_parser.add_argument(
        "--model",
        required=False,
        type=str,
        help="Model identifier (e.g. llama3.2, gpt2, mock-deterministic-v1).",
    )
    run_parser.add_argument(
        "--model-version",
        required=False,
        default=None,
        type=str,
        help="Model version or revision (default: latest or profile setting).",
    )
    run_parser.add_argument(
        "--endpoint",
        required=False,
        default=None,
        type=str,
        help="Optional provider HTTP endpoint (e.g. http://localhost:11434 for Ollama).",
    )
    run_parser.add_argument(
        "--metrics",
        nargs="+",
        default=None,
        help="Metrics to evaluate (default: exact_match accuracy keyword_coverage, or profile setting).",
    )
    run_parser.add_argument(
        "--concurrency",
        type=int,
        default=None,
        help="Number of concurrent model requests (default: 1).",
    )
    run_parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Per-request timeout in seconds (default: 30.0).",
    )
    run_parser.add_argument(
        "--max-attempts",
        type=int,
        default=None,
        help="Maximum retry attempts per item (default: 3).",
    )
    run_parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable deterministic response caching for this run.",
    )
    run_parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Directory for response cache (default: .openbench_cache).",
    )
    run_parser.add_argument(
        "--experiments-dir",
        type=str,
        default=None,
        help="Directory for storing experiment runs (default: experiments).",
    )
    run_parser.add_argument(
        "--output",
        required=False,
        default=None,
        type=str,
        help="Optional output directory to also write top-level .json/.md/.csv reports.",
    )

    # 10. runs (v0.3.0 new: list, show, verify, resume, compare)
    runs_parser = subparsers.add_parser(
        "runs",
        help="Manage, inspect, verify, resume, and compare experiment runs.",
    )
    runs_subparsers = runs_parser.add_subparsers(dest="runs_action")

    runs_list_p = runs_subparsers.add_parser("list", help="List stored experiment runs.")
    runs_list_p.add_argument("--experiments-dir", type=str, default=None)
    runs_list_p.add_argument("--status", type=str, default=None)
    runs_list_p.add_argument("--provider", type=str, default=None)
    runs_list_p.add_argument("--format", choices=["text", "json"], default="text")

    runs_show_p = runs_subparsers.add_parser("show", help="Inspect a specific experiment run.")
    runs_show_p.add_argument("run_id", type=str)
    runs_show_p.add_argument("--experiments-dir", type=str, default=None)
    runs_show_p.add_argument("--format", choices=["text", "json"], default="text")

    runs_verify_p = runs_subparsers.add_parser(
        "verify",
        help="Verify cryptographic SHA-256 artifact checksums and provenance of a run.",
    )
    runs_verify_p.add_argument("run_id", type=str)
    runs_verify_p.add_argument("--experiments-dir", type=str, default=None)
    runs_verify_p.add_argument("--format", choices=["text", "json"], default="text")

    runs_resume_p = runs_subparsers.add_parser(
        "resume",
        help="Resume an interrupted or partial provider-executed experiment run.",
    )
    runs_resume_p.add_argument("run_id", type=str)
    runs_resume_p.add_argument("--experiments-dir", type=str, default=None)
    runs_resume_p.add_argument("--cache-dir", type=str, default=None)

    runs_compare_p = runs_subparsers.add_parser(
        "compare",
        help="Compare two or more experiment runs by their RUN_IDs.",
    )
    runs_compare_p.add_argument("run_ids", nargs="+", help="Two or more experiment run IDs.")
    runs_compare_p.add_argument("--experiments-dir", type=str, default=None)
    runs_compare_p.add_argument("--output", type=str, default=None)
    runs_compare_p.add_argument("--format", choices=["text", "json", "markdown"], default="text")

    # 11. providers (v0.3.0 new: list, check)
    prov_parser = subparsers.add_parser(
        "providers",
        help="List registered model providers or check provider connectivity.",
    )
    prov_subparsers = prov_parser.add_subparsers(dest="providers_action")

    prov_list_p = prov_subparsers.add_parser("list", help="List all registered model provider adapters.")
    prov_list_p.add_argument("--format", choices=["text", "json"], default="text")

    prov_check_p = prov_subparsers.add_parser("check", help="Check connectivity/availability of a provider.")
    prov_check_p.add_argument("provider_name", type=str, help="Provider name (e.g. ollama, huggingface, mock).")
    prov_check_p.add_argument("--model", type=str, default=None)
    prov_check_p.add_argument("--endpoint", type=str, default=None)
    prov_check_p.add_argument("--format", choices=["text", "json"], default="text")

    # 12. cache (v0.3.0 new: clear, stats)
    cache_parser = subparsers.add_parser(
        "cache",
        help="Manage or clear the deterministic response cache.",
    )
    cache_subparsers = cache_parser.add_subparsers(dest="cache_action")

    cache_clear_p = cache_subparsers.add_parser("clear", help="Clear all cached responses.")
    cache_clear_p.add_argument("--cache-dir", type=str, default=None)

    cache_stats_p = cache_subparsers.add_parser("stats", help="Show cache entry count and location.")
    cache_stats_p.add_argument("--cache-dir", type=str, default=None)

    return parser


def _instantiate_provider_from_args_or_profile(
    provider_name: str,
    model_name: str,
    model_version: str,
    endpoint: str | None,
    gen_cfg: GenerationConfig,
    timeout_seconds: float,
    extra_prov_cfg: dict[str, Any],
):
    key = provider_name.strip().lower()
    if key == "ollama":
        return create_provider(
            "ollama",
            model_name=model_name,
            model_version=model_version,
            endpoint=endpoint or extra_prov_cfg.get("endpoint", DEFAULT_OLLAMA_ENDPOINT),
            generation_config=gen_cfg,
            timeout_seconds=timeout_seconds,
            allow_remote_http=bool(extra_prov_cfg.get("allow_remote_http", False)),
        )
    if key in {"huggingface", "huggingface_transformers", "hf"}:
        return create_provider(
            "huggingface_transformers",
            model_id=model_name,
            revision=model_version,
            allow_download=bool(extra_prov_cfg.get("allow_download", False)),
            device=str(extra_prov_cfg.get("device", "cpu")),
            max_new_tokens=gen_cfg.max_tokens,
            temperature=gen_cfg.temperature,
            generation_config=gen_cfg,
            timeout_seconds=timeout_seconds,
        )
    if key == "mock":
        return create_provider(
            "mock",
            model_name=model_name,
            model_version=model_version,
            use_expected_answer=bool(extra_prov_cfg.get("use_expected_answer", True)),
            simulated_latency_ms=float(extra_prov_cfg.get("simulated_latency_ms", 2.0)),
            generation_config=gen_cfg,
            timeout_seconds=timeout_seconds,
        )
    return create_provider(
        key,
        model_name=model_name,
        model_version=model_version,
        generation_config=gen_cfg,
        timeout_seconds=timeout_seconds,
    )


def _handle_validate_dataset(dataset_path: str) -> int:
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
        if getattr(args, "ids_file", None):
            config.ids_file = args.ids_file
    else:
        metric_names = list(args.metrics) if args.metrics else list(DEFAULT_METRIC_NAMES)
        config = EvaluationConfig(
            metric_names=metric_names,
            categories=list(args.category) if args.category else None,
            difficulties=list(args.difficulty) if args.difficulty else None,
            ids_file=getattr(args, "ids_file", None),
        )

    dataset = DatasetLoader.load(dataset_path)
    response_sets = ResponseLoader.load_all(args.responses)
    evaluator = Evaluator(config=config)
    results = evaluator.evaluate_multiple(dataset, response_sets)

    for res in results:
        for warning_msg in res.warnings:
            print(f"Warning [{res.model_name}]: {warning_msg}", file=sys.stderr)

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
        else:
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
    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(f"Input result file does not exist: {in_path}")

    raw_text = in_path.read_text(encoding="utf-8")
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise OpenBenchError(f"Malformed JSON in {in_path}: {exc.msg}") from exc

    out_path = Path(args.output)
    if "models" in payload and "highest_by_metric" in payload:
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


def _handle_run(args: argparse.Namespace) -> int:
    """Execute `openbench run` via ExecutionEngine and ExperimentManager."""
    profile = ProfileLoader.load_file(args.config) if args.config else None

    dataset_path = args.dataset or (profile.dataset_path if profile else None)
    if not dataset_path:
        print(
            "Error: Specify --dataset or provide a --config profile with dataset.path.",
            file=sys.stderr,
        )
        return 2

    prov_cfg = dict(profile.provider_config) if profile else {}
    provider_name = args.provider or prov_cfg.get("name")
    if not provider_name:
        print(
            "Error: Specify --provider (e.g. ollama, huggingface, mock) or set provider.name in --config.",
            file=sys.stderr,
        )
        return 2

    model_name = args.model or prov_cfg.get("model")
    if not model_name:
        print(
            "Error: Specify --model or set provider.model in --config.",
            file=sys.stderr,
        )
        return 2

    model_version = (
        args.model_version
        or prov_cfg.get("model_version")
        or prov_cfg.get("revision")
        or "1.0"
    )
    gen_cfg = profile.generation_config if profile else GenerationConfig()

    base_exec = profile.execution_config if profile else ExecutionConfig()
    timeout_sec = args.timeout if args.timeout is not None else base_exec.timeout_seconds
    concurrency = args.concurrency if args.concurrency is not None else base_exec.concurrency
    max_attempts = (
        args.max_attempts
        if args.max_attempts is not None
        else base_exec.retry_policy.max_attempts
    )
    use_cache = False if args.no_cache else base_exec.use_cache

    retry_policy = RetryPolicy(
        max_attempts=max_attempts,
        initial_backoff_seconds=base_exec.retry_policy.initial_backoff_seconds,
        backoff_multiplier=base_exec.retry_policy.backoff_multiplier,
        max_backoff_seconds=base_exec.retry_policy.max_backoff_seconds,
        retry_on_timeouts=base_exec.retry_policy.retry_on_timeouts,
    )
    exec_cfg = ExecutionConfig(
        timeout_seconds=timeout_sec,
        concurrency=concurrency,
        retry_policy=retry_policy,
        use_cache=use_cache,
        estimate_missing_tokens=base_exec.estimate_missing_tokens,
        privacy_mode=base_exec.privacy_mode,
        pricing=base_exec.pricing,
    )

    provider = _instantiate_provider_from_args_or_profile(
        provider_name=str(provider_name),
        model_name=str(model_name),
        model_version=str(model_version),
        endpoint=args.endpoint,
        gen_cfg=gen_cfg,
        timeout_seconds=timeout_sec,
        extra_prov_cfg=prov_cfg,
    )

    if profile is not None:
        eval_cfg = profile.to_evaluation_config(override_metrics=args.metrics)
    else:
        eval_cfg = EvaluationConfig(
            metric_names=list(args.metrics) if args.metrics else list(DEFAULT_METRIC_NAMES)
        )

    dataset = DatasetLoader.load(dataset_path)
    cache_dir = args.cache_dir or (profile.cache_dir if profile else None)
    cache = ResponseCache(cache_dir=cache_dir, enabled=use_cache)

    exp_dir = args.experiments_dir or (profile.experiments_dir if profile else None)
    manager = ExperimentManager(experiments_root=exp_dir)

    repeat_count = profile.repeat_count if profile else 1
    run_records = []
    eval_results = []

    for _ in range(repeat_count):
        rec, eval_res, perf_rep = manager.execute_and_record(
            dataset=dataset,
            provider=provider,
            evaluation_config=eval_cfg,
            execution_config=exec_cfg,
            cache=cache,
        )
        run_records.append(rec)
        eval_results.append(eval_res)
        print(
            f"Run '{rec.run_id}' [{rec.status}]: {rec.completed_items}/{rec.total_items} completed "
            f"(provider={rec.model_provider}, model={rec.model_name}, scores={rec.aggregate_scores}) -> {rec.run_dir}"
        )

    if repeat_count > 1:
        variance_summary = BenchmarkAnalytics.analyze_repeated_runs(eval_results)
        var_file = Path(run_records[-1].run_dir) / "repeated_runs_analysis.json"
        var_file.write_text(json.dumps(variance_summary, indent=2) + "\n", encoding="utf-8")
        print(f"Repeated-run variance analysis (n={repeat_count}) saved to: {var_file}")

    if args.output:
        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)
        last_eval = eval_results[-1]
        slug = _slugify(last_eval.model_name)
        ReportGenerator.save_json(last_eval, out_dir / f"{slug}.json")
        ReportGenerator.save_markdown(last_eval, out_dir / f"{slug}.md")

    return 0 if all(r.status == "completed" for r in run_records) else 1


def _handle_runs(args: argparse.Namespace) -> int:
    """Execute `openbench runs list|show|verify|resume|compare`."""
    if not args.runs_action:
        print("Error: Specify a runs subcommand (list, show, verify, resume, compare).", file=sys.stderr)
        return 2

    manager = ExperimentManager(experiments_root=getattr(args, "experiments_dir", None))

    if args.runs_action == "list":
        runs = manager.list_runs(status=args.status, provider=args.provider)
        if args.format == "json":
            print(json.dumps([r.to_dict() for r in runs], indent=2))
            return 0
        if not runs:
            print("No experiment runs found.")
            return 0
        print(f"Experiment Runs ({len(runs)}):")
        for r in runs:
            mock_tag = " [MOCK]" if r.is_mock_provider else ""
            print(
                f"  - {r.run_id} | {r.status} | origin={r.execution_origin} | "
                f"{r.model_provider}:{r.model_name}{mock_tag} | dataset={r.dataset_identifier} "
                f"({r.completed_items}/{r.total_items}) | scores={r.aggregate_scores}"
            )
        return 0

    if args.runs_action == "show":
        try:
            details = manager.inspect_run(args.run_id)
        except KeyError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        if args.format == "json":
            print(json.dumps(details, indent=2))
            return 0
        exp = details["experiment"]
        integ = details["integrity"]
        print(f"Experiment Run: {exp['run_id']} ({exp['status']})")
        print(f"  Origin        : {exp['execution_origin']}")
        print(f"  Provider      : {exp['model_provider']} (is_mock={exp['is_mock_provider']})")
        print(f"  Model         : {exp['model_name']} (v{exp['model_version']})")
        print(f"  Dataset       : {exp['dataset_identifier']} (checksum={exp['dataset_checksum']})")
        print(f"  Items         : {exp['completed_items']} / {exp['total_items']} completed ({exp['failed_items']} failed)")
        print(f"  Scores        : {exp['aggregate_scores']}")
        print(f"  Integrity     : {'VALID' if integ['valid'] else 'INVALID'}")
        print(f"  Run Directory : {exp['run_dir']}")
        return 0

    if args.runs_action == "verify":
        try:
            check = manager.verify_run(args.run_id)
        except KeyError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        if args.format == "json":
            print(json.dumps(check.to_dict(), indent=2))
            return 0 if check.valid else 1
        status_label = "VALID" if check.valid else "INVALID"
        print(
            f"Run Integrity Verification [{check.run_id}]: {status_label} "
            f"(origin={check.execution_origin}, schema={check.schema_version})"
        )
        if check.missing_artifacts:
            print(f"  Missing Artifacts      : {check.missing_artifacts}")
        if check.changed_artifacts:
            print(f"  Changed Artifacts      : {check.changed_artifacts}")
        if check.inconsistent_metadata:
            print(f"  Inconsistent Metadata  : {check.inconsistent_metadata}")
        for w in check.warnings:
            print(f"  Note: {w}")
        return 0 if check.valid else 1

    if args.runs_action == "resume":
        try:
            info = manager.inspect_run(args.run_id)["experiment"]
        except KeyError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1

        gen_cfg = GenerationConfig(
            temperature=float(info["generation_configuration"].get("temperature", 0.0)),
            max_tokens=int(info["generation_configuration"].get("max_tokens", 256)),
        )
        provider = _instantiate_provider_from_args_or_profile(
            provider_name=info["model_provider"],
            model_name=info["model_name"],
            model_version=info["model_version"],
            endpoint=info["generation_configuration"].get("endpoint"),
            gen_cfg=gen_cfg,
            timeout_seconds=float(info["execution_configuration"].get("timeout_seconds", 30.0)),
            extra_prov_cfg=info["generation_configuration"],
        )
        cache = ResponseCache(cache_dir=args.cache_dir, enabled=True)
        rec, _, _ = manager.resume_run(args.run_id, provider=provider, cache=cache)
        print(
            f"Resumed run '{rec.run_id}' -> status={rec.status} "
            f"({rec.completed_items}/{rec.total_items} completed, scores={rec.aggregate_scores})"
        )
        return 0 if rec.status == "completed" else 1

    if args.runs_action == "compare":
        if len(args.run_ids) < 2:
            print("Error: Provide at least two RUN_IDs to compare.", file=sys.stderr)
            return 2
        try:
            comparison = manager.compare_runs(args.run_ids)
        except KeyError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1

        if args.output:
            out_p = Path(args.output)
            if out_p.suffix.lower() == ".json":
                ReportGenerator.save_json(comparison, out_p)
            elif out_p.suffix.lower() == ".csv":
                ReportGenerator.save_csv(comparison, out_p)
            else:
                ReportGenerator.save_markdown(comparison, out_p)
            print(f"Saved run comparison report to: {out_p}")
            return 0

        if args.format == "json":
            print(ReportGenerator.to_json(comparison))
        else:
            print(ReportGenerator.comparison_to_markdown(comparison))
        return 0

    return 2


def _handle_providers(args: argparse.Namespace) -> int:
    """Execute `openbench providers list|check`."""
    if not args.providers_action:
        print("Error: Specify a providers subcommand ('list' or 'check').", file=sys.stderr)
        return 2

    if args.providers_action == "list":
        providers = list_registered_providers()
        if args.format == "json":
            print(json.dumps(providers, indent=2))
            return 0
        print(f"Registered Model Providers ({len(providers)}):")
        for p in providers:
            flags = []
            if p["is_mock"]:
                flags.append("mock")
            if p["is_experimental"]:
                flags.append("experimental")
            flag_str = f" [{', '.join(flags)}]" if flags else ""
            print(f"  - {p['provider_name']}{flag_str}: {p['description']}")
        return 0

    if args.providers_action == "check":
        p_name = args.provider_name.strip().lower()
        default_model = args.model or ("llama3.2" if p_name == "ollama" else "test-model")
        provider = _instantiate_provider_from_args_or_profile(
            provider_name=p_name,
            model_name=default_model,
            model_version="latest",
            endpoint=args.endpoint,
            gen_cfg=GenerationConfig(),
            timeout_seconds=5.0,
            extra_prov_cfg={},
        )
        health = provider.check_connection()
        if args.format == "json":
            print(json.dumps(health.to_dict(), indent=2))
            return 0 if health.available else 1
        state = "AVAILABLE" if health.available else "UNAVAILABLE"
        print(f"Provider Check [{health.provider_name}]: {state} — {health.message}")
        return 0 if health.available else 1

    return 2


def _handle_cache(args: argparse.Namespace) -> int:
    """Execute `openbench cache clear|stats`."""
    if not args.cache_action:
        print("Error: Specify a cache subcommand ('clear' or 'stats').", file=sys.stderr)
        return 2

    cache = ResponseCache(cache_dir=getattr(args, "cache_dir", None), enabled=True)
    if args.cache_action == "clear":
        cleared = cache.clear()
        print(f"Cleared {cleared} cached response(s) from {cache.cache_dir}")
        return 0
    if args.cache_action == "stats":
        print(json.dumps(cache.stats(), indent=2))
        return 0
    return 2


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
        if args.command == "run":
            return _handle_run(args)
        if args.command == "runs":
            return _handle_runs(args)
        if args.command == "providers":
            return _handle_providers(args)
        if args.command == "cache":
            return _handle_cache(args)
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
