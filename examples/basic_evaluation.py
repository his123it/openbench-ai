#!/usr/bin/env python3
"""Example: Run a basic single-model evaluation using the OpenBench AI Python API."""

from pathlib import Path

from openbench.dataset import DatasetLoader
from openbench.evaluator import EvaluationConfig, Evaluator, ResponseLoader
from openbench.reporting import ReportGenerator


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    dataset_path = repo_root / "datasets" / "reasoning" / "sample.jsonl"
    responses_path = repo_root / "examples" / "sample_responses.json"
    output_dir = repo_root / "results"

    dataset = DatasetLoader.load_file(dataset_path)
    response_set = ResponseLoader.load_file(responses_path)

    config = EvaluationConfig(
        metric_names=["exact_match", "accuracy", "keyword_coverage"],
    )
    evaluator = Evaluator(config=config)
    result = evaluator.evaluate(dataset, response_set)

    json_path = ReportGenerator.save_json(result, output_dir / "alphabaseline_7b.json")
    md_path = ReportGenerator.save_markdown(result, output_dir / "alphabaseline_7b.md")

    print(f"Evaluated model: {result.model_name} (v{result.model_version})")
    print(f"Dataset: {dataset.name} ({len(dataset)} items)")
    for metric_name, agg in result.aggregate_results.items():
        print(f"  - {metric_name}: {agg.score:.4f} ({agg.score * 100:.2f}%)")
    print(f"Saved JSON report to: {json_path}")
    print(f"Saved Markdown report to: {md_path}")


if __name__ == "__main__":
    main()
