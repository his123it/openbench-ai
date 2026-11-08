#!/usr/bin/env python3
"""Example: Evaluate and compare two language models on the reasoning dataset."""

from pathlib import Path

from openbench.comparison import ModelComparator
from openbench.dataset import DatasetLoader
from openbench.evaluator import (
    EvaluationConfig,
    Evaluator,
    ModelResponseSet,
    ResponseItem,
    ResponseLoader,
)
from openbench.reporting import ReportGenerator


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    dataset_path = repo_root / "datasets" / "reasoning" / "sample.jsonl"
    responses_a_path = repo_root / "examples" / "sample_responses.json"
    output_dir = repo_root / "results"

    dataset = DatasetLoader.load_file(dataset_path)
    model_a = ResponseLoader.load_file(responses_a_path)

    # Construct a second model response set representing a more conversational, slightly less accurate model
    model_b = ModelResponseSet(
        model_name="BetaChat-3B",
        model_version="0.9",
        responses=[
            ResponseItem(id="reasoning_001", response="The answer is 180."),
            ResponseItem(id="reasoning_002", response="Each box has 12 cupcakes."),
            ResponseItem(id="reasoning_003", response="37"),
            ResponseItem(id="reasoning_004", response="Yes, Device X-9 is thermally calibrated."),
            ResponseItem(id="reasoning_005", response="260"),
            ResponseItem(id="reasoning_006", response="125"),  # incorrect
            ResponseItem(id="reasoning_007", response="R2"),
            ResponseItem(id="reasoning_008", response="The value of x is 7."),
            ResponseItem(id="reasoning_009", response="30"),  # incorrect
            ResponseItem(id="reasoning_010", response="4"),
            ResponseItem(id="reasoning_011", response="84"),
            # reasoning_012 intentionally omitted to demonstrate missing response tracking
        ],
    )

    evaluator = Evaluator(
        config=EvaluationConfig(
            metric_names=["exact_match", "accuracy", "keyword_coverage"]
        )
    )
    eval_results = evaluator.evaluate_multiple(dataset, [model_a, model_b])

    ReportGenerator.save_json(eval_results[0], output_dir / "model_a.json")
    ReportGenerator.save_json(eval_results[1], output_dir / "model_b.json")

    comparison = ModelComparator.compare(eval_results)
    cmp_md_path = ReportGenerator.save_markdown(comparison, output_dir / "comparison.md")
    cmp_json_path = ReportGenerator.save_json(comparison, output_dir / "comparison.json")

    print("Model Comparison Summary:")
    for entry in comparison.models:
        print(
            f"  * {entry.model_identifier}: "
            f"exact_match={entry.exact_match_score:.2%}, "
            f"accuracy={entry.accuracy_score:.2%}, "
            f"keyword_coverage={entry.keyword_coverage_score:.2%} "
            f"(missing={entry.missing_responses_count})"
        )
    print(f"Saved Markdown comparison report to: {cmp_md_path}")
    print(f"Saved JSON comparison report to: {cmp_json_path}")


if __name__ == "__main__":
    main()
