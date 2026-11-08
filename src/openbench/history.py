"""Local evaluation result history store for OpenBench AI."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openbench.evaluator import EvaluationResult
from openbench.exceptions import OpenBenchError


@dataclass(frozen=True)
class HistoryEntry:
    """Lightweight summary record stored in the local JSONL evaluation history."""

    evaluation_id: str
    timestamp: str
    model_name: str
    model_version: str
    dataset_name: str
    dataset_version: str
    dataset_checksum: str
    metric_configuration: dict[str, Any]
    openbench_version: str
    result_summary: dict[str, float]
    total_items: int
    evaluated_items: int
    missing_responses_count: int
    report_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "evaluation_id": self.evaluation_id,
            "timestamp": self.timestamp,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "dataset_name": self.dataset_name,
            "dataset_version": self.dataset_version,
            "dataset_checksum": self.dataset_checksum,
            "metric_configuration": dict(self.metric_configuration),
            "openbench_version": self.openbench_version,
            "result_summary": dict(self.result_summary),
            "total_items": self.total_items,
            "evaluated_items": self.evaluated_items,
            "missing_responses_count": self.missing_responses_count,
            "report_path": self.report_path,
        }


class ResultHistoryStore:
    """File-backed JSONL evaluation history store requiring zero database servers."""

    def __init__(self, history_path: str | Path | None = None) -> None:
        if history_path is not None:
            self.path = Path(history_path)
        else:
            self.path = Path.cwd() / "results" / "history.jsonl"

    def record(
        self,
        result: EvaluationResult,
        report_path: str | Path | None = None,
    ) -> HistoryEntry:
        """Append an EvaluationResult summary to the local JSONL history store."""
        entry = HistoryEntry(
            evaluation_id=result.evaluation_id,
            timestamp=result.evaluation_timestamp,
            model_name=result.model_name,
            model_version=result.model_version,
            dataset_name=str(result.dataset_summary.get("name", "unknown")),
            dataset_version=str(result.dataset_summary.get("version", "0.1.0")),
            dataset_checksum=str(result.dataset_summary.get("checksum", "unknown")),
            metric_configuration=dict(result.config),
            openbench_version=result.project_version,
            result_summary={
                k: round(v.score, 6) for k, v in result.aggregate_results.items()
            },
            total_items=result.total_dataset_items,
            evaluated_items=result.evaluated_items_count,
            missing_responses_count=result.missing_responses_count,
            report_path=str(report_path) if report_path else None,
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")
        return entry

    def list_entries(
        self,
        model_name: str | None = None,
        dataset_name: str | None = None,
    ) -> list[HistoryEntry]:
        """Return all recorded HistoryEntry items, optionally filtered by model or dataset."""
        if not self.path.exists():
            return []

        entries: list[HistoryEntry] = []
        for idx, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                data = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise OpenBenchError(
                    f"Corrupted JSONL history entry at {self.path}:{idx}: {exc.msg}"
                ) from exc

            entry = HistoryEntry(
                evaluation_id=str(data["evaluation_id"]),
                timestamp=str(data["timestamp"]),
                model_name=str(data["model_name"]),
                model_version=str(data["model_version"]),
                dataset_name=str(data["dataset_name"]),
                dataset_version=str(data.get("dataset_version", "0.1.0")),
                dataset_checksum=str(data.get("dataset_checksum", "unknown")),
                metric_configuration=dict(data.get("metric_configuration", {})),
                openbench_version=str(data.get("openbench_version", "0.2.0")),
                result_summary=dict(data.get("result_summary", {})),
                total_items=int(data.get("total_items", 0)),
                evaluated_items=int(data.get("evaluated_items", 0)),
                missing_responses_count=int(data.get("missing_responses_count", 0)),
                report_path=data.get("report_path"),
            )
            if model_name and entry.model_name.lower() != model_name.lower():
                continue
            if dataset_name and entry.dataset_name.lower() != dataset_name.lower():
                continue
            entries.append(entry)
        return entries

    def get_entry(self, evaluation_id: str) -> HistoryEntry:
        """Look up a specific HistoryEntry by its `evaluation_id`."""
        for entry in self.list_entries():
            if entry.evaluation_id == evaluation_id:
                return entry
        raise KeyError(f"Evaluation ID '{evaluation_id}' not found in history store {self.path}")
