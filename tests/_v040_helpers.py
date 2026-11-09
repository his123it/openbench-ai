"""Shared fixtures for v0.4.0 tests: small datasets, response sets and evaluation files."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from openbench import DatasetLoader, EvaluationConfig, Evaluator, ModelResponseSet, ResponseItem
from openbench.reporting import ReportGenerator


def write_dataset(path: Path, n: int = 8, *, version: str | None = None, tweak: dict[str, str] | None = None) -> Path:
    """Write an n-record JSONL dataset (two categories, three difficulties)."""
    lines = []
    for i in range(1, n + 1):
        rid = f"item_{i:03d}"
        meta: dict[str, Any] = {"difficulty": ["easy", "medium", "hard"][i % 3], "keywords": [f"ans{i}"]}
        if version:
            meta["version"] = version
        rec = {
            "id": rid,
            "category": "alpha" if i % 2 else "beta",
            "prompt": f"Question number {i}?",
            "expected_answer": (tweak or {}).get(rid, f"ans{i}"),
            "metadata": meta,
        }
        lines.append(json.dumps(rec))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def evaluate(
    dataset_path: Path,
    answers: dict[str, str | None],
    model: str,
    out: Path,
    metrics: tuple[str, ...] = ("exact_match", "keyword_coverage"),
    *,
    shuffle: bool = False,
    config: EvaluationConfig | None = None,
) -> Path:
    """Evaluate ``answers`` (None = missing) and save the evaluation JSON to ``out``."""
    ds = DatasetLoader.load(str(dataset_path))
    items = [ResponseItem(id=k, response=v) for k, v in answers.items() if v is not None]
    if shuffle:
        items = list(reversed(items))
    rs = ModelResponseSet(model_name=model, model_version="1.0", responses=items, model_provider="local_lookup")
    ev = Evaluator(config=config or EvaluationConfig(metric_names=list(metrics))).evaluate(ds, rs)
    payload = ev.to_dict()
    if shuffle:
        payload["individual_results"] = list(reversed(payload["individual_results"]))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out


def correct_answers(n: int, wrong: tuple[int, ...] = (), missing: tuple[int, ...] = (), empty: tuple[int, ...] = ()) -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for i in range(1, n + 1):
        rid = f"item_{i:03d}"
        if i in missing:
            out[rid] = None
        elif i in empty:
            out[rid] = "   "
        elif i in wrong:
            out[rid] = "wrong"
        else:
            out[rid] = f"ans{i}"
    return out


def snapshot(directory: Path) -> dict[str, tuple[str, int, int]]:
    out = {}
    for path in sorted(directory.rglob("*")):
        if path.is_file():
            st = path.stat()
            out[str(path.relative_to(directory))] = (
                hashlib.sha256(path.read_bytes()).hexdigest(), st.st_size, st.st_mtime_ns,
            )
    return out


def save_report(result, path: Path) -> Path:
    ReportGenerator.save_json(result, path)
    return path


def strip_volatile(payload: Any) -> Any:
    """Remove fields that legitimately differ between equivalent analyses."""
    volatile = {"created_at", "source_paths", "input_sha256", "evaluation_ids", "input_digest_sha256", "analysis_id"}
    if isinstance(payload, dict):
        return {k: strip_volatile(v) for k, v in payload.items() if k not in volatile}
    if isinstance(payload, list):
        return [strip_volatile(v) for v in payload]
    return payload
