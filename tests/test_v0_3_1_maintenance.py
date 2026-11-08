"""Comprehensive regression test suite for OpenBench AI v0.3.1 maintenance & security fixes."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import pytest

from openbench import (
    Dataset,
    DatasetLoader,
    DatasetRecord,
    DatasetValidationError,
    EmptyDatasetError,
    EvaluationConfig,
    Evaluator,
    ExperimentManager,
    LexicalSimilarityMetric,
    ModelComparator,
    ModelResponseSet,
    ReportGenerator,
    ResponseCache,
    ResponseItem,
    escape_markdown_cell,
    load_ids_file,
    redact_sensitive_data,
)
from openbench.cli import main
from openbench.metrics import (
    CallableEmbeddingProvider,
    LocalNgramEmbeddingProvider,
    MetricError,
    SemanticSimilarityMetric,
    StructuredOutputMetric,
)


def _five_item_dataset(tmp_path: Path) -> Dataset:
    ds_file = tmp_path / "five.jsonl"
    items = [
        {"id": "item_001", "category": "reasoning", "prompt": "1+1?", "expected_answer": "2"},
        {"id": "item_002", "category": "reasoning", "prompt": "2+2?", "expected_answer": "4"},
        {"id": "item_003", "category": "reasoning", "prompt": "3+3?", "expected_answer": "6"},
        {"id": "item_004", "category": "reasoning", "prompt": "4+4?", "expected_answer": "8"},
        {"id": "item_005", "category": "reasoning", "prompt": "5+5?", "expected_answer": "10"},
    ]
    ds_file.write_text("\n".join(json.dumps(x) for x in items) + "\n", encoding="utf-8")
    return DatasetLoader.load_file(ds_file)


# ---------------------------------------------------------------------------
# 3.1 Like-for-like Evaluation using ID Filtering
# ---------------------------------------------------------------------------

def test_id_filtering_full_and_partial_dataset_evaluation(tmp_path: Path) -> None:
    """Evaluating with --ids-file evaluates only selected IDs and never counts excluded IDs as missing."""
    ds = _five_item_dataset(tmp_path)
    ids_file = tmp_path / "legacy_ids.txt"
    ids_file.write_text("# Legacy subset\nitem_001\nitem_003\nitem_005\n", encoding="utf-8")

    # Model only answered item_001, item_003, item_005 (like a v0.1.0 response file on an expanded dataset)
    resp_set = ModelResponseSet(
        model_name="LegacySubsetModel",
        model_version="1.0",
        responses=[
            ResponseItem(id="item_001", response="2"),
            ResponseItem(id="item_003", response="6"),
            ResponseItem(id="item_005", response="10"),
        ],
    )

    # Without ID filter: 3/5 = 0.60 exact_match (2 missing)
    res_unfiltered = Evaluator().evaluate(ds, resp_set)
    assert res_unfiltered.total_dataset_items == 5
    assert res_unfiltered.missing_responses_count == 2
    assert pytest.approx(res_unfiltered.aggregate_results["exact_match"].score) == 0.6

    # With ID filter: 3/3 = 1.00 exact_match, 0 missing, 2 excluded
    cfg_filtered = EvaluationConfig(ids_file=str(ids_file))
    res_filtered = Evaluator(config=cfg_filtered).evaluate(ds, resp_set)
    assert res_filtered.total_dataset_items == 3
    assert res_filtered.evaluated_items_count == 3
    assert res_filtered.missing_responses_count == 0
    assert res_filtered.excluded_ids == ["item_002", "item_004"]
    assert res_filtered.aggregate_results["exact_match"].score == 1.0
    assert res_filtered.id_filter_summary is not None
    assert res_filtered.id_filter_summary["selected_ids"] == ["item_001", "item_003", "item_005"]
    assert len(res_filtered.id_filter_summary["selected_ids_checksum"]) == 64

    # Verify JSON and Markdown reports include ID filter metadata
    json_report = json.loads(ReportGenerator.to_json(res_filtered))
    assert "id_filter" in json_report
    assert json_report["reproducibility_card"]["selected_ids_count"] == 3
    assert json_report["reproducibility_card"]["excluded_ids_count"] == 2
    md_report = ReportGenerator.to_markdown(res_filtered)
    assert "ID Filter Applied" in md_report


def test_id_filtering_duplicates_unknowns_missing_and_invalid_files(tmp_path: Path) -> None:
    """load_ids_file and Evaluator handle duplicate IDs, unknown IDs, missing selected responses, and invalid files."""
    ds = _five_item_dataset(tmp_path)

    # 1. Empty ID file raises EmptyDatasetError
    empty_ids = tmp_path / "empty_ids.txt"
    empty_ids.write_text("   \n# comment only\n", encoding="utf-8")
    with pytest.raises(EmptyDatasetError):
        load_ids_file(empty_ids)

    # 2. Invalid ID syntax raises DatasetValidationError
    bad_ids = tmp_path / "bad_ids.txt"
    bad_ids.write_text("item_001\ninvalid id with spaces!\n", encoding="utf-8")
    with pytest.raises(DatasetValidationError, match="Invalid dataset ID"):
        load_ids_file(bad_ids)

    # 3. Duplicate IDs, unknown IDs, and missing response among selected IDs, plus stable ordering
    mixed_ids = tmp_path / "mixed_ids.txt"
    mixed_ids.write_text("item_004\nitem_002\nitem_004\nunknown_id_99\n", encoding="utf-8")
    spec = load_ids_file(mixed_ids)
    assert spec.selected_ids == ["item_004", "item_002", "unknown_id_99"]
    assert spec.duplicate_ids_removed == ["item_004"]

    # Model provides item_002, misses item_004, and also provides item_001 (which is excluded)
    resp_set = ModelResponseSet(
        model_name="PartialSelectedModel",
        model_version="1.0",
        responses=[
            ResponseItem(id="item_001", response="2"),
            ResponseItem(id="item_002", response="4"),
        ],
    )
    res = Evaluator(config=EvaluationConfig(ids_file=str(mixed_ids))).evaluate(ds, resp_set)
    # Deterministic dataset ordering preserved: item_002 then item_004
    assert [it.id for it in res.individual_results] == ["item_002", "item_004"]
    assert res.missing_response_ids == ["item_004"]
    assert res.unknown_response_ids == []  # item_001 is an excluded dataset ID, not an unknown ID
    assert res.id_filter_summary is not None
    assert res.id_filter_summary["unknown_filter_ids"] == ["unknown_id_99"]
    assert res.id_filter_summary["duplicate_ids_removed"] == ["item_004"]


# ---------------------------------------------------------------------------
# 3.2 Stronger Value-Level Secret Redaction & Runtime Preservation
# ---------------------------------------------------------------------------

def test_value_level_secret_redaction_preserves_runtime_config(tmp_path: Path) -> None:
    """redact_sensitive_data scrubs secrets inside harmless keys/strings without mutating original runtime dict."""
    runtime_config = {
        "endpoint": "https://example.com/api?token=abc123&mode=fast&api_key=xyz987",
        "headers_note": "Authorization: Bearer secret_jwt_token_999",
        "conn_str": "host=db.local password=my_super_secret client_secret=cs_8888",
        "harmless_id": "reasoning_001_token_count",
        "normal_text": "The model evaluated 20 items successfully.",
    }
    original_endpoint = runtime_config["endpoint"]

    redacted = redact_sensitive_data(runtime_config)

    # Original runtime dictionary MUST remain untouched so live provider calls receive real credentials
    assert runtime_config["endpoint"] == original_endpoint
    assert "token=abc123" in runtime_config["endpoint"]

    # Redacted copy must scrub all embedded secrets while preserving harmless text & query params
    assert "abc123" not in redacted["endpoint"]
    assert "xyz987" not in redacted["endpoint"]
    assert "token=[REDACTED]" in redacted["endpoint"]
    assert "api_key=[REDACTED]" in redacted["endpoint"]
    assert "mode=fast" in redacted["endpoint"]
    assert "secret_jwt_token_999" not in redacted["headers_note"]
    assert "my_super_secret" not in redacted["conn_str"]
    assert "cs_8888" not in redacted["conn_str"]
    assert redacted["harmless_id"] == "reasoning_001_token_count"
    assert redacted["normal_text"] == "The model evaluated 20 items successfully."

    # Also verify cache never persists raw credentials and cache key is stable across credential rotation
    cache = ResponseCache(cache_dir=tmp_path / "sec_cache", enabled=True)
    key_a, _, _ = cache.compute_cache_key(
        provider_name="ollama",
        model_name="llama3.2",
        model_version="latest",
        dataset_name="ds",
        dataset_version="1.0",
        record_id="r1",
        prompt="Hello",
        generation_config={"endpoint": "https://api.example.com/v1?token=secret_A"},
    )
    key_b, _, _ = cache.compute_cache_key(
        provider_name="ollama",
        model_name="llama3.2",
        model_version="latest",
        dataset_name="ds",
        dataset_version="1.0",
        record_id="r1",
        prompt="Hello",
        generation_config={"endpoint": "https://api.example.com/v1?token=secret_B"},
    )
    assert key_a == key_b


# ---------------------------------------------------------------------------
# 3.3 Optional Fenced JSON Support in StructuredOutputMetric
# ---------------------------------------------------------------------------

def test_structured_output_optional_markdown_fence_stripping() -> None:
    """StructuredOutputMetric keeps strip_markdown_fence=False by default and strips single valid fences when enabled."""
    rec = DatasetRecord(
        id="struct_fence_01",
        category="structured_output",
        prompt="Return status JSON.",
        expected_answer='{"status": "success", "count": 3}',
        metadata={
            "required_keys": ["status", "count"],
            "field_types": {"status": "str", "count": "int"},
        },
    )
    fenced_json = '```json\n{"status": "success", "count": 3}\n```'
    plain_fenced = '```\n{"status": "success", "count": 3}\n```'
    surrounding_prose = 'Here is your JSON:\n```json\n{"status": "success", "count": 3}\n```'
    multiple_fences = '```json\n{"status": "success", "count": 3}\n```\n```json\n{"status": "other", "count": 1}\n```'
    malformed_fenced = '```json\n{"status": "success", "count": }\n```'

    # 1. Default (strip_markdown_fence=False): fenced input fails JSON parsing (v0.2.0 behavior preserved)
    default_metric = StructuredOutputMetric()
    res_default = default_metric.compute_item(rec, fenced_json)
    assert res_default.passed is False
    assert res_default.details["valid_json"] is False

    # 2. Enabled (strip_markdown_fence=True): single fenced JSON and plain fence succeed
    fence_metric = StructuredOutputMetric(strip_markdown_fence=True)
    res_fenced = fence_metric.compute_item(rec, fenced_json)
    assert res_fenced.passed is True
    assert res_fenced.score == 1.0
    assert res_fenced.details["fence_stripped"] is True
    assert res_fenced.details["value_match"] is True

    res_plain_fence = fence_metric.compute_item(rec, plain_fenced)
    assert res_plain_fence.passed is True
    assert res_plain_fence.details["fence_stripped"] is True

    # 3. Unfenced valid JSON still works
    res_unfenced = fence_metric.compute_item(rec, '{"status": "success", "count": 3}')
    assert res_unfenced.passed is True
    assert res_unfenced.details["fence_stripped"] is False

    # 4. Surrounding prose is NOT stripped
    res_prose = fence_metric.compute_item(rec, surrounding_prose)
    assert res_prose.passed is False
    assert res_prose.details["valid_json"] is False
    assert res_prose.details["fence_stripped"] is False

    # 5. Multiple fences are NOT stripped
    res_multi = fence_metric.compute_item(rec, multiple_fences)
    assert res_multi.passed is False
    assert res_multi.details["fence_stripped"] is False

    # 6. Malformed JSON inside fence is reported as invalid JSON
    res_malformed = fence_metric.compute_item(rec, malformed_fenced)
    assert res_malformed.passed is False
    assert res_malformed.details["valid_json"] is False
    assert res_malformed.details["fence_stripped"] is True


# ---------------------------------------------------------------------------
# 3.4 Semantic vs Lexical Similarity Naming and Correctness
# ---------------------------------------------------------------------------

def test_semantic_similarity_labels_default_provider_as_lexical_similarity() -> None:
    """Default LocalNgramEmbeddingProvider is labeled lexical_similarity and requires embedding provider for semantic."""
    rec = DatasetRecord(
        id="sim_01",
        category="reasoning",
        prompt="P",
        expected_answer="Use Argon2id password hashing",
    )

    # Default provider on SemanticSimilarityMetric
    default_metric = SemanticSimilarityMetric()
    assert default_metric.provider.is_semantic_embedding is False
    assert default_metric.similarity_type == "lexical_similarity"

    item_res = default_metric.compute_item(rec, "Use Argon2id password hashing")
    assert item_res.details["similarity_type"] == "lexical_similarity"
    assert item_res.details["is_semantic_embedding"] is False

    agg_res = default_metric.aggregate([item_res], total_dataset_items=1)
    assert agg_res.details["similarity_type"] == "lexical_similarity"
    assert "legacy alias" in agg_res.details["legacy_alias_note"]

    # Explicit LexicalSimilarityMetric
    lex_metric = LexicalSimilarityMetric()
    assert lex_metric.name == "lexical_similarity"
    assert lex_metric.similarity_type == "lexical_similarity"

    # Requiring a genuine semantic provider raises MetricError with LocalNgramEmbeddingProvider
    with pytest.raises(MetricError, match="performs lexical comparison"):
        SemanticSimilarityMetric(
            provider=LocalNgramEmbeddingProvider(),
            require_semantic_provider=True,
        )

    # Genuine embedding provider (e.g., CallableEmbeddingProvider with is_semantic_embedding=True) labels as semantic_similarity
    emb_provider = CallableEmbeddingProvider(
        embed_fn=lambda t: [0.5, 0.5],
        model_id="neural-embed-v1",
        is_semantic_embedding=True,
    )
    sem_metric = SemanticSimilarityMetric(provider=emb_provider, require_semantic_provider=True)
    assert sem_metric.similarity_type == "semantic_similarity"
    sem_item = sem_metric.compute_item(rec, "test")
    assert sem_item.details["similarity_type"] == "semantic_similarity"


# ---------------------------------------------------------------------------
# 3.5 CLI Evaluation Warnings Printed to stderr
# ---------------------------------------------------------------------------

def test_cli_evaluate_prints_warnings_to_stderr_and_preserves_stdout(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`openbench evaluate` emits missing/unknown/excluded warnings to stderr while keeping stdout clean."""
    ds = _five_item_dataset(tmp_path)
    ds_path = Path(ds.source_path)

    # Response file misses item_002 and includes an unknown_999 ID
    resp_file = tmp_path / "partial_resp.json"
    resp_file.write_text(
        json.dumps(
            {
                "model_name": "WarnModel",
                "model_version": "1.0",
                "responses": [
                    {"id": "item_001", "response": "2"},
                    {"id": "unknown_999", "response": "extra"},
                ],
            }
        ),
        encoding="utf-8",
    )
    ids_file = tmp_path / "subset.txt"
    ids_file.write_text("item_001\nitem_002\n", encoding="utf-8")

    out_dir = tmp_path / "out"
    exit_code = main(
        [
            "evaluate",
            "--dataset",
            str(ds_path),
            "--responses",
            str(resp_file),
            "--ids-file",
            str(ids_file),
            "--output",
            str(out_dir),
        ]
    )
    assert exit_code == 0
    captured = capsys.readouterr()

    # stdout has only the completion summary; stderr has the warnings
    assert "Evaluated 1 model(s)" in captured.out
    assert "Warning [WarnModel]" not in captured.out
    assert "Warning [WarnModel]" in captured.err
    assert "item_002" in captured.err
    assert "unknown_999" in captured.err
    assert "excluded 3 record(s)" in captured.err


# ---------------------------------------------------------------------------
# 3.6 Markdown Table Pipe Escaping
# ---------------------------------------------------------------------------

def test_markdown_table_pipe_escaping() -> None:
    """Pipe characters in model names, dataset names, or provider names are escaped in Markdown tables."""
    assert escape_markdown_cell("Model A | Experimental") == r"Model A \| Experimental"

    ds = Dataset(
        name="reasoning | pipe_ds",
        source_path="datasets/reasoning/sample.jsonl",
        records=[
            DatasetRecord(
                id="p1",
                category="cat | one",
                prompt="1+1?",
                expected_answer="2",
                metadata={"difficulty": "easy"},
            )
        ],
    )
    rs1 = ModelResponseSet(
        model_name="Model A | Experimental",
        model_version="1.0 | beta",
        model_provider="local | custom",
        responses=[ResponseItem(id="p1", response="2")],
    )
    rs2 = ModelResponseSet(
        model_name="Model B | Baseline",
        model_version="1.0",
        responses=[ResponseItem(id="p1", response="2")],
    )
    evaluator = Evaluator()
    res1 = evaluator.evaluate(ds, rs1)
    res2 = evaluator.evaluate(ds, rs2)

    md_single = ReportGenerator.to_markdown(res1)
    assert r"`Model A \| Experimental`" in md_single
    assert r"`cat \| one`" in md_single

    cmp = ModelComparator.compare([res1, res2])
    md_cmp = ReportGenerator.comparison_to_markdown(cmp)
    assert r"`Model A \| Experimental`" in md_cmp
    # Verify every table row in Model Comparison Table has the exact same number of unescaped '|' delimiters
    in_cmp_table = False
    pipe_counts: list[int] = []
    for line in md_cmp.splitlines():
        if line.startswith("## Model Comparison Table"):
            in_cmp_table = True
            continue
        if in_cmp_table:
            if line.startswith("## "):
                break
            if line.startswith("|"):
                unescaped_pipes = len(line.replace(r"\|", "").split("|")) - 1
                pipe_counts.append(unescaped_pipes)
    assert len(set(pipe_counts)) == 1


# ---------------------------------------------------------------------------
# 4.4 & 5.2 Experiment Corruption Resilience & Zero SyntaxWarnings
# ---------------------------------------------------------------------------

def test_experiment_manager_skips_corrupted_run_dirs_gracefully(tmp_path: Path) -> None:
    """ExperimentManager.list_runs skips corrupted run directories without crashing."""
    exp_root = tmp_path / "experiments"
    corrupt_dir = exp_root / "run_corrupt_001"
    corrupt_dir.mkdir(parents=True)
    (corrupt_dir / "config.json").write_text("{corrupt json", encoding="utf-8")

    manager = ExperimentManager(experiments_root=exp_root)
    assert manager.list_runs() == []


def test_zero_syntax_warnings_across_all_repository_python_files() -> None:
    """Compiling every .py file in src/, tests/, examples/, and scripts/ with SyntaxWarning=error succeeds."""
    repo_root = Path(__file__).resolve().parent.parent
    py_files = sorted(repo_root.rglob("*.py"))
    assert len(py_files) >= 50

    with warnings.catch_warnings():
        warnings.simplefilter("error", SyntaxWarning)
        for py_file in py_files:
            source = py_file.read_text(encoding="utf-8")
            compile(source, str(py_file), "exec")
