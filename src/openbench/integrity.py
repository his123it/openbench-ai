"""Benchmark integrity, provenance classification, and artifact checksum verifier for OpenBench AI."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ALLOWED_EXECUTION_ORIGINS = {
    "provider_executed",
    "external_responses_evaluated",
    "manually_imported",
}
SUPPORTED_PROVENANCE_SCHEMA_VERSIONS = {"0.3.0", "0.4.0"}


def compute_file_sha256(file_path: str | Path) -> str:
    """Compute the SHA-256 hex digest of a file on disk."""
    target = Path(file_path)
    hasher = hashlib.sha256()
    hasher.update(target.read_bytes())
    return hasher.hexdigest()


def compute_dict_sha256(payload: dict[str, Any]) -> str:
    """Compute deterministic SHA-256 hex digest of a JSON-serializable dictionary."""
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class IntegrityCheckResult:
    """Result of verifying the integrity and provenance of an experiment run directory."""

    run_id: str
    valid: bool
    execution_origin: str
    schema_version: str
    missing_artifacts: list[str] = field(default_factory=list)
    changed_artifacts: list[str] = field(default_factory=list)
    invalid_checksums: dict[str, dict[str, str]] = field(default_factory=dict)
    inconsistent_metadata: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    integrity_disclaimer: str = (
        "Cryptographic SHA-256 checksums verify file and configuration integrity on disk; "
        "they do NOT prove the factual truthfulness of a language model's responses."
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "valid": self.valid,
            "execution_origin": self.execution_origin,
            "schema_version": self.schema_version,
            "missing_artifacts": list(self.missing_artifacts),
            "changed_artifacts": list(self.changed_artifacts),
            "invalid_checksums": dict(self.invalid_checksums),
            "inconsistent_metadata": list(self.inconsistent_metadata),
            "warnings": list(self.warnings),
            "integrity_disclaimer": self.integrity_disclaimer,
        }


class RunIntegrityVerifier:
    """Verifies artifact completeness, SHA-256 checksums, and provenance consistency of experiment runs.

    Provenance Honesty Guarantee:
    Clearly distinguishes between:
    - `"provider_executed"`: A run executed directly by `ExecutionEngine` against a model provider.
    - `"external_responses_evaluated"`: An evaluation performed on an externally supplied response file.
    - `"manually_imported"`: A result imported from an external report.
    Never falsely labels imported or external response evaluations as provider-executed runs.
    """

    REQUIRED_RUN_ARTIFACTS = (
        "config.json",
        "model_metadata.json",
        "responses.json",
        "evaluation.json",
        "performance.json",
        "reproducibility.json",
        "logs.jsonl",
    )

    @staticmethod
    def _check_lineage(target: Path, repro_data: dict[str, Any]) -> list[str]:
        """v0.4.0 runs: cross-check the lineage block between config.json and reproducibility.json."""
        from openbench.lineage import LINEAGE_CROSS_CHECK_FIELDS

        problems: list[str] = []
        repro_lineage = repro_data.get("lineage")
        if not isinstance(repro_lineage, dict):
            return ["reproducibility.json (schema 0.4.0) has no lineage block"]
        cfg_path = target / "config.json"
        if not cfg_path.exists():
            return problems
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        cfg_lineage = cfg.get("lineage") if isinstance(cfg, dict) else None
        if not isinstance(cfg_lineage, dict):
            return ["config.json has no lineage block although reproducibility.json is schema 0.4.0"]
        for key in LINEAGE_CROSS_CHECK_FIELDS:
            if cfg_lineage.get(key) != repro_lineage.get(key):
                problems.append(
                    f"Lineage field '{key}' differs between config.json and reproducibility.json"
                )
        ids = cfg_lineage.get("selected_record_ids")
        hashes = cfg_lineage.get("record_content_sha256")
        if isinstance(ids, list) and isinstance(hashes, dict) and sorted(hashes) != sorted(ids):
            problems.append("Lineage record_content_sha256 keys differ from selected_record_ids")
        model_meta_path = target / "model_metadata.json"
        if model_meta_path.exists():
            meta = json.loads(model_meta_path.read_text(encoding="utf-8"))
            if isinstance(meta, dict) and "repeat_group_id" in meta:
                if meta.get("repeat_group_id") != repro_lineage.get("repeat_group_id") or meta.get(
                    "repetition_index"
                ) != repro_lineage.get("repetition_index"):
                    problems.append(
                        "Repeat-group identity differs between model_metadata.json and "
                        "reproducibility.json"
                    )
        return problems

    @staticmethod
    def verify_run_directory(run_dir: str | Path) -> IntegrityCheckResult:
        target = Path(run_dir)
        run_id = target.name
        if not target.exists() or not target.is_dir():
            return IntegrityCheckResult(
                run_id=run_id,
                valid=False,
                execution_origin="unknown",
                schema_version="unknown",
                missing_artifacts=list(RunIntegrityVerifier.REQUIRED_RUN_ARTIFACTS),
            )

        missing_artifacts: list[str] = []
        for fname in RunIntegrityVerifier.REQUIRED_RUN_ARTIFACTS:
            if not (target / fname).exists():
                missing_artifacts.append(fname)

        repro_path = target / "reproducibility.json"
        if not repro_path.exists():
            return IntegrityCheckResult(
                run_id=run_id,
                valid=False,
                execution_origin="unknown",
                schema_version="unknown",
                missing_artifacts=missing_artifacts,
            )

        try:
            repro_data = json.loads(repro_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            return IntegrityCheckResult(
                run_id=run_id,
                valid=False,
                execution_origin="unknown",
                schema_version="unknown",
                missing_artifacts=missing_artifacts,
                inconsistent_metadata=[f"Corrupted reproducibility.json: {exc.msg}"],
            )

        schema_version = str(repro_data.get("schema_version", "unknown"))
        execution_origin = str(repro_data.get("execution_origin", "unknown"))
        run_id = str(repro_data.get("run_id", run_id))

        inconsistent_metadata: list[str] = []
        if schema_version not in SUPPORTED_PROVENANCE_SCHEMA_VERSIONS:
            inconsistent_metadata.append(
                f"Unsupported provenance schema_version '{schema_version}'. "
                f"Supported: {sorted(SUPPORTED_PROVENANCE_SCHEMA_VERSIONS)}"
            )
        if execution_origin not in ALLOWED_EXECUTION_ORIGINS:
            inconsistent_metadata.append(
                f"Invalid execution_origin '{execution_origin}'. "
                f"Allowed: {sorted(ALLOWED_EXECUTION_ORIGINS)}"
            )

        # Verify file artifact SHA-256 checksums
        changed_artifacts: list[str] = []
        invalid_checksums: dict[str, dict[str, str]] = {}
        recorded_checksums = repro_data.get("artifact_checksums", {})

        for artifact_key, filename in (
            ("responses_sha256", "responses.json"),
            ("evaluation_sha256", "evaluation.json"),
            ("performance_sha256", "performance.json"),
            ("config_sha256", "config.json"),
            ("model_metadata_sha256", "model_metadata.json"),
        ):
            fpath = target / filename
            expected_hash = recorded_checksums.get(artifact_key)
            if fpath.exists() and expected_hash:
                actual_hash = compute_file_sha256(fpath)
                if actual_hash != expected_hash:
                    changed_artifacts.append(filename)
                    invalid_checksums[filename] = {
                        "expected_sha256": str(expected_hash),
                        "actual_sha256": actual_hash,
                    }

        # Verify metadata consistency between config.json, model_metadata.json, evaluation.json, and reproducibility.json
        try:
            if (target / "model_metadata.json").exists():
                model_meta = json.loads((target / "model_metadata.json").read_text(encoding="utf-8"))
                if model_meta.get("model_name") != repro_data.get("model_name"):
                    inconsistent_metadata.append(
                        f"Model name mismatch between model_metadata.json ('{model_meta.get('model_name')}') "
                        f"and reproducibility.json ('{repro_data.get('model_name')}')"
                    )
                if model_meta.get("execution_origin") and model_meta.get("execution_origin") != execution_origin:
                    inconsistent_metadata.append(
                        "Execution origin mismatch between model_metadata.json and reproducibility.json"
                    )
            if (target / "evaluation.json").exists():
                eval_meta = json.loads((target / "evaluation.json").read_text(encoding="utf-8"))
                eval_ds_checksum = eval_meta.get("dataset", {}).get("checksum")
                repro_ds_checksum = repro_data.get("dataset_checksum")
                if eval_ds_checksum and repro_ds_checksum and eval_ds_checksum != repro_ds_checksum:
                    inconsistent_metadata.append(
                        f"Dataset checksum mismatch between evaluation.json ('{eval_ds_checksum}') "
                        f"and reproducibility.json ('{repro_ds_checksum}')"
                    )
            if schema_version == "0.4.0":
                inconsistent_metadata.extend(
                    RunIntegrityVerifier._check_lineage(target, repro_data)
                )
        except json.JSONDecodeError as exc:
            inconsistent_metadata.append(f"Malformed JSON in run artifact: {exc.msg}")

        warnings: list[str] = []
        if repro_data.get("is_mock_provider"):
            warnings.append(
                "Run was executed using the deterministic MockModelProvider (`is_mock=True`); "
                "results do not represent a real language model."
            )
        if execution_origin != "provider_executed":
            warnings.append(
                f"Run provenance is '{execution_origin}'; responses were not generated live by the execution engine."
            )

        is_valid = (
            len(missing_artifacts) == 0
            and len(changed_artifacts) == 0
            and len(invalid_checksums) == 0
            and len(inconsistent_metadata) == 0
        )

        return IntegrityCheckResult(
            run_id=run_id,
            valid=is_valid,
            execution_origin=execution_origin,
            schema_version=schema_version,
            missing_artifacts=missing_artifacts,
            changed_artifacts=changed_artifacts,
            invalid_checksums=invalid_checksums,
            inconsistent_metadata=inconsistent_metadata,
            warnings=warnings,
        )
