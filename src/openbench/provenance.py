"""Provenance blocks for v0.4.0 analysis, preference and audit reports."""

from __future__ import annotations

import hashlib
import json
import platform
from datetime import datetime, timezone
from typing import Any

INTEGRITY_DISCLAIMER = (
    "SHA-256 checksums show that files match recorded checksums. They do not prove that "
    "results are factually correct, that a model produced them, or who produced them."
)


def parameters_sha256(parameters: dict[str, Any]) -> str:
    canonical = json.dumps(parameters, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def provenance_block(
    *,
    parameters: dict[str, Any] | None = None,
    inputs: list[dict[str, Any]] | None = None,
    analysis_seed: int | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Software, environment, input checksums and parameter hash for a report."""
    from openbench import __version__

    block: dict[str, Any] = {
        "openbench_version": __version__,
        "python_version": platform.python_version(),
        "platform": f"{platform.system()} {platform.machine()}",
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "analysis_seed": analysis_seed,
        "parameters_sha256": parameters_sha256(parameters or {}),
        "inputs": list(inputs or []),
        "provider_calls": 0,
        "integrity_disclaimer": INTEGRITY_DISCLAIMER,
    }
    if extra:
        block.update(extra)
    return block


def arm_inputs(arms: list[Any]) -> list[dict[str, Any]]:
    """Input provenance for comparison arms: files, checksums, dataset and config hashes."""
    out = []
    for arm in arms:
        evs = arm.evaluations
        out.append(
            {
                "arm_id": arm.arm_id,
                "source": arm.source,
                "files": [
                    {"path": p, "sha256": h}
                    for p, h in zip(arm.source_paths, arm.input_sha256, strict=False)
                ],
                "datasets": sorted(
                    {
                        (str(ev.dataset_summary.get("name")), str(ev.dataset_summary.get("version")), str(ev.dataset_summary.get("checksum")))
                        for ev in evs
                    }
                ),
                "evaluation_configuration_sha256": sorted({parameters_sha256(ev.config) for ev in evs}),
                "metric_versions": sorted(
                    {(str(m.get("name")), str(m.get("version"))) for ev in evs for m in ev.metrics_used if isinstance(m, dict)}
                ),
                "run_lineage": [
                    {
                        k: (r.get("lineage") or {}).get(k)
                        for k in (
                            "repeat_group_id",
                            "repetition_index",
                            "repetition_count",
                            "generation_base_seed",
                            "repetition_seed",
                            "seed_status",
                            "selected_record_ids_sha256",
                            "experiment_id",
                            "parent_experiment_id",
                        )
                    }
                    | {"run_id": r.get("run_id"), "status": r.get("status"), "started_at": r.get("started_at"), "completed_at": r.get("completed_at")}
                    for r in arm.run_records
                ],
            }
        )
    return out
