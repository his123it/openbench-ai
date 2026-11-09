"""Run lineage, repeat groups and generation-seed provenance for OpenBench AI (v0.4.0).

Every run created by v0.4.0 records a ``lineage`` block in ``config.json`` and
``reproducibility.json`` (run ``schema_version`` ``"0.4.0"``). The block identifies the
run's repeat group and repetition index, the generation seeds and what happened to them,
the exact analysed record IDs and a content hash of every analysed record, and the cache
context the run used. Per-run lineage is authoritative; the repeat-group index file is a
rebuildable summary of it.

Runs written by v0.3.x have no lineage. Readers map them to a single run
(``repeat_group_id = None``, ``repetition_index = 0``, ``repetition_count = 1``) with
``seed_status = "unknown"``; nothing is ever written back to those runs.
"""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from openbench.exceptions import ConfigurationError

RUN_SCHEMA_VERSION = "0.4.0"
LEGACY_RUN_SCHEMA_VERSION = "0.3.0"
GROUP_INDEX_SCHEMA_VERSION = "0.4.0"

SEED_STATUSES = (
    "not_supplied",
    "unsupported_by_adapter",
    "passed_to_provider",
    "no_provider_requests",
    "unknown",
)

SEED_DISCLAIMER = (
    "A seed passed to a provider does not guarantee deterministic output; the provider may "
    "still produce non-deterministic responses."
)

_GROUP_ID_RE = re.compile(r"^grp_[A-Za-z0-9_-]{1,64}$")
_RUN_ID_RE = re.compile(r"^run_[A-Za-z0-9_.-]{1,128}$")
_EXPERIMENT_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")


def new_repeat_group_id() -> str:
    """Return a new random repeat-group ID: ``grp_`` followed by 12 hex characters."""
    return f"grp_{uuid.uuid4().hex[:12]}"


def is_repeat_group_id(value: str) -> bool:
    return isinstance(value, str) and _GROUP_ID_RE.match(value) is not None


def validate_repeat_group_id(value: str) -> str:
    """Validate a repeat-group ID (it is also used as a file name, so it is restricted)."""
    if not is_repeat_group_id(value):
        raise ConfigurationError(
            f"Invalid repeat group ID {value!r}: expected 'grp_' followed by 1-64 characters "
            "from [A-Za-z0-9_-]"
        )
    return value


def validate_run_id(value: str) -> str:
    """Validate a run ID used to address a run directory (prevents path traversal)."""
    candidate = value if value.startswith("run_") else f"run_{value}"
    if _RUN_ID_RE.match(candidate) is None or candidate in {"run_.", "run_.."}:
        raise ConfigurationError(
            f"Invalid run ID {value!r}: expected 'run_' followed by 1-128 characters from "
            "[A-Za-z0-9_.-]"
        )
    return candidate


def validate_experiment_id(value: str | None, field_name: str = "experiment_id") -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or _EXPERIMENT_ID_RE.match(value) is None:
        raise ConfigurationError(
            f"Invalid {field_name} {value!r}: expected 1-128 characters from [A-Za-z0-9_.:-]"
        )
    return value


@dataclass(frozen=True)
class RunLineage:
    """Identity of one run within an experiment (v0.4.0).

    ``repeat_group_id`` is ``None`` for a single run, which then has
    ``repetition_index = 0`` and ``repetition_count = 1``. ``repetition_seed`` is
    ``generation_base_seed + repetition_index`` when a base seed was supplied.
    """

    repeat_group_id: str | None = None
    repetition_index: int = 0
    repetition_count: int = 1
    generation_base_seed: int | None = None
    repetition_seed: int | None = None
    experiment_id: str | None = None
    parent_experiment_id: str | None = None

    def __post_init__(self) -> None:
        for name in ("repetition_index", "repetition_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ConfigurationError(f"RunLineage.{name} must be an integer")
        for name in ("generation_base_seed", "repetition_seed"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int)):
                raise ConfigurationError(f"RunLineage.{name} must be an integer or None")
        if self.repetition_count < 1:
            raise ConfigurationError("RunLineage.repetition_count must be >= 1")
        if not 0 <= self.repetition_index < self.repetition_count:
            raise ConfigurationError(
                "RunLineage.repetition_index must lie in [0, repetition_count)"
            )
        if self.repeat_group_id is None:
            if self.repetition_count != 1 or self.repetition_index != 0:
                raise ConfigurationError(
                    "A run without a repeat_group_id is a single run (index 0, count 1)"
                )
        else:
            validate_repeat_group_id(self.repeat_group_id)
        if (
            self.generation_base_seed is not None
            and self.repetition_seed is not None
            and self.repetition_seed != self.generation_base_seed + self.repetition_index
        ):
            raise ConfigurationError(
                "RunLineage.repetition_seed must equal generation_base_seed + repetition_index"
            )
        validate_experiment_id(self.experiment_id)
        validate_experiment_id(self.parent_experiment_id, "parent_experiment_id")

    @classmethod
    def for_repetition(
        cls,
        repeat_group_id: str,
        repetition_index: int,
        repetition_count: int,
        generation_base_seed: int | None = None,
        experiment_id: str | None = None,
        parent_experiment_id: str | None = None,
    ) -> RunLineage:
        seed = None if generation_base_seed is None else generation_base_seed + repetition_index
        return cls(
            repeat_group_id=repeat_group_id,
            repetition_index=repetition_index,
            repetition_count=repetition_count,
            generation_base_seed=generation_base_seed,
            repetition_seed=seed,
            experiment_id=experiment_id,
            parent_experiment_id=parent_experiment_id,
        )

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> RunLineage:
        """Rebuild the identity part of a stored lineage block."""
        return cls(
            repeat_group_id=payload.get("repeat_group_id"),
            repetition_index=int(payload.get("repetition_index", 0)),
            repetition_count=int(payload.get("repetition_count", 1)),
            generation_base_seed=payload.get("generation_base_seed"),
            repetition_seed=payload.get("repetition_seed"),
            experiment_id=payload.get("experiment_id"),
            parent_experiment_id=payload.get("parent_experiment_id"),
        )

    @property
    def repetition_context(self) -> tuple[str, int] | None:
        """The cache context ``(repeat_group_id, repetition_index)``, or ``None``."""
        if self.repeat_group_id is None:
            return None
        return (self.repeat_group_id, self.repetition_index)

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "parent_experiment_id": self.parent_experiment_id,
            "repeat_group_id": self.repeat_group_id,
            "repetition_index": self.repetition_index,
            "repetition_count": self.repetition_count,
            "generation_base_seed": self.generation_base_seed,
            "repetition_seed": self.repetition_seed,
        }


def seed_status(
    seed_supplied: int | None, supported_by_adapter: bool, forwarded_requests: int
) -> str:
    """Classify what happened to a generation seed in one run."""
    if seed_supplied is None:
        return "not_supplied"
    if not supported_by_adapter:
        return "unsupported_by_adapter"
    if forwarded_requests > 0:
        return "passed_to_provider"
    return "no_provider_requests"


def canonical_record_sha256(record_dict: dict[str, Any]) -> str:
    """SHA-256 of the canonical JSON of a full dataset record (ID, prompt, answer, metadata)."""
    canonical = json.dumps(record_dict, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def record_content_hashes(records: Iterable[Any]) -> dict[str, str]:
    """``{record_id: sha256(canonical record JSON)}`` sorted by ID."""
    hashes = {rec.id: canonical_record_sha256(rec.to_dict()) for rec in records}
    return {key: hashes[key] for key in sorted(hashes)}


def ids_sha256(ids: Iterable[str]) -> str:
    """SHA-256 of the canonical JSON list of sorted IDs."""
    ordered = sorted(ids)
    return hashlib.sha256(json.dumps(ordered, ensure_ascii=False).encode("utf-8")).hexdigest()


def legacy_lineage() -> dict[str, Any]:
    """Lineage view of a v0.3.x run (no lineage on disk). Read-only mapping."""
    return {
        "schema_version": LEGACY_RUN_SCHEMA_VERSION,
        "legacy_run": True,
        "experiment_id": None,
        "parent_experiment_id": None,
        "repeat_group_id": None,
        "repetition_index": 0,
        "repetition_count": 1,
        "generation_base_seed": None,
        "repetition_seed": None,
        "seed_supplied": None,
        "seed_supported_by_adapter": None,
        "seed_passed_to_provider": None,
        "seed_status": "unknown",
        "selected_record_ids": None,
        "selected_record_ids_sha256": None,
        "record_content_sha256": None,
        "cache_provenance": None,
    }


#: Lineage fields cross-checked between config.json and reproducibility.json.
LINEAGE_CROSS_CHECK_FIELDS = (
    "experiment_id",
    "parent_experiment_id",
    "repeat_group_id",
    "repetition_index",
    "repetition_count",
    "generation_base_seed",
    "repetition_seed",
    "seed_supplied",
    "seed_status",
    "selected_record_ids_sha256",
    "record_content_sha256",
)


def generation_config_without_seed(config: dict[str, Any]) -> dict[str, Any]:
    """Copy of a generation configuration without the per-repetition ``seed`` key."""
    return {k: v for k, v in config.items() if k != "seed"}
