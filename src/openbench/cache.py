"""Deterministic, configuration-aware response cache for OpenBench AI."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openbench.evaluator import ResponseItem
from openbench.exceptions import OpenBenchError
from openbench.security import redact_sensitive_data


@dataclass(frozen=True)
class CacheRecord:
    """Auditable cache entry storing response data and full request provenance."""

    cache_key: str
    provider_name: str
    model_name: str
    model_version: str
    dataset_name: str
    dataset_version: str
    record_id: str
    prompt_sha256: str
    generation_config_sha256: str
    response: str
    latency_ms: float | None
    token_usage: dict[str, int] | None
    cached_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "cache_key": self.cache_key,
            "provider_name": self.provider_name,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "dataset_name": self.dataset_name,
            "dataset_version": self.dataset_version,
            "record_id": self.record_id,
            "prompt_sha256": self.prompt_sha256,
            "generation_config_sha256": self.generation_config_sha256,
            "response": self.response,
            "latency_ms": self.latency_ms,
            "token_usage": self.token_usage,
            "cached_at": self.cached_at,
        }


class ResponseCache:
    """File-backed deterministic response cache preventing cross-model/cross-config collisions.

    Cache Key Isolation & Credential Hygiene Guarantee:
    - The SHA-256 `cache_key` incorporates `provider_name`, `model_name`, `model_version`,
      `dataset_name`, `dataset_version`, `record_id`, `prompt_sha256`, and the canonical JSON
      hash of `redact_sensitive_data(generation_config)`.
    - Redacting credentials before hashing and storing ensures that rotating an API key or
      token does not invalidate valid cached responses and that raw credentials are never
      persisted inside cache artifacts on disk.
    """

    def __init__(
        self,
        cache_dir: str | Path | None = None,
        enabled: bool = True,
    ) -> None:
        if cache_dir is not None:
            self.cache_dir = Path(cache_dir)
        else:
            self.cache_dir = Path.cwd() / ".openbench_cache"
        self.enabled = enabled
        self.hits = 0
        self.misses = 0
        self._index_path = self.cache_dir / "cache_entries.jsonl"
        self._memory_map: dict[str, CacheRecord] | None = None

    @staticmethod
    def compute_cache_key(
        provider_name: str,
        model_name: str,
        model_version: str,
        dataset_name: str,
        dataset_version: str,
        record_id: str,
        prompt: str,
        generation_config: dict[str, Any] | None = None,
    ) -> tuple[str, str, str]:
        """Return `(cache_key, prompt_sha256, generation_config_sha256)`."""
        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        safe_gen_cfg = redact_sensitive_data(generation_config or {})
        gen_canonical = json.dumps(safe_gen_cfg, sort_keys=True, ensure_ascii=False)
        gen_hash = hashlib.sha256(gen_canonical.encode("utf-8")).hexdigest()

        composite = json.dumps(
            {
                "provider_name": provider_name.strip(),
                "model_name": model_name.strip(),
                "model_version": model_version.strip(),
                "dataset_name": dataset_name.strip(),
                "dataset_version": dataset_version.strip(),
                "record_id": record_id.strip(),
                "prompt_sha256": prompt_hash,
                "generation_config_sha256": gen_hash,
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        key = hashlib.sha256(composite.encode("utf-8")).hexdigest()
        return key, prompt_hash, gen_hash

    def _ensure_loaded(self) -> dict[str, CacheRecord]:
        if self._memory_map is not None:
            return self._memory_map
        self._memory_map = {}
        if not self._index_path.exists():
            return self._memory_map

        for line in self._index_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            try:
                data = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise OpenBenchError(f"Corrupted cache file {self._index_path}: {exc.msg}") from exc
            rec = CacheRecord(
                cache_key=str(data["cache_key"]),
                provider_name=str(data["provider_name"]),
                model_name=str(data["model_name"]),
                model_version=str(data["model_version"]),
                dataset_name=str(data["dataset_name"]),
                dataset_version=str(data["dataset_version"]),
                record_id=str(data["record_id"]),
                prompt_sha256=str(data["prompt_sha256"]),
                generation_config_sha256=str(data["generation_config_sha256"]),
                response=str(data["response"]),
                latency_ms=data.get("latency_ms"),
                token_usage=data.get("token_usage"),
                cached_at=str(data["cached_at"]),
            )
            self._memory_map[rec.cache_key] = rec
        return self._memory_map

    def get(
        self,
        provider_name: str,
        model_name: str,
        model_version: str,
        dataset_name: str,
        dataset_version: str,
        record_id: str,
        prompt: str,
        generation_config: dict[str, Any] | None = None,
    ) -> CacheRecord | None:
        """Look up a cached response for the exact request parameters."""
        if not self.enabled:
            self.misses += 1
            return None

        key, _, _ = self.compute_cache_key(
            provider_name=provider_name,
            model_name=model_name,
            model_version=model_version,
            dataset_name=dataset_name,
            dataset_version=dataset_version,
            record_id=record_id,
            prompt=prompt,
            generation_config=generation_config,
        )
        store = self._ensure_loaded()
        if key in store:
            self.hits += 1
            return store[key]
        self.misses += 1
        return None

    def put(
        self,
        provider_name: str,
        model_name: str,
        model_version: str,
        dataset_name: str,
        dataset_version: str,
        record_id: str,
        prompt: str,
        response_item: ResponseItem,
        generation_config: dict[str, Any] | None = None,
    ) -> CacheRecord:
        """Store a response in the cache with full configuration provenance (redacting any secrets)."""
        key, prompt_hash, gen_hash = self.compute_cache_key(
            provider_name=provider_name,
            model_name=model_name,
            model_version=model_version,
            dataset_name=dataset_name,
            dataset_version=dataset_version,
            record_id=record_id,
            prompt=prompt,
            generation_config=generation_config,
        )
        now_iso = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        rec = CacheRecord(
            cache_key=key,
            provider_name=provider_name,
            model_name=model_name,
            model_version=model_version,
            dataset_name=dataset_name,
            dataset_version=dataset_version,
            record_id=record_id,
            prompt_sha256=prompt_hash,
            generation_config_sha256=gen_hash,
            response=response_item.response,
            latency_ms=response_item.latency_ms,
            token_usage=response_item.token_usage,
            cached_at=now_iso,
        )
        if not self.enabled:
            return rec

        store = self._ensure_loaded()
        store[key] = rec
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        with self._index_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec.to_dict(), ensure_ascii=False) + "\n")
        return rec

    def clear(self) -> int:
        """Delete all cached entries on disk and in memory, returning the number of cleared entries."""
        store = self._ensure_loaded()
        count = len(store)
        self._memory_map = {}
        if self._index_path.exists():
            self._index_path.unlink()
        return count

    def stats(self) -> dict[str, Any]:
        """Return cache audit summary."""
        store = self._ensure_loaded()
        return {
            "enabled": self.enabled,
            "cache_dir": str(self.cache_dir),
            "stored_entries": len(store),
            "session_hits": self.hits,
            "session_misses": self.misses,
        }
