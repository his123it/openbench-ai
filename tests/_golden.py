"""Masking helpers for comparing evaluate output with the frozen v0.3.1 golden digests."""

from __future__ import annotations

import hashlib
import json
import re

VOLATILE_JSON_KEYS = {"evaluation_id", "evaluation_timestamp", "project_version", "openbench_version", "python_version", "operating_system", "platform"}
_MD_VOLATILE = re.compile(r"^(- \*\*Evaluation ID\*\*|- \*\*Evaluation Timestamp\*\*|- \*\*OpenBench AI Version\*\*|\| \*\*Evaluation ID\*\*|\| \*\*Evaluation Timestamp \(UTC\)\*\*|\| \*\*OpenBench AI Version\*\*|\| \*\*Python Environment\*\*).*$", re.M)


def _strip(value):
    if isinstance(value, dict):
        return {k: _strip(v) for k, v in value.items() if k not in VOLATILE_JSON_KEYS}
    if isinstance(value, list):
        return [_strip(v) for v in value]
    return value


def masked_json_digest(text: str) -> str:
    payload = _strip(json.loads(text))
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def masked_md_digest(text: str) -> str:
    return hashlib.sha256(_MD_VOLATILE.sub(r"\1 <masked>", text).encode()).hexdigest()
