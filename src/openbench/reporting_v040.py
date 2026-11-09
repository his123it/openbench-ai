"""Renderers for v0.4.0 analysis artifacts (JSON, Markdown, CSV)."""

from __future__ import annotations

import json
from typing import Any

from openbench.exceptions import ConfigurationError


def to_json(payload: dict[str, Any]) -> str:
    """Serialise strictly (no NaN or infinity) with stable key order."""
    return json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n"


def render_analysis(payload: dict[str, Any], fmt: str) -> str:
    if fmt == "json":
        return to_json(payload)
    raise ConfigurationError(f"Unsupported analysis format '{fmt}'")


def render_preferences(payload: dict[str, Any], fmt: str) -> str:
    if fmt == "json":
        return to_json(payload)
    raise ConfigurationError(f"Unsupported preference report format '{fmt}'")


def render_audit(payload: dict[str, Any], fmt: str) -> str:
    if fmt == "json":
        return to_json(payload)
    raise ConfigurationError(f"Unsupported audit report format '{fmt}'")
