"""Configurable text normalization utilities for deterministic evaluation."""

from __future__ import annotations

import re
import string
from dataclasses import dataclass
from typing import Any

_WHITESPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class NormalizationConfig:
    """Configuration controlling text normalization prior to metric comparison.

    By default:
    - Leading and trailing whitespace is stripped (`strip_whitespace=True`).
    - Case is normalized to lowercase (`case_sensitive=False`).
    - Repeated internal whitespace is collapsed to a single space (`collapse_whitespace=True`).
    - Punctuation is preserved (`strip_punctuation=False`) so meaningful symbols
      in code, math, or structured formats are never silently discarded.
    """

    strip_whitespace: bool = True
    case_sensitive: bool = False
    collapse_whitespace: bool = True
    strip_punctuation: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Serialize configuration to a plain dictionary."""
        return {
            "strip_whitespace": self.strip_whitespace,
            "case_sensitive": self.case_sensitive,
            "collapse_whitespace": self.collapse_whitespace,
            "strip_punctuation": self.strip_punctuation,
        }


def normalize_text(text: str, config: NormalizationConfig | None = None) -> str:
    """Normalize a string according to the provided NormalizationConfig."""
    cfg = config or NormalizationConfig()
    result = text

    if cfg.strip_whitespace:
        result = result.strip()

    if cfg.strip_punctuation:
        result = result.translate(str.maketrans("", "", string.punctuation))

    if cfg.collapse_whitespace:
        result = _WHITESPACE_RE.sub(" ", result)
        if cfg.strip_whitespace:
            result = result.strip()

    if not cfg.case_sensitive:
        result = result.lower()

    return result
