"""Security, privacy, credential redaction, and path/URL validation utilities for OpenBench AI."""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from openbench.exceptions import ConfigurationError

_SECRET_KEY_REGEX = re.compile(
    r"(?i)(?:^|[_.\-\s])(api_key|apikey|access_token|auth_token|client_secret|private_key|secret_key|secret|password|passwd|pwd|credential|credentials|authorization|auth|bearer|token|key)(?:$|[_.\-\s])"
)
_SAFE_KEY_ALLOWLIST = {
    "keyword_coverage",
    "keyword_match",
    "keywords",
    "matched_keywords",
    "missing_keywords",
    "total_keywords",
    "required_keys",
    "missing_keys",
    "cache_key",
    "token_usage",
    "token_usage_supplied",
    "token_usage_summary",
    "token_source",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "total_prompt_tokens",
    "total_completion_tokens",
    "mean_total_tokens",
    "tokens_per_second",
    "max_tokens",
    "max_new_tokens",
    "estimate_missing_tokens",
    "report_tokens",
    "reference_token_count",
    "candidate_token_count",
    "overlap_tokens",
    "prompt_cost_per_1k_tokens",
    "completion_cost_per_1k_tokens",
    "records_with_explicit_keywords",
}


def _is_sensitive_key_name(key_str: str) -> bool:
    low = key_str.strip().lower()
    if low in _SAFE_KEY_ALLOWLIST:
        return False
    return _SECRET_KEY_REGEX.search(low) is not None

# Value-level credential patterns embedded inside strings (URLs, headers, assignments, token prefixes)
# 1. URL query parameters: ?token=abc, &api_key=xyz, ?client_secret=123, &password=foo, &access_token=bar, &secret=baz, &key=qux, &sig=123
_URL_QUERY_SECRET_RE = re.compile(
    r"(?i)([?&](?:access_token|api_key|apikey|auth_token|client_secret|key|password|passwd| pwd|secret|secret_key|sig|signature|token)=)([^&#\s\"']+)"
)

# 2. Authorization headers & schemes: Authorization: Bearer abc, Bearer eyJ..., Basic dXNlcjpwYXNz
_AUTH_HEADER_RE = re.compile(
    r"(?i)(\b(?:authorization\s*[:=]\s*(?:bearer|basic|token|apikey)\s+|bearer\s+|basic\s+))([A-Za-z0-9._\-~+/]+=*)"
)

# 3. Inline key=value or key: value assignments inside strings (e.g. password=example, client_secret=example, api_key="secret123")
_INLINE_ASSIGNMENT_RE = re.compile(
    r"(?i)(\b(?:api_key|apikey|access_token|auth_token|client_secret|password|passwd|private_key|secret_key|secret)\s*[:=]\s*[\"']?)([^\"'\s,;}{)(\]\[]+)([\"']?)"
)

# 4. Standard token prefixes (OpenAI sk-..., Hugging Face hf_..., GitHub ghp_/gho_/ghs_/github_pat_, AWS AKIA..., Slack xoxb-/xoxp-)
_TOKEN_PREFIX_RE = re.compile(
    r"\b(sk-[A-Za-z0-9_\-]{8,}|hf_[A-Za-z0-9]{8,}|gh[pousr]_[A-Za-z0-9]{8,}|github_pat_[A-Za-z0-9_]{10,}|AKIA[0-9A-Z]{16}|xox[baprs]-[A-Za-z0-9\-]{8,})\b"
)

# 5. URL-embedded user:password@host credentials
_URL_CRED_RE = re.compile(r"(https?://)([^/:\s]+):([^@/\s]+)@")


def sanitize_error_message(message: str) -> str:
    """Scrub embedded credentials, query tokens, headers, assignments, and API keys from a string.

    Preserves harmless text and ordinary identifiers while redacting:
    - URL query parameters (`?token=abc123`, `&api_key=abc123`, `&client_secret=...`)
    - Authorization headers and Bearer/Basic tokens (`Authorization: Bearer abc123`)
    - Inline credential assignments (`password=example`, `client_secret=example`)
    - Recognized API token prefixes (`sk-...`, `hf_...`, `ghp_...`, `AKIA...`)
    - URL-embedded `user:password@host` credentials
    """
    if not message:
        return message

    msg = _URL_CRED_RE.sub(r"\1[REDACTED]:[REDACTED]@", message)
    msg = _URL_QUERY_SECRET_RE.sub(r"\1[REDACTED]", msg)
    msg = _AUTH_HEADER_RE.sub(r"\1[REDACTED]", msg)

    def _replace_assignment(match: re.Match[str]) -> str:
        prefix, val, suffix = match.group(1), match.group(2), match.group(3)
        if val.startswith("[REDACTED"):
            return match.group(0)
        return f"{prefix}[REDACTED]{suffix}"

    msg = _INLINE_ASSIGNMENT_RE.sub(_replace_assignment, msg)
    msg = _TOKEN_PREFIX_RE.sub("[REDACTED_TOKEN]", msg)
    return msg


def redact_sensitive_data(data: Any) -> Any:
    """Recursively redact sensitive keys and string-embedded credentials across dicts, lists, and strings.

    Note: Does not mutate the input object in place; returns a new sanitized structure so
    runtime configurations and request payloads remain intact for live provider execution.
    """
    if isinstance(data, dict):
        cleaned: dict[str, Any] = {}
        for k, v in data.items():
            key_str = str(k)
            if _is_sensitive_key_name(key_str):
                cleaned[key_str] = "[REDACTED]"
            else:
                cleaned[key_str] = redact_sensitive_data(v)
        return cleaned
    if isinstance(data, list):
        return [redact_sensitive_data(item) for item in data]
    if isinstance(data, tuple):
        return tuple(redact_sensitive_data(item) for item in data)
    if isinstance(data, str):
        return sanitize_error_message(data)
    return data


def redact_prompt_if_private(text: str, privacy_mode: bool = False) -> str:
    """Mask prompt or response text in logs when configurable `privacy_mode=True` is active.

    Even when `privacy_mode=False`, scrubs any accidental inline credentials from logged text.
    """
    if not privacy_mode:
        return sanitize_error_message(text)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
    return f"[REDACTED_CONTENT sha256={digest} len={len(text)}]"


def validate_endpoint_url(
    url: str,
    allow_remote_http: bool = False,
) -> str:
    """Validate an external or local HTTP/HTTPS endpoint URL safely.

    Security Rules:
    - Only `http` and `https` schemes are permitted (`file://`, `ftp://`, `javascript:`, etc. are rejected).
    - Rejects URLs containing embedded `username:password@` credentials.
    - Rejects control characters or whitespace.
    - For non-loopback remote hosts, requires `https://` unless `allow_remote_http=True` is explicitly set.
    """
    if not isinstance(url, str) or not url.strip():
        raise ConfigurationError("Endpoint URL must be a non-empty string")
    cleaned = url.strip()
    if any(ch in cleaned for ch in (" ", "\n", "\r", "\t", "\0")):
        raise ConfigurationError("Endpoint URL contains invalid whitespace or control characters")

    parsed = urlparse(cleaned)
    if parsed.scheme not in {"http", "https"}:
        raise ConfigurationError(
            f"Unsupported URL scheme '{parsed.scheme}'. Only 'http' and 'https' are permitted."
        )
    if not parsed.hostname:
        raise ConfigurationError(f"Endpoint URL is missing a valid hostname: {sanitize_error_message(cleaned)}")
    if parsed.username or parsed.password:
        raise ConfigurationError(
            "Embedding credentials directly in endpoint URLs is forbidden for security."
        )

    is_loopback = parsed.hostname in {"localhost", "127.0.0.1", "::1"}
    if not is_loopback and parsed.scheme == "http" and not allow_remote_http:
        raise ConfigurationError(
            f"Plain HTTP is only allowed for loopback hosts (localhost / 127.0.0.1). "
            f"Use 'https://' for remote host '{parsed.hostname}' or explicitly enable allow_remote_http=True."
        )

    return cleaned.rstrip("/")


def validate_safe_output_path(
    target_path: str | Path,
    allowed_root: str | Path | None = None,
) -> Path:
    """Validate a filesystem output path to prevent null-byte injection or unauthorized traversal."""
    raw_str = str(target_path)
    if "\0" in raw_str:
        raise ConfigurationError("Output path contains forbidden null byte")

    path_obj = Path(target_path)
    resolved = path_obj.resolve()

    if allowed_root is not None:
        root_resolved = Path(allowed_root).resolve()
        try:
            resolved.relative_to(root_resolved)
        except ValueError as exc:
            raise ConfigurationError(
                f"Output path '{resolved}' escapes restricted workspace root '{root_resolved}'"
            ) from exc

    return path_obj


def resolve_env_credential(env_var_name: str | None) -> str | None:
    """Read an optional credential from an environment variable name without storing it in configs."""
    if not env_var_name:
        return None
    clean_name = env_var_name.strip()
    if not re.match(r"^[A-Z_][A-Z0-9_]*$", clean_name):
        raise ConfigurationError(
            f"Invalid environment variable identifier '{env_var_name}'"
        )
    return os.environ.get(clean_name)
