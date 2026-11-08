"""Unit tests for security & privacy controls: token redaction, URL validation, privacy mode, and path checks."""

from __future__ import annotations

from pathlib import Path

import pytest

from openbench.exceptions import ConfigurationError
from openbench.security import (
    redact_prompt_if_private,
    redact_sensitive_data,
    sanitize_error_message,
    validate_endpoint_url,
    validate_safe_output_path,
)


def test_redact_sensitive_data_and_sanitize_error_messages() -> None:
    """API keys, bearer tokens, and URL credentials are scrubbed from dicts and error strings."""
    raw_cfg = {
        "model": "llama3",
        "api_key": "sk-1234567890abcdef123456",
        "nested": {"auth_token": "hf_secret9876543210"},
    }
    cleaned = redact_sensitive_data(raw_cfg)
    assert cleaned["api_key"] == "[REDACTED]"
    assert cleaned["nested"]["auth_token"] == "[REDACTED]"

    raw_err = "Request to https://user:my_pass@example.com failed with Bearer eyJhbGciOiJIUzI1NiJ9 and key sk-abcdef1234567890"
    sanitized = sanitize_error_message(raw_err)
    assert "my_pass" not in sanitized
    assert "eyJhbGciOiJIUzI1NiJ9" not in sanitized
    assert "sk-abcdef1234567890" not in sanitized
    assert "[REDACTED]" in sanitized


def test_validate_endpoint_url_blocks_unsafe_schemes_and_remote_plain_http() -> None:
    """validate_endpoint_url permits localhost HTTP and remote HTTPS, but blocks file://, credentials, and remote HTTP."""
    assert validate_endpoint_url("http://localhost:11434/") == "http://localhost:11434"
    assert validate_endpoint_url("https://api.example.org/v1") == "https://api.example.org/v1"

    with pytest.raises(ConfigurationError, match="Unsupported URL scheme"):
        validate_endpoint_url("file:///etc/passwd")

    with pytest.raises(ConfigurationError, match="Embedding credentials"):
        validate_endpoint_url("https://admin:secret@example.org")

    with pytest.raises(ConfigurationError, match="Plain HTTP is only allowed for loopback"):
        validate_endpoint_url("http://remote-llm.example.org:11434")


def test_privacy_mode_redacts_prompt_content() -> None:
    """redact_prompt_if_private replaces sensitive prompt text with a SHA-256 fingerprint when enabled."""
    sensitive_prompt = "Patient SSN 000-11-2222 clinical note"
    assert redact_prompt_if_private(sensitive_prompt, privacy_mode=False) == sensitive_prompt

    masked = redact_prompt_if_private(sensitive_prompt, privacy_mode=True)
    assert "000-11-2222" not in masked
    assert masked.startswith("[REDACTED_CONTENT sha256=")


def test_validate_safe_output_path_prevents_workspace_escape(tmp_path: Path) -> None:
    """validate_safe_output_path rejects paths escaping allowed_root or containing null bytes."""
    safe_child = tmp_path / "results" / "out.json"
    assert validate_safe_output_path(safe_child, allowed_root=tmp_path) == safe_child

    with pytest.raises(ConfigurationError, match="escapes restricted workspace root"):
        validate_safe_output_path(tmp_path.parent / "outside.json", allowed_root=tmp_path)


def test_bare_auth_scheme_words_in_prose_are_not_redacted() -> None:
    """Regression: ordinary prose such as 'bearer tokens' must survive; real credentials must not."""
    from openbench.security import sanitize_error_message

    assert sanitize_error_message("Never log API keys or bearer tokens.") == (
        "Never log API keys or bearer tokens."
    )
    assert sanitize_error_message("This covers basic refusal patterns.") == (
        "This covers basic refusal patterns."
    )
    assert "[REDACTED]" in sanitize_error_message("Authorization: Bearer abcdef")
    assert "[REDACTED]" in sanitize_error_message("Bearer eyJhbGciOiJIUzI1NiJ9.e30.sig")
    assert "[REDACTED]" in sanitize_error_message("header was Basic dXNlcjpwYXNz")
