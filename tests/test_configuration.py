"""Unit tests for YAML configuration profiles and validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from openbench.configuration import ProfileLoader
from openbench.exceptions import ConfigurationError


def test_load_builtin_yaml_profiles() -> None:
    """All three built-in YAML profiles (reasoning, coding, instruction_following) load and validate."""
    repo_root = Path(__file__).resolve().parent.parent
    configs_dir = repo_root / "configs"

    reasoning = ProfileLoader.load_file(configs_dir / "reasoning.yaml")
    assert reasoning.name == "reasoning-profile"
    assert "semantic_similarity" in reasoning.metrics
    assert "csv" in reasoning.output_formats

    coding = ProfileLoader.load_file(configs_dir / "coding.yaml")
    assert coding.name == "coding-profile"
    assert "f1" in coding.metrics

    inst = ProfileLoader.load_file(configs_dir / "instruction_following.yaml")
    assert inst.normalization.case_sensitive is True
    assert inst.strict_unknown_ids is True


def test_profile_rejects_unknown_fields_and_invalid_metrics(tmp_path: Path) -> None:
    """ProfileLoader never silently ignores unknown fields or invalid metric names."""
    bad_field_yaml = tmp_path / "unknown_field.yaml"
    bad_field_yaml.write_text(
        "name: test-prof\nunknown_section: 123\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigurationError, match="Unknown top-level configuration field"):
        ProfileLoader.load_file(bad_field_yaml)

    bad_metric_yaml = tmp_path / "bad_metric.yaml"
    bad_metric_yaml.write_text(
        "name: test-prof\nmetrics:\n  - non_existent_metric\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigurationError, match="Unknown metric"):
        ProfileLoader.load_file(bad_metric_yaml)
