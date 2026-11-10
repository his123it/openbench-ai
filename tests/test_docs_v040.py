"""Documentation and example checks for the v0.4.0 release candidate (Stage 7)."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import openbench

ROOT = Path(__file__).resolve().parents[1]
DOCS = [ROOT / "README.md", ROOT / "CHANGELOG.md", ROOT / "V0_4_0_MIGRATION_GUIDE.md", ROOT / "CONTRIBUTING.md",
        ROOT / "SECURITY.md", *sorted((ROOT / "docs").glob("*.md"))]
LINK = re.compile(r"\]\(([^)#\s]+)(?:#[^)]*)?\)")


def test_generated_reference_docs_are_current():
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        import generate_reference_docs as gen
    finally:
        sys.path.pop(0)
    assert (ROOT / "docs" / "cli_reference.md").read_text(encoding="utf-8") == gen.cli_reference()
    assert (ROOT / "docs" / "api_reference.md").read_text(encoding="utf-8") == gen.api_reference()


def test_relative_links_resolve():
    broken = []
    for doc in DOCS:
        for target in LINK.findall(doc.read_text(encoding="utf-8")):
            if target.startswith(("http://", "https://", "mailto:")):
                continue
            if not (doc.parent / target).exists():
                broken.append(f"{doc.relative_to(ROOT)} -> {target}")
    assert broken == []


def test_versions_are_consistent():
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'version = "0.4.0"' in pyproject
    assert openbench.__version__ == "0.4.0"
    assert 'version: "0.4.0"' in (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    assert "## [0.4.0]" in (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert "# OpenBench AI (v0.4.0)" in (ROOT / "README.md").read_text(encoding="utf-8")


def test_project_wording_and_authorship_preserved():
    wording = "OpenBench AI is maintained by its project owner, and its initial development was AI-assisted."
    for name in ("README.md", "CONTRIBUTING.md"):
        assert wording in (ROOT / name).read_text(encoding="utf-8")
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert '{ name = "Lakshay Chhaba" }' in pyproject
    combined = "\n".join(p.read_text(encoding="utf-8") for p in DOCS).lower()
    for phrase in ("college project", "student project", "class project"):
        assert phrase not in combined


def test_docs_make_no_overclaims():
    combined = "\n".join(p.read_text(encoding="utf-8") for p in DOCS)
    for phrase in ("100% backward compatib", "fully backward compatible", "guarantees deterministic",
                   "semantic duplicate detection is", "proves correctness", "scientifically validated"):
        assert phrase.lower() not in combined.lower(), phrase


def test_v040_workflow_example_runs(tmp_path):
    res = subprocess.run([sys.executable, str(ROOT / "examples" / "v040_workflow.py"), str(tmp_path / "wf")],
                         capture_output=True, text=True, cwd=tmp_path, timeout=600,
                         env={"PYTHONPATH": str(ROOT / "src"), "PATH": ""})
    assert res.returncode == 0, res.stderr[-2000:]
    for name in ("dataset_audit.md", "comparison.md", "comparison.json", "variability.md", "preferences.md"):
        assert (tmp_path / "wf" / name).exists(), name
    tasks = (tmp_path / "wf" / "preference_tasks" / "tasks.jsonl").read_text()
    assert "example-model-a" not in tasks and "example-model-b" not in tasks


def test_example_profiles_load():
    import pytest

    pytest.importorskip("yaml")
    from openbench import ProfileLoader

    for path in sorted((ROOT / "configs").glob("*.yaml")):
        ProfileLoader.load_file(path)
