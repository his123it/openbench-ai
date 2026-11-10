#!/usr/bin/env python3
"""Generate docs/cli_reference.md and docs/api_reference.md from the code (v0.4.0).

Run from the repository root:  python scripts/generate_reference_docs.py
The documentation tests check that the committed files are up to date.
"""

from __future__ import annotations

import argparse
import inspect
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import openbench  # noqa: E402
from openbench.cli import build_parser  # noqa: E402

V040_COMMANDS = {"analyze", "preferences", "preferences create", "preferences validate", "preferences analyze"}


def _actions(parser: argparse.ArgumentParser) -> list[argparse.Action]:
    return [a for a in parser._actions if not isinstance(a, (argparse._HelpAction, argparse._SubParsersAction))]  # noqa: SLF001


def _walk(parser: argparse.ArgumentParser, prefix: str = ""):
    yield prefix, parser
    for action in parser._actions:  # noqa: SLF001
        if isinstance(action, argparse._SubParsersAction):  # noqa: SLF001
            helps = {c.dest: c.help for c in action._choices_actions}  # noqa: SLF001
            for name, sub in action.choices.items():
                sub._ob_help = helps.get(name)  # type: ignore[attr-defined]
                yield from _walk(sub, f"{prefix} {name}".strip())


def cli_reference() -> str:
    lines = [
        "# CLI reference (v0.4.0)",
        "",
        "Generated from the argument parser by `scripts/generate_reference_docs.py`; do not edit by hand.",
        "Flags marked *(v0.4.0)* are new; every v0.3.1 command, flag, default and choice is unchanged.",
        "",
    ]
    for path, parser in _walk(build_parser()):
        if not path:
            continue
        actions = _actions(parser)
        if not actions and any(isinstance(a, argparse._SubParsersAction) for a in parser._actions):  # noqa: SLF001
            lines += [f"## `openbench {path}`", "", getattr(parser, "_ob_help", "") or "", ""]
            continue
        lines += [f"## `openbench {path}`" + (" *(v0.4.0)*" if path in V040_COMMANDS else ""), ""]
        if getattr(parser, "_ob_help", None):
            lines += [parser._ob_help, ""]  # type: ignore[attr-defined]
        lines += ["| Argument | Required | Default | Choices | Help |", "| :--- | :---: | :--- | :--- | :--- |"]
        for a in actions:
            name = ", ".join(a.option_strings) if a.option_strings else a.dest
            default = "" if a.default in (None, argparse.SUPPRESS) or a.default is False else f"`{a.default}`"
            choices = ", ".join(f"`{c}`" for c in a.choices) if a.choices else ""
            help_text = (a.help or "").replace("|", "\\|").replace("\n", " ")
            if help_text.startswith("v0.4.0: "):
                help_text = "*(v0.4.0)* " + help_text[len("v0.4.0: "):]
            req = "yes" if (a.required or not a.option_strings) else ""
            lines.append(f"| `{name}` | {req} | {default} | {choices} | {help_text} |")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def api_reference() -> str:
    lines = [
        "# Python API reference (v0.4.0)",
        "",
        "Generated from `openbench.__all__` and `openbench.stats.__all__` by `scripts/generate_reference_docs.py`; do not edit by hand.",
        "The first 70 names are the unchanged v0.3.1 surface; v0.4.0 only appends names and appends optional parameters.",
        "",
        "## `openbench`",
        "",
        "| Name | Kind | Module | Summary |",
        "| :--- | :--- | :--- | :--- |",
    ]

    def row(name, obj):
        kind = "class" if inspect.isclass(obj) else ("function" if callable(obj) else type(obj).__name__)
        module = getattr(obj, "__module__", "") or ""
        doc = (inspect.getdoc(obj) or "").split("\n\n")[0].replace("\n", " ").replace("|", "\\|")
        if not callable(obj) and not inspect.isclass(obj):
            doc = ""
        return f"| `{name}`{' *(v0.4.0)*' if name in openbench.__all__[70:] else ''} | {kind} | `{module}` | {doc[:160]} |"

    for name in openbench.__all__:
        lines.append(row(name, getattr(openbench, name)))
    import openbench.stats as stats

    lines += ["", "## `openbench.stats`", "", "| Name | Kind | Module | Summary |", "| :--- | :--- | :--- | :--- |"]
    for name in stats.__all__:
        obj = getattr(stats, name)
        kind = "class" if inspect.isclass(obj) else ("function" if callable(obj) else type(obj).__name__)
        doc = (inspect.getdoc(obj) or "").split("\n\n")[0].replace("\n", " ").replace("|", "\\|") if (callable(obj) or inspect.isclass(obj)) else ""
        lines.append(f"| `{name}` | {kind} | `{getattr(obj, '__module__', '')}` | {doc[:160]} |")
    lines += [
        "",
        "## Other v0.4.0 modules",
        "",
        "- `openbench.lineage`: `RunLineage`, ID validation, seed status, record content hashes.",
        "- `openbench.paired`: `ComparisonArm`, `ComparisonPolicy`, `CompatibilityReport`, `PairedComparisonEngine`, `check_compatibility`.",
        "- `openbench.preferences`: task builder, key loader, annotation importer, analyzer, `cohens_kappa`, `fleiss_kappa`, `csv_safe`.",
        "- `openbench.quality`: `DatasetAuditor`, `ExperimentAuditor`, `AuditFinding`, `AuditReport`, normalisation and shingling helpers.",
        "- `openbench.evalstats`: `compute_evaluation_statistics` (the `evaluate --confidence-intervals` block).",
        "- `openbench.reporting_v040`: JSON, Markdown and CSV renderers; `openbench.provenance`: provenance blocks.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    (ROOT / "docs" / "cli_reference.md").write_text(cli_reference(), encoding="utf-8")
    (ROOT / "docs" / "api_reference.md").write_text(api_reference(), encoding="utf-8")
    print("Wrote docs/cli_reference.md and docs/api_reference.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
