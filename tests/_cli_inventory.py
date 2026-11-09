"""Helper: extract a stable inventory of the CLI (commands, flags, defaults, choices)."""

from __future__ import annotations

import argparse
from typing import Any


def _actions(parser: argparse.ArgumentParser) -> list[dict[str, Any]]:
    out = []
    for action in parser._actions:  # noqa: SLF001 - argparse has no public API for this
        if isinstance(action, (argparse._HelpAction, argparse._SubParsersAction)):  # noqa: SLF001
            continue
        out.append(
            {
                "dest": action.dest,
                "option_strings": sorted(action.option_strings),
                "required": bool(action.required),
                "nargs": None if action.nargs is None else str(action.nargs),
                "default": repr(action.default),
                "choices": None if action.choices is None else sorted(map(str, action.choices)),
            }
        )
    return sorted(out, key=lambda a: (a["option_strings"], a["dest"]))


def cli_inventory(parser: argparse.ArgumentParser, prefix: str = "") -> dict[str, Any]:
    """Return {command path: [action, ...]} for the parser and all sub-commands."""
    inventory: dict[str, Any] = {prefix or "<root>": _actions(parser)}
    for action in parser._actions:  # noqa: SLF001
        if isinstance(action, argparse._SubParsersAction):  # noqa: SLF001
            for name, sub in action.choices.items():
                path = f"{prefix} {name}".strip()
                inventory.update(cli_inventory(sub, path))
    return inventory
