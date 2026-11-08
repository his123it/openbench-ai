## Summary
Briefly describe the changes introduced by this pull request.

## Related Issue
Fixes # (if applicable)

## Verification Checklist
- [ ] `python -m pip install -e ".[dev]"` installs cleanly
- [ ] `pytest` passes with new unit tests covering the changes
- [ ] `ruff check src tests examples scripts` passes with zero warnings
- [ ] `python scripts/validate_datasets.py` passes (if datasets were added or modified)
- [ ] Documentation updated (`README.md` or `docs/`) to reflect new functionality and limitations
