# Security Policy

## Supported Versions

OpenBench AI is currently in its initial baseline release (`v0.1.0`), developed and maintained as an educational open-source college project. Security fixes are applied to the latest `0.1.x` release series.

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Security Model of OpenBench AI v0.1.0

- **Zero External Network Calls**: OpenBench AI `v0.1.0` operates entirely on local JSONL dataset files and local JSON response files. It does not transmit prompts, responses, or telemetry over the network.
- **No Arbitrary Code Execution**: The `coding` dataset evaluates language model responses using reference-based deterministic metrics (`exact_match`, `accuracy`, `keyword_coverage`). It does **not** execute untrusted model-generated code via `eval()` or `subprocess`.

## Reporting a Vulnerability

If you discover a security vulnerability in OpenBench AI (such as path traversal in report output handling or unsafe deserialization):

1. **Do not open a public GitHub issue** if the vulnerability could put users at risk.
2. **Use GitHub Private Vulnerability Reporting** (if enabled on the repository) or contact the project maintainer directly via their GitHub profile contact address with the subject line `[SECURITY] OpenBench AI Vulnerability Report`.
3. Include:
   - Affected version and component (`dataset.py`, `cli.py`, etc.).
   - Minimal steps or input files required to reproduce the issue.
   - Potential impact and suggested remediation (if available).

The maintainer will acknowledge receipt within 7 days and coordinate a fix and disclosure timeline.
