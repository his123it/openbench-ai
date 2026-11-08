# Security Policy

## Supported Versions

OpenBench AI is maintained by its project owner, and its initial development was AI-assisted. Security updates and maintenance fixes are applied to the latest `0.3.x` release series.

| Version | Supported          |
| ------- | ------------------ |
| 0.3.x   | :white_check_mark: |
| < 0.3.0 | :x:                |

## Security Model of OpenBench AI

- **Offline-First Core Evaluation**: Evaluating static JSON response files (`openbench evaluate`) operates entirely on local files and makes zero network requests.
- **No Arbitrary Code Execution**: Model outputs and dataset prompts are treated as untrusted text or parsed with `json.loads()` / `yaml.safe_load()`. OpenBench AI never passes prompts or model responses to `eval()`, `exec()`, `pickle.loads()`, or shell execution.
- **Centralized Credential Redaction**: `openbench.security.redact_sensitive_data()` and `sanitize_error_message()` scrub sensitive keys, URL query parameters (`?token=...`, `&api_key=...`), `Authorization: Bearer ...` headers, inline credential assignments (`password=...`, `client_secret=...`), and token prefixes from serialized reports, experiment metadata, cache keys, and error logs while preserving live runtime credentials during provider requests.

## Reporting a Vulnerability

If you discover a security vulnerability in OpenBench AI:

1. **Do not open a public GitHub issue** if the vulnerability could put users at risk.
2. **Use GitHub Private Vulnerability Reporting** on [https://github.com/lakshaychhaba0001-sys/openbench-ai](https://github.com/lakshaychhaba0001-sys/openbench-ai) or contact the project maintainer (**Lakshay Chhaba**) with the subject line `[SECURITY] OpenBench AI Vulnerability Report`.
3. Include:
   - Affected version and module (`security.py`, `cli.py`, etc.).
   - Minimal steps or input files required to reproduce the issue.
   - Potential impact and suggested remediation.
