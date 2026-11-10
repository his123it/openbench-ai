# Security Policy

## Supported Versions

OpenBench AI is maintained by its project owner, and its initial development was AI-assisted. Security updates and maintenance fixes are applied to the latest release series (`0.4.x` once released; `0.3.x` until then).

| Version | Supported          |
| ------- | ------------------ |
| 0.4.x (release candidate) | :white_check_mark: |
| 0.3.x   | :white_check_mark: |
| < 0.3.0 | :x:                |

## Security Model of OpenBench AI

- **Offline-First Core Evaluation**: Evaluating static JSON response files (`openbench evaluate`) operates entirely on local files and makes zero network requests.
- **No Arbitrary Code Execution**: Model outputs and dataset prompts are treated as untrusted text or parsed with `json.loads()` / `yaml.safe_load()`. OpenBench AI never passes prompts or model responses to `eval()`, `exec()`, `pickle.loads()`, or shell execution.
- **Centralized Credential Redaction**: `openbench.security.redact_sensitive_data()` and `sanitize_error_message()` scrub sensitive keys, URL query parameters (`?token=...`, `&api_key=...`), `Authorization: Bearer ...` headers, inline credential assignments (`password=...`, `client_secret=...`), and token prefixes from serialized reports, experiment metadata, cache keys, and error logs while preserving live runtime credentials during provider requests.

- **v0.4.0 additions**:
  - CSV exports neutralise spreadsheet formula injection (`openbench.security.csv_safe`).
  - CLI error messages are scrubbed of credentials.
  - Run and group IDs are validated, so they never resolve outside the experiments directory.
  - Imported annotation files are untrusted input. They are parsed with `json` and `csv` only, unknown fields are rejected, and size limits apply.
  - Blind preference task files contain no model identities; the identity key is a separate file.
  - Analyses never call providers and never write into run directories.
- **Integrity scope**: SHA-256 checksums detect changes relative to recorded checksums. They do not prove that results are factually correct or authentic.

## Reporting a Vulnerability

If you discover a security vulnerability in OpenBench AI:

1. **Do not open a public GitHub issue** if the vulnerability could put users at risk.
2. **Use GitHub Private Vulnerability Reporting** on [https://github.com/lakshaychhaba0001-sys/openbench-ai](https://github.com/lakshaychhaba0001-sys/openbench-ai) or contact the project maintainer (**Lakshay Chhaba**) with the subject line `[SECURITY] OpenBench AI Vulnerability Report`.
3. Include:
   - Affected version and module (`security.py`, `cli.py`, etc.).
   - Minimal steps or input files required to reproduce the issue.
   - Potential impact and suggested remediation.
