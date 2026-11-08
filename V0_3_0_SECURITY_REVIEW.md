# OpenBench AI v0.3.0 — Security and Privacy Review

This document summarizes the security threat model, implemented mitigations, privacy controls, and known operational boundaries of **OpenBench AI v0.3.0**.

---

## 1. Threat Model & Mitigations Implemented

| Threat / Risk Area | Mitigation Implemented in `v0.3.0` | Module & Test Verification |
| :--- | :--- | :--- |
| **Accidental Credential Leakage in Reports or Logs** | `redact_sensitive_data()` recursively replaces any dict key matching `key`, `token`, `secret`, `password`, `credential`, `auth`, or `bearer` with `"[REDACTED]"`. `sanitize_error_message()` scrubs `Bearer ...`, `sk-...`, `hf_...`, `ghp_...`, and `user:pass@` strings from exception logs. | `src/openbench/security.py`, `tests/test_security_and_cache.py` |
| **Sensitive Prompt Exposure in Execution Logs** | Configurable `privacy_mode: true` (`redact_prompt_if_private()`) replaces raw prompt text in `logs.jsonl` with a one-way SHA-256 digest (`[REDACTED_CONTENT sha256=... len=...]`). | `src/openbench/security.py`, `src/openbench/execution.py`, `tests/test_security_and_cache.py` |
| **Server-Side Request Forgery (SSRF) / Unsafe URL Schemes** | `validate_endpoint_url()` rejects `file://`, `ftp://`, `javascript:`, control characters, and embedded credentials, and blocks unencrypted `http://` to non-loopback hosts unless `allow_remote_http=True` is explicitly configured. | `src/openbench/security.py`, `src/openbench/adapters/ollama.py`, `tests/test_security_and_cache.py` |
| **Arbitrary Code Execution via Untrusted Model Outputs or Datasets** | OpenBench AI uses `json.loads()` and `yaml.safe_load()` exclusively; it never invokes `eval()`, `exec()`, `pickle.loads()`, or `subprocess(..., shell=True)` on dataset prompts or model responses. | All modules in `src/openbench/` |
| **Path Traversal in Output Directories** | `validate_safe_output_path()` rejects null bytes and verifies path containment within `allowed_root` when workspace confinement is enabled. | `src/openbench/security.py`, `tests/test_security_and_cache.py` |
| **Silent Benchmark Tampering or Cache Poisoning** | `RunIntegrityVerifier` validates SHA-256 checksums across `config.json`, `model_metadata.json`, `responses.json`, `evaluation.json`, and `performance.json`. `ResponseCache` keys entries by the full SHA-256 of model, revision, dataset, prompt, and generation config. | `src/openbench/integrity.py`, `src/openbench/cache.py`, `tests/test_experiments_and_integrity.py` |

---

## 2. Data Retention & Deletion Behavior

- OpenBench AI stores all data locally on the user's filesystem (`experiments/`, `results/`, `.openbench_cache/`) and transmits zero telemetry.
- Cached responses can be inspected with `openbench cache stats` and deleted at any time via `openbench cache clear`.
- Experiment run directories (`experiments/run_<id>/`) are self-contained and can be archived or deleted using standard filesystem tools.

---

## 3. Honest Security Boundaries

- OpenBench AI does not sandbox external third-party Python callables passed by a developer into `LocalCallableAdapter` or custom `BaseMetric` subclasses—developers are responsible for the safety of their own custom Python functions.
- Cryptographic SHA-256 checksums verify file integrity on disk after a run completes; they do not attest to hardware enclave execution or the factual accuracy of a model's output.
