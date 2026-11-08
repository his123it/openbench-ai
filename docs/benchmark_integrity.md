# Benchmark Integrity & Provenance Verification (v0.3.0)

OpenBench AI `v0.3.0` includes a cryptographic integrity and provenance verification system in `openbench.integrity.RunIntegrityVerifier`.

---

## 1. What `reproducibility.json` Records

Every experiment run records SHA-256 checksums for:
- **Dataset Content**: Canonical SHA-256 checksum of all dataset records (`dataset_checksum`).
- **Evaluation Configuration**: `evaluation_configuration_checksum`.
- **Generation Configuration**: `generation_configuration_checksum`.
- **On-Disk Artifacts**:
  - `config_sha256` (`config.json`)
  - `model_metadata_sha256` (`model_metadata.json`)
  - `responses_sha256` (`responses.json`)
  - `evaluation_sha256` (`evaluation.json`)
  - `performance_sha256` (`performance.json`)

---

## 2. Provenance Classification (`execution_origin`)

OpenBench AI enforces a strict distinction between three run origins:
1. **`"provider_executed"`**: Generated live by `ExecutionEngine` calling a model provider adapter.
2. **`"external_responses_evaluated"`**: Evaluated from an externally supplied JSON response file.
3. **`"manually_imported"`**: Imported from an external report.

External or imported evaluations are **never** labeled as `provider_executed`, and `MockModelProvider` runs are permanently flagged with `is_mock_provider: true`.

---

## 3. Running the Integrity Verifier

```bash
openbench runs verify <RUN_ID>
```

The verifier detects:
- Missing artifact files
- Modified or tampered `responses.json`, `evaluation.json`, `performance.json`, `config.json`, or `model_metadata.json`
- Metadata mismatches across files
- Unsupported provenance schema versions

> **Methodological Limitation**:
> A cryptographic SHA-256 checksum proves that benchmark files and configurations have not been altered since the run was recorded. It does **not** prove the factual truthfulness of a model's answers or guarantee that an external provider did not experience non-deterministic hardware drift.
