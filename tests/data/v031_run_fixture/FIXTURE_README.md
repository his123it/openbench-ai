# v0.3.1 run-directory fixture

Two experiment runs written by the unmodified OpenBench AI v0.3.1 code (commit
`8d3295abbabeff4a0c20052a1bbb70fed48c424e`) with the mock provider, on
`datasets/reasoning/sample.jsonl` (paths are relative to a working directory that contains
`datasets/`). They have `schema_version` `0.3.0` and no lineage.

- `run_v031fixturea`: `fixture-model-a`, all expected answers, status `completed`.
- `run_v031fixtureb`: `fixture-model-b`, three changed answers and one failed item
  (`reasoning_010`), status `partial`.

They are used to test that v0.4.0 reads v0.3.1 artifacts without modifying them. Tests copy
the directory before use; do not edit these files (their checksums are recorded in
`reproducibility.json`).
