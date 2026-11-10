# Dataset and evaluation quality audits (v0.4.0)

Audits are **read-only**. They never rewrite, delete, reorder or "fix" records, never change scores, and never hide a finding. Exit code 1 means at least one *error* finding; warnings and info exit 0.

```bash
openbench validate-dataset datasets/ --quality-checks --format markdown --output dataset_audit.md
openbench validate-dataset my.jsonl --quality-checks --metrics exact_match f1 --near-duplicate-threshold 0.9
openbench runs verify run_<id> --quality-checks          # or grp_<id>
```

Without `--quality-checks`, `validate-dataset` behaves exactly as in v0.3.1.

Each finding has a stable check ID, a severity (`error`, `warning` or `info`), a title, an explanation, the affected record or run IDs, a suggested remediation, details and, for reviewed records, a classification.

## Dataset checks

| Check | Severity | Rule |
| --- | --- | --- |
| DQ001 | error | Duplicate record ID, including across files. |
| DQ002 | warning | Exact duplicate prompt after normalisation. |
| DQ003 | warning | Possible near-duplicate prompt (lexical heuristic). |
| DQ004 | error | Prompt missing, empty or not a string. |
| DQ005 | error | `expected_answer` missing or not a string. |
| DQ006 | warning | Reference empty or a placeholder (`todo`, `tbd`, `n/a`, `none`, `...`). Short answers such as `2` are fine. |
| DQ007 | warning | A `metadata.keywords` entry does not appear in the reference, so the reference itself does not score 1.0 on `keyword_coverage`. |
| DQ008 | error | Invalid metric configuration (with `--metrics`), or an `expected_schema` that is not an object. |
| DQ009a | warning | Several version labels in one file. |
| DQ009b | info | A file mixes records with and without `metadata.version`. |
| DQ009c | warning | Unexpected difficulty label. |
| DQ009d | warning | Category differs from the directory name. |
| DQ010 | info | Reviewed record: an accepted answer variant cannot be represented by keyword lists. |
| DQ011 | warning | Reviewed record: an ambiguous item with more than one valid answer. |
| DQ012 | error | Malformed structure: invalid JSON, a line that is not an object, a missing ID or category, metadata that is not an object, keywords that are not a list of strings, a file that is not UTF-8, or a file that is too large. |
| DQ013 | warning below 10 records, info below 30 | Too few records for statistical conclusions. |
| DQ014 | info | The near-duplicate check was skipped: more than 5,000 records, or prompts longer than 20,000 characters. A skip is always reported, never silent. |

**Near-duplicates (DQ003)** are a lexical heuristic:

- **Normalisation:** Unicode NFKC, then case folding, then whitespace collapsing. Punctuation and accents are kept.
- **Similarity:** character 5-gram Jaccard similarity, with a default threshold of 0.85.
- **Short prompts:** prompts under 5 characters are excluded from DQ003 but still checked for exact duplicates.

This is **not semantic duplicate detection**, and a high score is not proof of duplication. Review flagged pairs manually.

**Review notes.** `src/openbench/quality_notes.json` holds reviewed notes for built-in records.

- Notes are keyed by record ID and the SHA-256 of the record's full content, so they never apply to edited content or to a same-ID record elsewhere.
- A note can only attach a classification to a DQ007 finding, or add a DQ010 or DQ011 finding.
- Notes never suppress or alter heuristic findings; a test checks this.

## The built-in datasets

The audit of the 200 built-in records gives **0 errors, 6 warnings and 8 info**:

- **DQ007, documented historical exceptions** unchanged since v0.1.0:
  - `coding_003`, `coding_007`: explanatory "constant" and "logarithmic" keywords for Big-O answers;
  - `coding_010`: a keyword that echoes the prompt;
  - `reasoning_004`: a keyword that conflicts with the "answer Yes or No" instruction;
  - `reasoning_012`: explanatory "3" and "5".
- **DQ011:** `multilingual_006`. Both "warm" and "heiß" are valid opposites of "kalt", but only "warm" scores.
- **DQ009b:** four files (`coding`, `instruction_following`, `reasoning`, `safety`) each mix 12 unversioned v0.1.0 records with versioned ones.
- **DQ010, accepted variants that keyword lists cannot express:** `coding_019` ("BFS"), `factual_005` ("mitochondria"), `factual_007` ("User Datagram Protocol"), `factual_018` ("Godel").

No duplicates or near-duplicates were found. The highest prompt similarity is 0.535, between `coding_001` and `coding_014`.

The defects corrected in v0.2.0 (`summarization_004`, `summarization_017`) stay corrected and are not flagged. No dataset record changed in v0.4.0. Encoding alternative answers properly would need an "any-of" keyword schema, which is deferred.

## Experiment checks

| Check | Severity | Rule |
| --- | --- | --- |
| EQ001 | warning | Missing model outputs. |
| EQ002 | warning | Run status `partial` or `failed`. |
| EQ003 | error | Integrity verification failed: missing artifacts or provenance, a checksum mismatch, or inconsistent metadata or lineage. |
| EQ004 | warning | A repeat group has fewer completed repetitions than planned. |
| EQ005 | warning below 10, info below 30 evaluated items | Insufficient data for statistical analysis. |
| EQ006 | error | Duplicate record IDs in `responses.json`. |
| EQ007 | warning | Response IDs that match no dataset record. |
| EQ008 | error | Metric results that are non-finite or outside [0, 1]. |
| EQ009 | info | A pre-v0.4.0 run without lineage. |
| EQ010 | error | Inconsistent repeat-group lineage (duplicate indices, disagreeing counts or configuration). |
| EQ011 | error | Unreadable configuration or evaluation. |
| EQ012 | info | Responses not generated by the execution engine (external or imported). |
