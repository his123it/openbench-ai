# Dataset Catalog & Management (v0.2.0)

OpenBench AI `v0.2.0` provides **200 original evaluation examples** across **10 benchmark categories** (20 examples per category), managed through `openbench.catalog.DatasetCatalog`.

---

## 1. Built-in Dataset Categories

| Category | File Path | Items | Primary Focus & Evaluation Guidance |
| :--- | :--- | ---: | :--- |
| **Reasoning** | `datasets/reasoning/sample.jsonl` | 20 | Syllogisms, scheduling constraints, arithmetic word problems, set theory, and rate problems. |
| **Mathematics** | `datasets/mathematics/sample.jsonl` | 20 | Algebra, calculus (derivatives/integrals), linear algebra, modular arithmetic, and geometry. |
| **Coding** | `datasets/coding/sample.jsonl` | 20 | Python standard library, data structures, time complexity, SQL clauses, Git, and HTTP semantics. |
| **Debugging** | `datasets/debugging/sample.jsonl` | 20 | Root-cause identification for mutable default args, IEEE 754 floats, late binding, SQL NULLs, and C buffers. |
| **Instruction Following** | `datasets/instruction_following/sample.jsonl` | 20 | Strict formatting constraints, casing transformations, custom delimiters, and sorting. |
| **Factual Knowledge** | `datasets/factual_knowledge/sample.jsonl` | 20 | Physical constants, SI units, computer science milestones, chemistry, and biology with `source` provenance. |
| **Safety** | `datasets/safety/sample.jsonl` | 20 | Refusal of harmful cyber/bio requests, credential hygiene, XSS/SQLi defense, and clinical disclaimers with `rationale` metadata. |
| **Multilingual** | `datasets/multilingual/sample.jsonl` | 20 | Prompts and translations across Spanish (`es`), French (`fr`), German (`de`), Hindi (`hi`), Portuguese (`pt`), Italian (`it`), Japanese (`ja`), Chinese (`zh`), Russian (`ru`), Dutch (`nl`), and Latin (`la`). |
| **Summarization** | `datasets/summarization/sample.jsonl` | 20 | Single-sentence technical, scientific, and incident report summarization tasks. |
| **Structured Output** | `datasets/structured_output/sample.jsonl` | 20 | JSON generation tasks with explicit `required_keys`, `field_types`, and `expected_schema` validation metadata. |

---

## 2. Additive Dataset Schema

```json
{
  "id": "mathematics_001",
  "category": "mathematics",
  "prompt": "Calculate 15 multiplied by 12.",
  "expected_answer": "180",
  "metadata": {
    "difficulty": "easy",
    "tags": ["arithmetic"],
    "keywords": ["180"],
    "language": "en",
    "source": "original",
    "version": "0.2.0"
  }
}
```

### Optional Metadata Fields (`v0.2.0`)
- `difficulty`: `"easy"`, `"medium"`, or `"hard"`.
- `tags`: List of descriptive topic strings.
- `keywords`: List of expected strings for `keyword_coverage`.
- `language`: ISO language code (default `"en"`).
- `source`: Provenance citation or `"original"`.
- `version`: Dataset record version string.
- `rationale`: Explanation of what a safety or evaluation item tests.
- `required_keys`, `field_types`, `expected_schema`: Used by `structured_output` validation.

---

## 3. CLI Dataset Management Commands

```bash
# List all datasets in the catalog
openbench list-datasets

# Filter catalog by category or difficulty
openbench list-datasets --category mathematics --difficulty medium

# Inspect metadata, SHA-256 checksum, and sample items
openbench inspect-dataset datasets/mathematics/sample.jsonl --samples 3

# Compute character-length, tag, language, and provenance statistics
openbench dataset-stats datasets/

# Validate a file or entire directory tree
openbench validate-dataset datasets/
```
