# OpenBench AI v0.2.0 Migration Guide

Upgrading from **OpenBench AI v0.1.0** to **v0.2.0** requires **zero breaking code or CLI changes**. All `v0.1.0` scripts, imports, datasets, response files, and CLI invocations work unmodified.

---

## 1. Upgrading Your Environment

```bash
# Core upgrade (zero third-party runtime dependencies)
python -m pip install -e .

# Recommended upgrade with YAML configuration and JSON Schema support
python -m pip install -e ".[dev,config,metrics]"
```

---

## 2. What Stays Exactly the Same

- **Imports**: `from openbench import DatasetLoader, Evaluator, ModelComparator, ReportGenerator` works identically.
- **CLI Commands**: `openbench validate-dataset`, `openbench evaluate`, `openbench compare`, and `openbench --version` retain all existing flags and exit codes.
- **Built-in Datasets**: All 48 original records in `datasets/reasoning/sample.jsonl`, `datasets/coding/sample.jsonl`, `datasets/instruction_following/sample.jsonl`, and `datasets/safety/sample.jsonl` retain their exact IDs (`reasoning_001`–`012`, etc.). Note that each of those 4 files now includes 8 additional items (`_013`–`_020`) bringing each category to 20 items.
  - **Score impact**: a `v0.1.0` response file that only answers the original 12 IDs will report the 8 new items as missing, and missing items count as `0.0` in aggregate scores. For like-for-like comparison with earlier results, use `evaluated_only_score` in each metric's `details`, add responses for the new IDs, or evaluate against the `v0.1.0` copy of the dataset.

---

## 3. Adopting New v0.2.0 Features Incrementally

### A. Filter Datasets to Specific Categories or Difficulties
```python
from openbench.dataset import DatasetLoader

easy_math = DatasetLoader.load(
    "datasets/mathematics/sample.jsonl",
    difficulties=["easy"],
)
```

### B. Enable Advanced Metrics
Add any of `precision`, `recall`, `f1`, `rouge`, `rouge_1`, `rouge_l`, `bleu`, `semantic_similarity`, or `structured_output` to `EvaluationConfig(metric_names=[...])` or `--metrics`:

```bash
openbench evaluate   --dataset datasets/summarization/sample.jsonl   --responses my_summaries.json   --metrics rouge_1 rouge_l bleu semantic_similarity   --output results/
```

### C. Switch to Declarative YAML Profiles
Instead of passing long `--metrics` lists on the command line, point `openbench evaluate` at a YAML profile:

```bash
openbench evaluate   --config configs/reasoning.yaml   --responses examples/sample_responses.json   --output results/
```

### D. Export CSV Reports & Track Run History
```bash
openbench export   --input results/alphabaseline-7b.json   --format csv   --mode individual   --output results/alphabaseline-7b.csv

openbench history list --store results/history.jsonl
```
