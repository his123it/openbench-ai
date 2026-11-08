# Advanced Evaluation Metrics Reference (v0.3.1)

OpenBench AI `v0.3.1` provides **13 registered metric identifiers** across baseline and advanced deterministic/embedding metrics.

---

## 1. Precision (`precision`), Recall (`recall`), and F1 (`f1`)

Operates on two deterministic task structures via `extract_decision_elements()`:
1. **Binary Classification Decisions (`mode="binary_classification"`)**: Triggered when `metadata["task_type"] == "binary_classification"` or when the normalized expected answer is a canonical binary label (`yes`/`no`, `true`/`false`, `positive`/`negative`, `safe`/`unsafe`, `valid`/`invalid`).
2. **Discrete Element-Set Prediction (`mode="element_set"`)**: Triggered for delimited lists (`blue, green, red`), explicit `metadata["expected_elements"]`, or token sets:
   $$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}, \quad \text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}, \quad \text{F1} = \frac{2 \cdot \text{TP}}{2 \cdot \text{TP} + \text{FP} + \text{FN}}$$

---

## 2. ROUGE (`rouge`, `rouge_1`, `rouge_l`) & BLEU (`bleu`)

- **`rouge_1`**: Unigram multiset overlap precision, recall, and F1.
- **`rouge_l`** (default for `rouge`): Longest Common Subsequence (LCS) precision, recall, and F1.
- **`bleu`**: Smoothed sentence BLEU with clipped $n$-gram precisions (up to 4-grams) and exponential brevity penalty.

---

## 3. Lexical Similarity (`lexical_similarity`) vs. Semantic Similarity (`semantic_similarity`)

### Important Methodological Distinction (v0.3.1)
- **`LocalNgramEmbeddingProvider` (Default Provider)**: Computes SHA-256 feature hashes over word unigrams and character 3-grams (`is_semantic_embedding = False`). Because this measures surface lexical overlap rather than semantic comprehension, results generated with the default provider are explicitly labeled `similarity_type: "lexical_similarity"` in item and aggregate report details.
- **`lexical_similarity` Metric Identifier**: Registered as a first-class metric identifier in `v0.3.1` for surface n-gram cosine similarity.
- **`semantic_similarity` Metric Identifier**: Retained for backward compatibility. When used with the default `LocalNgramEmbeddingProvider`, it acts as a documented legacy alias for `lexical_similarity` (`similarity_type: "lexical_similarity"`). To require a genuine neural embedding model (`is_semantic_embedding = True`, such as `HuggingFaceEmbeddingProvider`), pass `require_semantic_provider=True`.
- **Limitation**: Neither `lexical_similarity` nor embedding-based `semantic_similarity` understands logical negation or proves factual correctness.

---

## 4. Structured Output Validation (`structured_output`)

- Validates JSON parsing (`valid_json`), required keys (`required_keys`), field types (`field_types`), and optional `jsonschema` compliance (`expected_schema`).
- Reports structural validity (`valid_structure`, `structure_score`) separately from value accuracy (`value_match`).
- **Optional Fenced JSON Stripping (`strip_markdown_fence`, New in v0.3.1)**:
  - Defaults to `False` (preserving v0.2.0 behavior).
  - When `strip_markdown_fence=True`, removes at most one valid surrounding Markdown code fence (```` ```json ... ``` ```` or ```` ``` ... ``` ````) before JSON parsing. It never strips arbitrary surrounding conversational prose, never concatenates multiple code fences, and never repairs malformed JSON.
- Without the optional `jsonschema` package, schema validation is skipped and reported as `schema_validated: false`.
- With the default `strip_markdown_fence=False`, responses wrapped in Markdown code fences fail JSON parsing.
