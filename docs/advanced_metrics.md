# Advanced Evaluation Metrics Reference (v0.2.0)

OpenBench AI `v0.2.0` expands the metric suite from 3 baseline metrics to **12 registered metric identifiers**.

---

## 1. Precision (`precision`), Recall (`recall`), and F1 (`f1`)

### Task Modes & Derivation of TP, FP, FN, TN
OpenBench AI never treats arbitrary free-form text similarity as classification precision/recall. Instead, `extract_decision_elements()` in `openbench.metrics.precision` operates on two well-defined deterministic task structures:

1. **Binary Classification Decisions (`mode="binary_classification"`)**:
   - Triggered when `record.metadata["task_type"] == "binary_classification"` or when the normalized expected answer is a canonical binary label (`yes`/`no`, `true`/`false`, `positive`/`negative`, `safe`/`unsafe`, `valid`/`invalid`).
   - Positive class set: `{"yes", "true", "positive", "1", "safe", "valid", "correct", "pass"}` (or custom `metadata["positive_label"]`).
   - Negative class set: `{"no", "false", "negative", "0", "unsafe", "invalid", "incorrect", "fail"}`.
   - $\text{TP}=1$ if both expected and predicted are positive; $\text{FP}=1$ if expected is negative and predicted is positive; $\text{FN}=1$ if expected is positive and predicted is not positive; $\text{TN}=1$ if both expected and predicted are negative.
2. **Discrete Element-Set Prediction (`mode="element_set"`)**:
   - Triggered for delimited lists (`blue, green, red`, `42;29;14;8;3`, `alpha|beta|gamma`), explicit `metadata["expected_elements"]`, or token sets.
   - Let $E$ be the set of normalized expected elements and $P$ be the set of normalized predicted elements:
     $$\text{TP} = |P \cap E|, \quad \text{FP} = |P \setminus E|, \quad \text{FN} = |E \setminus P|$$
   - Formulas:
     $$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}, \quad \text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}, \quad \text{F1} = \frac{2 \cdot \text{TP}}{2 \cdot \text{TP} + \text{FP} + \text{FN}}$$
   - Aggregate results report both macro-averaged item scores (`score`) and corpus-wide micro-averaged scores (`micro_precision`, `micro_recall`, `micro_f1`).

---

## 2. ROUGE (`rouge`, `rouge_1`, `rouge_l`)

- **Tokenization**: Applies `NormalizationConfig` followed by Unicode word tokenization (`re.findall(r"[\w]+", text)`).
- **`rouge_1`**: Unigram multiset overlap precision, recall, and F1 between candidate and reference tokens.
- **`rouge_l`** (default for `rouge`): Longest Common Subsequence (LCS) precision, recall, and F1, capturing sentence-level word order without requiring contiguous matches.
- **Limitations**: Measures lexical overlap against a single reference; cannot detect factual hallucinations or antonym swaps.

---

## 3. BLEU (`bleu`)

- **N-gram Behavior**: Computes clipped $n$-gram precisions for $n \in \{1, \dots, N\}$ where $N = \min(\text{max\_n}, |\text{ref}|, |\text{cand}|)$ (default $\text{max\_n}=4$).
- **Smoothing**: Applies add-$\epsilon$ smoothing ($\epsilon = 0.1$) to higher-order $n$-grams ($n > 1$) when unigram overlap is non-zero, preventing single-sentence scores from collapsing to zero on short references.
- **Brevity Penalty (BP)**:
  $$\text{BP} = \begin{cases} 1.0 & \text{if } c > r \\ \exp(1 - r / c) & \text{if } c \le r \end{cases}$$
- **Limitations**: Penalizes valid synonyms and paraphrases that diverge from the exact reference wording.

---

## 4. Semantic Similarity (`semantic_similarity`)

- **Adapter Architecture**: Accepts any `EmbeddingProvider` implementation:
  - `LocalNgramEmbeddingProvider` (default): Zero-dependency, deterministic SHA-256 word + character 3-gram feature hashing into a normalized vector space.
  - `CallableEmbeddingProvider`: Wraps any user-supplied function `fn(text: str) -> list[float]`.
  - `HuggingFaceEmbeddingProvider`: Optional adapter for `sentence-transformers` models (defaults to `local_files_only=True` so weights are never downloaded without explicit user consent).
- **Methodological Warning**:
  > **Semantic similarity is NOT factual correctness.** Vector cosine similarity measures topical and lexical proximity; a response stating the opposite of the reference answer often has high cosine similarity.

---

## 5. Structured Output Validation (`structured_output`)

- **Capabilities**:
  1. Verifies `json.loads()` syntactic validity (`valid_json`).
  2. Checks presence of all `metadata["required_keys"]` (or keys inferred from a JSON reference answer).
  3. Checks Python/JSON types specified in `metadata["field_types"]`.
  4. Validates against `metadata["expected_schema"]` using `jsonschema` when installed.
- **Separation of Structure vs. Semantics**:
  Reports `valid_structure` (`structure_score`) separately from `value_match` (`value_match_rate`). By default (`require_value_match=False`), an item passes if its JSON structure and schema are 100% valid even if specific field values differ.
