# Human preference evaluation (v0.4.0)

A local-first workflow for blinded pairwise judgements of two models' responses. No hosted service is involved, and annotations and responses never leave your machine.

## 1. Create blind tasks

```bash
openbench preferences create --runs run_<a> run_<b> --seed 7 \
  --output tasks/ --key-output private/tasks.key.json [--rubric rubric.json] [--show-reference] [--csv]
```

You can also use `--results a.json b.json` with evaluation files.

- Eligible items are records with a valid response from both sources and an identical prompt, in sorted record-ID order. Excluded counts are recorded in the key.
- Each task's display order (A/B or B/A) is random but **reproducible from `--seed`**.
- **`tasks.jsonl` is the blind file** given to annotators. It holds the task ID, prompt, `response_1`, `response_2`, instructions, an optional rubric and, only with `--show-reference`, the reference answer. It contains no model name, version, provider, run ID, latency, token count, seed, record ID or display order. The task ID is a hash that does not reveal the order.
- **`tasks.key.json` is the unblinded key**: model identities, record IDs, categories, difficulties and display orders. Keep it away from annotators; `--key-output` lets you store it elsewhere. It records the SHA-256 of `tasks.jsonl`.
- A **leakage audit** warns, listing task IDs, if a model name, provider, run ID or non-trivial version appears in task text, for example a response saying "As model X...". Outputs are never edited.
- Existing task files are never overwritten unless you pass `--overwrite`.

Example `rubric.json`:

```json
{"criteria": [{"name": "helpfulness", "min": 1, "max": 5, "description": "How useful is the answer?"}],
 "instructions": "Prefer the more accurate answer; length alone is not a reason."}
```

## 2. Annotate

Annotators write one record per judgement, as JSONL or CSV:

```json
{"annotation_id": "r1-0001", "task_id": "3f2a...", "annotator_id": "rater_01", "outcome": "response_1",
 "rubric_scores": {"helpfulness": 4}, "rationale": "More precise.", "confidence": 0.8,
 "timestamp": "2026-11-10T10:00:00Z"}
```

| Field | Required | Values |
| --- | --- | --- |
| `annotation_id`, `task_id`, `annotator_id`, `outcome` | yes | `outcome` is one of `response_1`, `response_2`, `tie` or `neither` (both unacceptable). |
| `rubric_scores` | no | Numbers within each criterion's range. In CSV, use `rubric:<criterion>` columns. |
| `rationale` | no | Plain text, at most 5,000 characters. |
| `confidence` | no | A number in [0, 1]. |
| `timestamp` | no | ISO 8601. |
| `metadata` | no | A small object of scalar values. |
| `schema_version` | no | If present, must be `0.4.0`. |

Annotator IDs must be **pseudonymous** (`[A-Za-z0-9_.-]`, up to 64 characters). Email-like IDs are rejected or warned about.

## 3. Validate

```bash
openbench preferences validate --key private/tasks.key.json --annotations annotations.jsonl [--annotators roster.txt]
```

Imported files are treated as untrusted. The following are **errors**:

- unknown fields;
- an unknown task ID;
- a missing required field or an invalid outcome;
- a duplicate annotation ID, or the same annotator judging the same task twice;
- an out-of-range rubric score or an unknown rubric criterion;
- an over-long rationale, or an invalid timestamp, confidence or metadata value;
- malformed JSON, or a line that is not an object;
- an annotator missing from the roster (with `--annotators`);
- a schema-version mismatch.

Files over 50 MB are refused. Parsing uses only `json` and `csv`, and imported text is never executed.

## 4. Analyse

```bash
openbench preferences analyze --key private/tasks.key.json --annotations annotations.jsonl --elo --format markdown --output prefs.md
```

**Win rates**

- The **headline** win rate of A over B is **task-weighted**. Each task's value is the mean of its annotations (A win = 1, tie = 0.5, B win = 0), so every task counts once even if it has more annotators.
- The CI comes from a **bootstrap over tasks**. Annotations stay clustered within their task, so they are never treated as independent.
- The annotation-weighted rate is also reported.
- `neither` outcomes are counted but excluded from win rates. Tasks with only `neither` judgements, and unannotated tasks, are counted.

**Breakdowns and checks**

- Win rates are broken down by category and difficulty.
- Per-annotator counts and win rates are reported, with an `annotator_imbalance` warning when one annotator contributes more than half of the annotations.
- **Position bias**: the share of decisive choices that went to `response_1`, with a CI. A warning is given if the CI excludes 0.5.

**Agreement**

| Measure | Requirements and behaviour |
| --- | --- |
| Percent agreement | Over annotator pairs within each task. |
| Cohen's κ | Per annotator pair, over shared tasks. Needs at least 2 shared tasks; `null` if expected agreement is 1. |
| Fleiss' κ | Over tasks with the modal number of ratings (at least 2). Tasks with other counts are excluded and counted. |

- All measures are computed in **display space** (`response_1`/`response_2`) and **model space** (A/B).
- Observed agreement is identical in both spaces, because all annotators of a task see the same order. κ can differ, because relabelling changes the category marginals.
- κ is unstable with few shared tasks and is affected by imbalanced categories (the prevalence paradox).

**Elo (optional, `--elo`)**

- K = 32, initial rating 1,000, ties count 0.5, and `neither` is skipped.
- Ratings are updated in a deterministic order: by timestamp, then by annotation ID.
- *Elo ratings are relative to this pool of models, prompts and annotators, depend on update order, and are not absolute measures of answer quality.* They do not replace the win-rate statistics above.

Analyse several model pairs at once by passing several keys, for example `--key k_ab.json k_bc.json`; Elo then covers every model involved.

The analysis output is **unblinded**, because it joins annotations with the key. Do not share it with annotators who are still working.
