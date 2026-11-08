# Model Execution Engine (v0.3.0)

OpenBench AI `v0.3.0` introduces a provider-independent **Model Execution Engine** (`openbench.execution.ExecutionEngine`) that automates prompt dispatch, concurrency, retries, timeouts, deterministic caching, and telemetry collection while keeping evaluation completely decoupled.

---

## 1. Architectural Separation: Execution vs. Evaluation

- **`ExecutionEngine` (`openbench.execution`)**: Takes a `Dataset` and a `BaseModelAdapter`, sends prompts to the provider, handles transport errors/retries/caching, records per-request latency and token usage, and outputs a standard `ModelResponseSet` + `PerformanceReport`.
- **`Evaluator` (`openbench.evaluator`)**: Takes a `Dataset` and a `ModelResponseSet` (either produced live by `ExecutionEngine` or loaded from a static `.json` file via `ResponseLoader`) and computes deterministic evaluation metrics.

Users can always evaluate pre-generated JSON response files with `openbench evaluate` without invoking any model provider.

---

## 2. Execution Configuration (`ExecutionConfig`)

```python
from openbench.adapters import MockModelProvider
from openbench.dataset import DatasetLoader
from openbench.execution import ExecutionConfig, ExecutionEngine, RetryPolicy

dataset = DatasetLoader.load("datasets/reasoning/sample.jsonl")
provider = MockModelProvider(model_name="mock-v1")

exec_config = ExecutionConfig(
    timeout_seconds=30.0,
    concurrency=4,
    use_cache=True,
    estimate_missing_tokens=False,
    privacy_mode=False,
    retry_policy=RetryPolicy(
        max_attempts=3,
        initial_backoff_seconds=0.1,
        backoff_multiplier=2.0,
        max_backoff_seconds=2.0,
        retry_on_timeouts=True,
    ),
)

engine = ExecutionEngine(provider=provider, config=exec_config)
outcome = engine.execute_dataset(dataset)
print(f"Completed: {len(outcome.completed_record_ids)}, Failed: {len(outcome.failed_record_ids)}")
```

---

## 3. CLI Execution (`openbench run`)

```bash
# Execute using a YAML configuration profile
openbench run --config configs/mock_experiment.yaml

# Execute using direct CLI flags
openbench run   --dataset datasets/reasoning/sample.jsonl   --provider mock   --model mock-deterministic-v1   --concurrency 2   --timeout 15.0   --max-attempts 3
```
