# Optional Model Integrations & Adapters (v0.2.0)

OpenBench AI keeps its core evaluation engine completely independent of external inference servers and paid APIs. When you want to generate responses programmatically rather than loading a static JSON file, use the `openbench.adapters` package.

---

## 1. Adapter Architecture (`BaseModelAdapter`)

All adapters inherit from `openbench.adapters.BaseModelAdapter` and implement:
- `generate_one(record: DatasetRecord) -> ResponseItem`
- `generate_responses(dataset: Dataset) -> ModelResponseSet`
- `reproducibility_metadata() -> dict[str, Any]`

---

## 2. Local Callable Adapter (`LocalCallableAdapter`)

Wraps any Python function `fn(prompt: str) -> str | dict` (such as a local `llama-cpp-python` model, an Ollama local socket client, or a custom heuristic baseline) and optionally measures wall-clock latency (`latency_ms`):

```python
from openbench.adapters import LocalCallableAdapter
from openbench.dataset import DatasetLoader
from openbench.evaluator import Evaluator

dataset = DatasetLoader.load("datasets/mathematics/sample.jsonl")

adapter = LocalCallableAdapter(
    model_name="MyLocalModel",
    model_version="0.1",
    generate_fn=lambda prompt: "180" if "15" in prompt else "0",
    measure_latency=True,
)

response_set = adapter.generate_responses(dataset)
result = Evaluator().evaluate(dataset, response_set)
```

---

## 3. Hugging Face Adapter (`HuggingFaceAdapter` — Experimental)

Wraps local Hugging Face `transformers` text-generation pipelines.

- **Optional Dependency**: Install via `python -m pip install -e ".[huggingface]"`.
- **Offline-First Safety**: Defaults to `local_files_only=True` so model weights are **never** downloaded automatically without explicit `allow_download=True`.
- **Zero Hardcoded Tokens**: Never uses hardcoded API keys or paid endpoints.
- **Experimental Status**: Marked `is_experimental = True` in `v0.2.0` because local GPU/CPU weight availability varies across user environments; unit tests validate the adapter deterministically using injected mock pipelines.
