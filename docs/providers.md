# Model Providers & Adapter Framework (v0.3.0)

OpenBench AI provides a unified adapter architecture in `openbench.adapters` so local inference servers, local Python functions, Hugging Face pipelines, and deterministic test mocks share a single contract.

---

## 1. Registered Providers

| Provider Identifier | Class | Type | Dependencies | Description |
| :--- | :--- | :--- | :--- | :--- |
| `ollama` | `OllamaAdapter` | Local HTTP API | None (`urllib.request` stdlib) | Connects to a local Ollama server (`http://localhost:11434`). |
| `huggingface_transformers` (`huggingface`, `hf`) | `HuggingFaceAdapter` | Local Pipeline (Experimental) | `transformers`, `torch` (optional) | Runs local Hugging Face `text-generation` pipelines with `local_files_only=True` by default. |
| `local_callable` | `LocalCallableAdapter` | In-Process Python Callable | None | Wraps any Python function `fn(prompt: str) -> str \| dict`. |
| `mock` | `MockModelProvider` | Deterministic Test Mock | None | Offline mock provider (`is_mock=True`) for tests, CI, and failure simulation. Never represents a real model. |
| `local_lookup` | `DeterministicLookupAdapter` | In-Memory Dictionary Lookup | None | Returns canned responses mapped by record ID (`is_mock=True`). |

---

## 2. CLI Provider Inspection

```bash
# List all registered model providers
openbench providers list

# Check connectivity and readiness of a provider
openbench providers check mock
openbench providers check ollama --endpoint http://localhost:11434 --model llama3.2
openbench providers check huggingface --model gpt2
```
