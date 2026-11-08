# Implementing a Custom Model Provider (v0.3.0)

Developers can extend OpenBench AI with custom model adapters by subclassing `BaseModelAdapter` and registering the class in `PROVIDER_REGISTRY`.

---

## 1. Required Interface (`BaseModelAdapter`)

```python
from openbench.adapters import (
    BaseModelAdapter,
    GenerationConfig,
    ProviderHealthStatus,
    register_provider,
)
from openbench.dataset import DatasetRecord
from openbench.evaluator import ResponseItem


class MyCustomAdapter(BaseModelAdapter):
    provider_name = "my_custom_provider"
    description = "Custom adapter for an internal inference service."
    is_experimental = False
    is_mock = False

    def __init__(
        self,
        model_name: str,
        model_version: str = "1.0",
        generation_config: GenerationConfig | None = None,
        timeout_seconds: float = 30.0,
    ) -> None:
        super().__init__(
            model_name=model_name,
            model_version=model_version,
            generation_config=generation_config,
            timeout_seconds=timeout_seconds,
        )

    def generate_one(self, record: DatasetRecord) -> ResponseItem:
        # 1. Read record.prompt and self.generation_config
        # 2. Call your model and return a ResponseItem preserving record.id
        output_text = f"Answer for {record.id}"
        return ResponseItem(
            id=record.id,
            response=output_text,
            latency_ms=12.4,
            token_usage={"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
        )

    def check_connection(self) -> ProviderHealthStatus:
        return ProviderHealthStatus(
            provider_name=self.provider_name,
            available=True,
            message="Custom service is reachable.",
        )


register_provider(MyCustomAdapter)
```

---

## 2. Contract Requirements

1. **Preserve Record IDs**: `generate_one(record)` must set `ResponseItem.id = record.id`.
2. **Never Fabricate Telemetry**: Set `latency_ms=None` and `token_usage=None` if your backend does not measure or report them.
3. **Redact Secrets**: Never include raw API keys in `self.inference_config`.
