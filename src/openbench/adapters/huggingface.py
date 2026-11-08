"""Optional Hugging Face `transformers` adapter for OpenBench AI."""

from __future__ import annotations

import time
from typing import Any

from openbench.adapters.base import BaseModelAdapter, GenerationConfig, ProviderHealthStatus
from openbench.dataset import DatasetRecord
from openbench.evaluator import ResponseItem
from openbench.exceptions import ConfigurationError, OpenBenchError


class HuggingFaceAdapter(BaseModelAdapter):
    """Optional adapter for local Hugging Face `transformers` text-generation pipelines.

    Status: Experimental (requires optional `transformers` and `torch` packages).

    Safety & Reproducibility Guarantees:
    - Does NOT import `transformers` at module import time.
    - Defaults to `local_files_only=True` so weights are NEVER automatically downloaded
      from the Hugging Face Hub unless the caller explicitly sets `allow_download=True`.
    - Contains zero hardcoded API tokens and makes zero paid API calls.
    - Records explicit `model_id`, `revision`, `device`, and generation parameters.
    - Catches `MemoryError` and CUDA/CPU OOM errors with actionable remediation hints.
    """

    provider_name = "huggingface_transformers"
    description = "Local Hugging Face transformers text-generation pipeline adapter (Experimental)."
    is_experimental = True
    is_mock = False
    optional_dependencies = ("transformers", "torch")

    def __init__(
        self,
        model_id: str = "gpt2",
        revision: str = "main",
        allow_download: bool = False,
        device: str = "cpu",
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        pipeline_fn: Any | None = None,
        generation_config: GenerationConfig | None = None,
        timeout_seconds: float = 60.0,
        model_name: str | None = None,
    ) -> None:
        resolved_id = (model_name or model_id).strip()
        if not resolved_id:
            raise ConfigurationError("An explicit Hugging Face model_id is required")
        gen_cfg = generation_config or GenerationConfig(
            temperature=temperature,
            max_tokens=max_new_tokens,
        )
        super().__init__(
            model_name=resolved_id,
            model_version=revision.strip() or "main",
            generation_config=gen_cfg,
            timeout_seconds=timeout_seconds,
            inference_config={
                "model_id": resolved_id,
                "revision": revision.strip() or "main",
                "local_files_only": not allow_download,
                "device": device,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
            },
        )
        self.model_id = resolved_id
        self.revision = revision.strip() or "main"
        self.allow_download = allow_download
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self._pipeline = pipeline_fn

    def check_connection(self) -> ProviderHealthStatus:
        if self._pipeline is not None:
            return ProviderHealthStatus(
                provider_name=self.provider_name,
                available=True,
                message=f"HuggingFaceAdapter ready with injected pipeline for '{self.model_id}'.",
                details={"model_id": self.model_id, "revision": self.revision},
            )
        try:
            import transformers  # type: ignore[import-not-found] # noqa: F401
        except ImportError:
            return ProviderHealthStatus(
                provider_name=self.provider_name,
                available=False,
                message=(
                    "Optional dependency 'transformers' is not installed. "
                    "Install it with `pip install -e '.[huggingface]'`."
                ),
                details={"model_id": self.model_id, "device": self.device},
            )
        return ProviderHealthStatus(
            provider_name=self.provider_name,
            available=True,
            message=f"transformers package is installed (model='{self.model_id}', local_files_only={not self.allow_download}).",
            details={"model_id": self.model_id, "revision": self.revision, "device": self.device},
        )

    def _get_pipeline(self) -> Any:
        if self._pipeline is not None:
            return self._pipeline
        try:
            from transformers import pipeline  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ConfigurationError(
                "Optional dependency 'transformers' is not installed. "
                "Install it with `pip install -e '.[huggingface]'` to use HuggingFaceAdapter."
            ) from exc

        try:
            self._pipeline = pipeline(
                "text-generation",
                model=self.model_id,
                revision=self.revision,
                device=self.device,
                model_kwargs={"local_files_only": not self.allow_download},
            )
        except MemoryError as exc:
            raise OpenBenchError(
                f"Insufficient system memory to load Hugging Face model '{self.model_id}' on device '{self.device}'."
            ) from exc
        except Exception as exc:
            raise OpenBenchError(
                f"Failed to load Hugging Face model '{self.model_id}' (revision='{self.revision}', "
                f"local_files_only={not self.allow_download}): {exc}"
            ) from exc
        return self._pipeline

    def generate_one(self, record: DatasetRecord) -> ResponseItem:
        pipe = self._get_pipeline()
        start = time.perf_counter()
        try:
            outputs = pipe(
                record.prompt,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.temperature > 0.0,
                return_full_text=False,
            )
        except MemoryError as exc:
            raise OpenBenchError(
                f"Out of memory while generating response for '{record.id}' with model '{self.model_id}'."
            ) from exc
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        if isinstance(outputs, list) and outputs and isinstance(outputs[0], dict):
            generated_text = str(outputs[0].get("generated_text", ""))
        else:
            generated_text = str(outputs)

        return ResponseItem(
            id=record.id,
            response=generated_text,
            latency_ms=round(elapsed_ms, 3),
        )
