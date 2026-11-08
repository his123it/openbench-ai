"""Optional Hugging Face `transformers` adapter for OpenBench AI (Experimental)."""

from __future__ import annotations

import time
from typing import Any

from openbench.adapters.base import BaseModelAdapter
from openbench.dataset import DatasetRecord
from openbench.evaluator import ResponseItem
from openbench.exceptions import ConfigurationError


class HuggingFaceAdapter(BaseModelAdapter):
    """Optional adapter for local Hugging Face `transformers` text-generation pipelines.

    Status: Experimental in v0.2.0 (requires optional `transformers` and `torch` packages).

    Safety & Reproducibility Guarantees:
    - Does NOT import `transformers` at module import time.
    - Defaults to `local_files_only=True` so weights are NEVER automatically downloaded
      from the Hugging Face Hub unless the caller explicitly sets `allow_download=True`.
    - Contains zero hardcoded API tokens and makes zero paid API calls.
    - Records explicit `model_id`, `revision`, `device`, and generation parameters.
    """

    provider_name = "huggingface_transformers"
    is_experimental = True

    def __init__(
        self,
        model_id: str,
        revision: str = "main",
        allow_download: bool = False,
        device: str = "cpu",
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        pipeline_fn: Any | None = None,
    ) -> None:
        if not model_id or not model_id.strip():
            raise ConfigurationError("An explicit Hugging Face model_id is required")
        super().__init__(
            model_name=model_id.strip(),
            model_version=revision.strip() or "main",
            inference_config={
                "model_id": model_id.strip(),
                "revision": revision.strip() or "main",
                "local_files_only": not allow_download,
                "device": device,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
            },
        )
        self.model_id = model_id.strip()
        self.revision = revision.strip() or "main"
        self.allow_download = allow_download
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self._pipeline = pipeline_fn

    def _get_pipeline(self) -> Any:
        if self._pipeline is not None:
            return self._pipeline
        try:
            from transformers import pipeline  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ConfigurationError(
                "Optional dependency 'transformers' is not installed. "
                "Install it with `pip install -e '\".[huggingface]\"'` to use HuggingFaceAdapter."
            ) from exc

        self._pipeline = pipeline(
            "text-generation",
            model=self.model_id,
            revision=self.revision,
            device=self.device,
            model_kwargs={"local_files_only": not self.allow_download},
        )
        return self._pipeline

    def generate_one(self, record: DatasetRecord) -> ResponseItem:
        pipe = self._get_pipeline()
        start = time.perf_counter()
        outputs = pipe(
            record.prompt,
            max_new_tokens=self.max_new_tokens,
            do_sample=self.temperature > 0.0,
            return_full_text=False,
        )
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
