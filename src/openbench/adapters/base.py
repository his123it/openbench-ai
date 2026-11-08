"""Base model inference adapter interface for OpenBench AI."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from openbench.dataset import Dataset, DatasetRecord
from openbench.evaluator import ModelResponseSet, ResponseItem


class BaseModelAdapter(ABC):
    """Abstract interface for connecting local or external language models to OpenBench AI.

    Design Principles:
    - Adapters are strictly optional; core OpenBench evaluation works directly on JSON files.
    - Never hardcode tokens or automatically call paid APIs.
    - Record reproducibility metadata (`model_name`, `model_version`, `model_provider`,
      `inference_config`) alongside generated responses.
    """

    provider_name: str = "base_adapter"
    is_experimental: bool = False

    def __init__(
        self,
        model_name: str,
        model_version: str = "1.0",
        inference_config: dict[str, Any] | None = None,
    ) -> None:
        if not model_name or not model_name.strip():
            raise ValueError("model_name must be a non-empty string")
        self.model_name = model_name.strip()
        self.model_version = model_version.strip() or "1.0"
        self.inference_config = dict(inference_config or {})

    @abstractmethod
    def generate_one(self, record: DatasetRecord) -> ResponseItem:
        """Generate a single ResponseItem for a given DatasetRecord."""

    def generate_responses(self, dataset: Dataset) -> ModelResponseSet:
        """Run inference across all records in a Dataset and return a ModelResponseSet."""
        items = [self.generate_one(record) for record in dataset.records]
        return ModelResponseSet(
            model_name=self.model_name,
            model_version=self.model_version,
            responses=items,
            model_provider=self.provider_name,
            inference_config=self.reproducibility_metadata(),
        )

    def reproducibility_metadata(self) -> dict[str, Any]:
        """Return safe reproducibility metadata for reports and cards."""
        return {
            "provider": self.provider_name,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "experimental": self.is_experimental,
            **self.inference_config,
        }
