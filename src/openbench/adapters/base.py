"""Base model inference adapter interface, generation config, and provider registry for OpenBench AI."""

from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from openbench.dataset import Dataset, DatasetRecord
from openbench.evaluator import ModelResponseSet, ResponseItem
from openbench.exceptions import ConfigurationError
from openbench.security import redact_sensitive_data


@dataclass(frozen=True)
class GenerationConfig:
    """Standardized generation parameters passed to model providers."""

    temperature: float = 0.0
    max_tokens: int = 256
    top_p: float = 1.0
    seed: int | None = None
    stop_sequences: list[str] = field(default_factory=list)
    stream: bool = False
    extra_options: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "stream": self.stream,
        }
        if self.seed is not None:
            payload["seed"] = self.seed
        if self.stop_sequences:
            payload["stop_sequences"] = list(self.stop_sequences)
        if self.extra_options:
            payload["extra_options"] = redact_sensitive_data(dict(self.extra_options))
        return payload


@dataclass(frozen=True)
class ProviderHealthStatus:
    """Result of checking a model provider's availability and configuration."""

    provider_name: str
    available: bool
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider_name": self.provider_name,
            "available": self.available,
            "message": self.message,
            "details": redact_sensitive_data(dict(self.details)),
        }


class BaseModelAdapter(ABC):
    """Abstract interface for connecting local or external language models to OpenBench AI.

    Design Principles:
    - Adapters are strictly optional; core OpenBench evaluation works directly on JSON files.
    - Never hardcode tokens or automatically call paid APIs.
    - Record reproducibility metadata (`model_name`, `model_version`, `model_provider`,
      `inference_config`) alongside generated responses.
    """

    provider_name: str = "base_adapter"
    description: str = "Abstract base model provider adapter."
    is_experimental: bool = False
    is_mock: bool = False
    optional_dependencies: tuple[str, ...] = ()

    def __init__(
        self,
        model_name: str,
        model_version: str = "1.0",
        inference_config: dict[str, Any] | None = None,
        generation_config: GenerationConfig | None = None,
        timeout_seconds: float = 30.0,
    ) -> None:
        if not model_name or not model_name.strip():
            raise ValueError("model_name must be a non-empty string")
        if timeout_seconds <= 0:
            raise ConfigurationError("timeout_seconds must be positive")
        self.model_name = model_name.strip()
        self.model_version = model_version.strip() or "1.0"
        self.generation_config = generation_config or GenerationConfig()
        self.timeout_seconds = float(timeout_seconds)
        base_inf = self.generation_config.to_dict()
        if inference_config:
            base_inf.update(redact_sensitive_data(dict(inference_config)))
        self.inference_config = base_inf

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

    def check_connection(self) -> ProviderHealthStatus:
        """Verify whether the provider is reachable and properly configured."""
        return ProviderHealthStatus(
            provider_name=self.provider_name,
            available=True,
            message=f"Provider '{self.provider_name}' is ready.",
            details={"model_name": self.model_name, "is_mock": self.is_mock},
        )

    def reproducibility_metadata(self) -> dict[str, Any]:
        """Return safe reproducibility metadata for reports and cards (with secrets redacted)."""
        return redact_sensitive_data(
            {
                "provider": self.provider_name,
                "model_name": self.model_name,
                "model_version": self.model_version,
                "experimental": self.is_experimental,
                "is_mock": self.is_mock,
                "timeout_seconds": self.timeout_seconds,
                **self.inference_config,
            }
        )


PROVIDER_REGISTRY: dict[str, type[BaseModelAdapter]] = {}


def register_provider(adapter_cls: type[BaseModelAdapter], aliases: list[str] | None = None) -> None:
    """Register a BaseModelAdapter subclass in `PROVIDER_REGISTRY`."""
    if not isinstance(adapter_cls, type) or not issubclass(adapter_cls, BaseModelAdapter):
        raise ConfigurationError("Provider class must inherit from BaseModelAdapter")
    p_name = getattr(adapter_cls, "provider_name", "").strip().lower()
    if not p_name:
        raise ConfigurationError("Provider class must define a non-empty 'provider_name'")
    PROVIDER_REGISTRY[p_name] = adapter_cls
    for alias in aliases or []:
        PROVIDER_REGISTRY[alias.strip().lower()] = adapter_cls


def create_provider(provider_name: str, **kwargs: Any) -> BaseModelAdapter:
    """Instantiate a registered provider adapter by name."""
    key = provider_name.strip().lower()
    if key not in PROVIDER_REGISTRY:
        available = sorted({cls.provider_name for cls in PROVIDER_REGISTRY.values()})
        raise ConfigurationError(
            f"Unknown model provider '{provider_name}'. Registered providers: {', '.join(available)}"
        )
    adapter_cls = PROVIDER_REGISTRY[key]
    sig = inspect.signature(adapter_cls.__init__)
    accepts_var_kw = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )
    filtered_kwargs: dict[str, Any] = {}
    for k, v in kwargs.items():
        if accepts_var_kw or k in sig.parameters:
            filtered_kwargs[k] = v
        else:
            raise ConfigurationError(
                f"Provider '{provider_name}' does not accept configuration argument '{k}'"
            )
    return adapter_cls(**filtered_kwargs)


def list_registered_providers() -> list[dict[str, Any]]:
    """Return metadata dictionaries for all unique registered model providers."""
    unique: dict[str, type[BaseModelAdapter]] = {}
    for cls in PROVIDER_REGISTRY.values():
        unique[cls.provider_name] = cls
    result: list[dict[str, Any]] = []
    for name in sorted(unique):
        cls = unique[name]
        result.append(
            {
                "provider_name": cls.provider_name,
                "description": cls.description,
                "is_experimental": cls.is_experimental,
                "is_mock": cls.is_mock,
                "optional_dependencies": list(cls.optional_dependencies),
            }
        )
    return result
