"""Model providers and adapters package for OpenBench AI."""

from openbench.adapters.base import (
    PROVIDER_REGISTRY,
    BaseModelAdapter,
    GenerationConfig,
    ProviderHealthStatus,
    create_provider,
    list_registered_providers,
    register_provider,
)
from openbench.adapters.huggingface import HuggingFaceAdapter
from openbench.adapters.local import DeterministicLookupAdapter, LocalCallableAdapter
from openbench.adapters.mock import MockModelProvider
from openbench.adapters.ollama import DEFAULT_OLLAMA_ENDPOINT, OllamaAdapter, OllamaConnectionError

# Register all built-in providers
register_provider(MockModelProvider)
register_provider(OllamaAdapter)
register_provider(HuggingFaceAdapter, aliases=["huggingface", "hf"])
register_provider(LocalCallableAdapter)
register_provider(DeterministicLookupAdapter)

__all__ = [
    "BaseModelAdapter",
    "DEFAULT_OLLAMA_ENDPOINT",
    "DeterministicLookupAdapter",
    "GenerationConfig",
    "HuggingFaceAdapter",
    "LocalCallableAdapter",
    "MockModelProvider",
    "OllamaAdapter",
    "OllamaConnectionError",
    "PROVIDER_REGISTRY",
    "ProviderHealthStatus",
    "create_provider",
    "list_registered_providers",
    "register_provider",
]
