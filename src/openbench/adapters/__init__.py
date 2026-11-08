"""Optional model adapters package for OpenBench AI."""

from openbench.adapters.base import BaseModelAdapter
from openbench.adapters.huggingface import HuggingFaceAdapter
from openbench.adapters.local import DeterministicLookupAdapter, LocalCallableAdapter

__all__ = [
    "BaseModelAdapter",
    "DeterministicLookupAdapter",
    "HuggingFaceAdapter",
    "LocalCallableAdapter",
]
