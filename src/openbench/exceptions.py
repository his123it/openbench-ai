"""Custom exceptions for OpenBench AI."""

from __future__ import annotations

from pathlib import Path


class OpenBenchError(Exception):
    """Base exception for all OpenBench AI errors."""


class DatasetValidationError(OpenBenchError):
    """Raised when a dataset file or record fails validation."""

    def __init__(
        self,
        message: str,
        file_path: str | Path | None = None,
        line_number: int | None = None,
        record_id: str | None = None,
    ) -> None:
        self.raw_message = message
        self.file_path = str(file_path) if file_path is not None else None
        self.line_number = line_number
        self.record_id = record_id

        context_parts: list[str] = []
        if self.file_path:
            context_parts.append(f"file={self.file_path}")
        if self.line_number is not None:
            context_parts.append(f"line={self.line_number}")
        if self.record_id:
            context_parts.append(f"id={self.record_id}")

        if context_parts:
            formatted = f"{message} ({', '.join(context_parts)})"
        else:
            formatted = message
        super().__init__(formatted)


class EmptyDatasetError(DatasetValidationError):
    """Raised when a dataset file or directory contains no valid evaluation records."""


class ResponseValidationError(OpenBenchError):
    """Raised when a model response file or entry fails schema validation."""

    def __init__(
        self,
        message: str,
        file_path: str | Path | None = None,
        record_id: str | None = None,
    ) -> None:
        self.raw_message = message
        self.file_path = str(file_path) if file_path is not None else None
        self.record_id = record_id

        context_parts: list[str] = []
        if self.file_path:
            context_parts.append(f"file={self.file_path}")
        if self.record_id:
            context_parts.append(f"id={self.record_id}")

        if context_parts:
            formatted = f"{message} ({', '.join(context_parts)})"
        else:
            formatted = message
        super().__init__(formatted)


class MetricError(OpenBenchError):
    """Raised when a metric configuration or computation fails."""


class ConfigurationError(OpenBenchError):
    """Raised when evaluation or comparison configuration is invalid."""
