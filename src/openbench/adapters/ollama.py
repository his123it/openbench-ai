"""Local Ollama-compatible HTTP model adapter for OpenBench AI."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any

from openbench.adapters.base import BaseModelAdapter, GenerationConfig, ProviderHealthStatus
from openbench.dataset import DatasetRecord
from openbench.evaluator import ResponseItem
from openbench.exceptions import ConfigurationError, OpenBenchError
from openbench.security import sanitize_error_message, validate_endpoint_url

DEFAULT_OLLAMA_ENDPOINT = "http://localhost:11434"


class OllamaConnectionError(OpenBenchError):
    """Raised when the local or configured Ollama server cannot be reached."""


class OllamaAdapter(BaseModelAdapter):
    """Adapter for local Ollama-compatible `/api/generate` and `/api/tags` endpoints.

    Features:
    - Uses Python's standard `urllib.request` by default (zero mandatory dependencies)
      or an injected `transport_fn` for deterministic unit testing.
    - Validates `endpoint` URL via `validate_endpoint_url`.
    - Supports both standard non-streaming (`stream=False`) and streaming NDJSON (`stream=True`).
    - Extracts provider-reported token counts (`prompt_eval_count`, `eval_count`) when returned by Ollama.
    - Provides clear, actionable error messages when Ollama is not installed or not running.
    """

    provider_name = "ollama"
    supports_generation_seed = True
    description = "Local Ollama HTTP API adapter (/api/generate and /api/tags)."
    is_experimental = False
    is_mock = False

    def __init__(
        self,
        model_name: str = "llama3.2",
        model_version: str = "latest",
        endpoint: str = DEFAULT_OLLAMA_ENDPOINT,
        generation_config: GenerationConfig | None = None,
        timeout_seconds: float = 30.0,
        allow_remote_http: bool = False,
        transport_fn: Callable[[str, dict[str, Any] | None, float], Any] | None = None,
    ) -> None:
        validated_endpoint = validate_endpoint_url(
            endpoint,
            allow_remote_http=allow_remote_http,
        )
        super().__init__(
            model_name=model_name,
            model_version=model_version,
            generation_config=generation_config,
            timeout_seconds=timeout_seconds,
            inference_config={
                "endpoint": validated_endpoint,
            },
        )
        self.endpoint = validated_endpoint
        self._transport_fn = transport_fn

    def _http_request(
        self,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> Any:
        url = f"{self.endpoint}{path}"
        if self._transport_fn is not None:
            return self._transport_fn(url, payload, self.timeout_seconds)

        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        data_bytes = json.dumps(payload).encode("utf-8") if payload is not None else None
        req = urllib.request.Request(
            url,
            data=data_bytes,
            headers=headers,
            method="POST" if payload is not None else "GET",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                raw_body = resp.read().decode("utf-8")
                if payload and payload.get("stream"):
                    return raw_body.splitlines()
                return json.loads(raw_body)
        except urllib.error.HTTPError as exc:
            err_body = ""
            try:
                err_body = exc.read().decode("utf-8")
            except Exception:
                err_body = exc.reason
            clean_err = sanitize_error_message(f"HTTP {exc.code}: {err_body}")
            raise OllamaConnectionError(
                f"Ollama request to {url} failed with {clean_err}"
            ) from exc
        except TimeoutError as exc:
            raise TimeoutError(
                f"Ollama request to {url} timed out after {self.timeout_seconds}s"
            ) from exc
        except urllib.error.URLError as exc:
            reason_str = sanitize_error_message(str(exc.reason))
            if "timed out" in reason_str.lower():
                raise TimeoutError(
                    f"Ollama request to {url} timed out after {self.timeout_seconds}s"
                ) from exc
            raise OllamaConnectionError(
                f"Could not connect to Ollama server at {self.endpoint} ({reason_str}). "
                "Ensure Ollama is installed and running (`ollama serve`) or verify the --endpoint URL."
            ) from exc

    def check_connection(self) -> ProviderHealthStatus:
        """Check whether the Ollama server is reachable at `/api/tags` and whether the model is listed."""
        try:
            tags_payload = self._http_request("/api/tags", payload=None)
        except Exception as exc:
            return ProviderHealthStatus(
                provider_name=self.provider_name,
                available=False,
                message=sanitize_error_message(str(exc)),
                details={"endpoint": self.endpoint, "model_name": self.model_name},
            )

        models_raw = tags_payload.get("models", []) if isinstance(tags_payload, dict) else []
        available_names = [
            str(m.get("name", "")) for m in models_raw if isinstance(m, dict)
        ]
        model_found = any(
            name == self.model_name or name.startswith(f"{self.model_name}:")
            for name in available_names
        )
        msg = (
            f"Connected to Ollama at {self.endpoint}; model '{self.model_name}' is available."
            if model_found
            else f"Connected to Ollama at {self.endpoint}, but model '{self.model_name}' was not found in local tags ({available_names})."
        )
        return ProviderHealthStatus(
            provider_name=self.provider_name,
            available=True,
            message=msg,
            details={
                "endpoint": self.endpoint,
                "model_name": self.model_name,
                "model_installed": model_found,
                "available_models": available_names,
            },
        )

    def generate_one(self, record: DatasetRecord) -> ResponseItem:
        """Send a generation request to `/api/generate` and normalize the response."""
        options: dict[str, Any] = {
            "temperature": self.generation_config.temperature,
            "num_predict": self.generation_config.max_tokens,
            "top_p": self.generation_config.top_p,
        }
        if self.generation_config.seed is not None:
            options["seed"] = self.generation_config.seed
            self.seed_forward_count += 1
        if self.generation_config.stop_sequences:
            options["stop"] = list(self.generation_config.stop_sequences)
        options.update(self.generation_config.extra_options)

        model_tag = (
            self.model_name
            if ":" in self.model_name or self.model_version in {"latest", "1.0"}
            else f"{self.model_name}:{self.model_version}"
        )

        payload = {
            "model": model_tag,
            "prompt": record.prompt,
            "stream": self.generation_config.stream,
            "options": options,
        }

        start = time.perf_counter()
        raw_resp = self._http_request("/api/generate", payload=payload)
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        if self.generation_config.stream and isinstance(raw_resp, list):
            chunks: list[str] = []
            final_obj: dict[str, Any] = {}
            for line in raw_resp:
                if isinstance(line, str) and line.strip():
                    part = json.loads(line)
                elif isinstance(line, dict):
                    part = line
                else:
                    continue
                chunks.append(str(part.get("response", "")))
                if part.get("done"):
                    final_obj = part
            response_text = "".join(chunks)
            meta_obj = final_obj
        elif isinstance(raw_resp, dict):
            response_text = str(raw_resp.get("response", ""))
            meta_obj = raw_resp
        else:
            raise ConfigurationError(
                f"Unexpected response format from Ollama endpoint: {type(raw_resp).__name__}"
            )

        token_usage: dict[str, int] | None = None
        prompt_eval = meta_obj.get("prompt_eval_count")
        eval_count = meta_obj.get("eval_count")
        if isinstance(prompt_eval, int) and isinstance(eval_count, int):
            token_usage = {
                "prompt_tokens": prompt_eval,
                "completion_tokens": eval_count,
                "total_tokens": prompt_eval + eval_count,
            }

        return ResponseItem(
            id=record.id,
            response=response_text,
            latency_ms=round(elapsed_ms, 3),
            token_usage=token_usage,
        )
