# Copyright (c) 2026 Chris. All rights reserved.
# Licensed under the Business Source License 1.1 — see LICENSE file.

"""
APIAdapter — Tier 1 adapter for HTTP-based models (OpenAI, Claude, HuggingFace Inference, etc.).

The most limited adapter by design — API models are black boxes. But even
input/output-only access enables behavioral comparison: run the same tasks
through HDNA (fully transparent) and an API model (opaque), and the contrast
itself is a research finding.

Usage:
    # OpenAI-compatible API
    adapter = APIAdapter(
        endpoint="https://api.openai.com/v1/chat/completions",
        api_key="sk-...",
        provider="openai",
        model="gpt-4",
    )
    output = adapter.predict("What is 2+2?")

    # Custom API
    adapter = APIAdapter(
        endpoint="https://my-model.example.com/predict",
        headers={"Authorization": "Bearer ..."},
        provider="custom",
        request_format=lambda input: {"text": input},
        response_parser=lambda resp: resp["result"],
    )

    # HuggingFace Inference API
    adapter = APIAdapter.huggingface("bert-base-uncased", api_key="hf_...")
"""

import json
import time
import numpy as np
from typing import Any, Callable, Optional
from dataclasses import dataclass, field

from .protocol import ModelAdapter, ModelInfo, Capability


@dataclass
class APICallRecord:
    """Record of a single API call for behavioral analysis."""
    timestamp: float
    input_data: Any
    output_data: Any
    latency_ms: float
    tokens_in: int = 0
    tokens_out: int = 0
    metadata: dict = field(default_factory=dict)


class APIAdapter(ModelAdapter):
    """
    Tier 1 adapter for any HTTP API model.

    Provides: predict(), get_info(), and behavioral logging.
    Does NOT provide: activations, gradients, attention, interventions.

    But behavioral analysis is still valuable — you can compare what
    HDNA does (with full transparency) against what the API model does
    (black box) on the same inputs.
    """

    def __init__(self, endpoint: str, api_key: str = None,
                 provider: str = "custom", model: str = "unknown",
                 headers: dict = None,
                 request_format: Callable = None,
                 response_parser: Callable = None,
                 name: str = None,
                 timeout: float = 30.0):
        self._endpoint = endpoint
        self._api_key = api_key
        self._provider = provider
        self._model_name = model
        self._name = name or f"{provider}/{model}"
        self._timeout = timeout

        # Headers
        self._headers = headers or {}
        if api_key:
            if provider == "openai":
                self._headers["Authorization"] = f"Bearer {api_key}"
                self._headers["Content-Type"] = "application/json"
            elif provider == "anthropic":
                self._headers["x-api-key"] = api_key
                self._headers["Content-Type"] = "application/json"
                self._headers["anthropic-version"] = "2023-06-01"
            elif provider == "huggingface":
                self._headers["Authorization"] = f"Bearer {api_key}"
            else:
                self._headers["Authorization"] = f"Bearer {api_key}"

        # Request/response formatting
        self._request_format = request_format or self._default_request_format
        self._response_parser = response_parser or self._default_response_parser

        # Behavioral logging
        self._call_log: list = []
        self._log_capacity = 10000

    @classmethod
    def openai(cls, model: str = "gpt-4", api_key: str = None,
               **kwargs) -> "APIAdapter":
        """Create an OpenAI adapter."""
        return cls(
            endpoint="https://api.openai.com/v1/chat/completions",
            api_key=api_key,
            provider="openai",
            model=model,
            **kwargs,
        )

    @classmethod
    def anthropic(cls, model: str = "claude-sonnet-4-6", api_key: str = None,
                  **kwargs) -> "APIAdapter":
        """Create an Anthropic (Claude) adapter."""
        return cls(
            endpoint="https://api.anthropic.com/v1/messages",
            api_key=api_key,
            provider="anthropic",
            model=model,
            **kwargs,
        )

    @classmethod
    def huggingface(cls, model: str, api_key: str = None,
                    **kwargs) -> "APIAdapter":
        """Create a HuggingFace Inference API adapter."""
        return cls(
            endpoint=f"https://api-inference.huggingface.co/models/{model}",
            api_key=api_key,
            provider="huggingface",
            model=model,
            **kwargs,
        )

    # --- Tier 1: Required ---

    def predict(self, input_data: Any) -> Any:
        """Send input to the API and return the response."""
        import urllib.request
        import urllib.error

        payload = self._request_format(input_data)
        body = json.dumps(payload).encode('utf-8')

        req = urllib.request.Request(
            self._endpoint,
            data=body,
            headers=self._headers,
            method='POST',
        )

        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                raw = json.loads(resp.read().decode('utf-8'))
        except urllib.error.HTTPError as e:
            error_body = e.read().decode('utf-8') if e.fp else ""
            raise RuntimeError(
                f"API error {e.code}: {error_body}"
            )
        latency = (time.perf_counter() - t0) * 1000

        output = self._response_parser(raw)

        # Log the call
        record = APICallRecord(
            timestamp=time.time(),
            input_data=input_data if isinstance(input_data, str) else str(input_data)[:200],
            output_data=output if isinstance(output, str) else str(output)[:200],
            latency_ms=latency,
            metadata={"raw_keys": list(raw.keys()) if isinstance(raw, dict) else []},
        )

        # Extract token counts if available
        if isinstance(raw, dict):
            usage = raw.get("usage", {})
            record.tokens_in = usage.get("prompt_tokens", usage.get("input_tokens", 0))
            record.tokens_out = usage.get("completion_tokens", usage.get("output_tokens", 0))

        self._call_log.append(record)
        if len(self._call_log) > self._log_capacity:
            self._call_log.pop(0)

        return output

    def get_info(self) -> ModelInfo:
        return ModelInfo(
            name=self._name,
            framework="api",
            architecture="unknown",
            parameter_count=0,
            layer_count=0,
            dtype="unknown",
            device="remote",
            extra={
                "provider": self._provider,
                "model": self._model_name,
                "endpoint": self._endpoint,
                "calls_made": len(self._call_log),
                "avg_latency_ms": self._avg_latency(),
            },
        )

    def capabilities(self) -> Capability:
        return Capability.PREDICT | Capability.INFO

    # --- Behavioral analysis (unique to API adapter) ---

    @property
    def call_log(self) -> list:
        """Access the behavioral log."""
        return self._call_log

    def behavioral_stats(self) -> dict:
        """Summary statistics from the call log."""
        if not self._call_log:
            return {"calls": 0}

        latencies = [r.latency_ms for r in self._call_log]
        return {
            "calls": len(self._call_log),
            "avg_latency_ms": round(np.mean(latencies), 1),
            "p50_latency_ms": round(np.percentile(latencies, 50), 1),
            "p99_latency_ms": round(np.percentile(latencies, 99), 1),
            "total_tokens_in": sum(r.tokens_in for r in self._call_log),
            "total_tokens_out": sum(r.tokens_out for r in self._call_log),
        }

    def _avg_latency(self) -> float:
        if not self._call_log:
            return 0.0
        return round(np.mean([r.latency_ms for r in self._call_log]), 1)

    # --- Default formatters ---

    def _default_request_format(self, input_data: Any) -> dict:
        """Format request based on provider."""
        if self._provider == "openai":
            if isinstance(input_data, str):
                return {
                    "model": self._model_name,
                    "messages": [{"role": "user", "content": input_data}],
                }
            return {"model": self._model_name, "input": input_data}

        elif self._provider == "anthropic":
            if isinstance(input_data, str):
                return {
                    "model": self._model_name,
                    "max_tokens": 1024,
                    "messages": [{"role": "user", "content": input_data}],
                }
            return {"model": self._model_name, "input": input_data}

        elif self._provider == "huggingface":
            if isinstance(input_data, str):
                return {"inputs": input_data}
            return {"inputs": input_data}

        # Custom: just wrap in a dict
        return {"input": input_data}

    def _default_response_parser(self, response: dict) -> Any:
        """Parse response based on provider."""
        if self._provider == "openai":
            try:
                return response["choices"][0]["message"]["content"]
            except (KeyError, IndexError):
                return response

        elif self._provider == "anthropic":
            try:
                return response["content"][0]["text"]
            except (KeyError, IndexError):
                return response

        elif self._provider == "huggingface":
            if isinstance(response, list) and len(response) > 0:
                if isinstance(response[0], dict):
                    return response[0].get("generated_text", response[0])
                return response[0]
            return response

        return response
