from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TextPart:
    type: str = "text"
    text: str = ""


@dataclass
class ImagePart:
    type: str = "image"
    url: str | None = None
    data: str | None = None       # base64-encoded bytes
    media_type: str = "image/jpeg"

    def __post_init__(self) -> None:
        if self.url is None and self.data is None:
            raise ValueError("ImagePart requires either url or data")
        if self.url is not None and self.data is not None:
            raise ValueError("ImagePart accepts url or data, not both")


ContentPart = TextPart | ImagePart


@dataclass
class CanonicalMessage:
    role: str
    content: str | list[ContentPart]


@dataclass
class CanonicalRequest:
    model: str
    messages: list[CanonicalMessage]
    system: str | None = None
    max_tokens: int | None = None
    temperature: float | None = None
    stream: bool = False
    # Escape hatch for provider-specific params; bypasses canonical filtering.
    provider_options: dict[str, Any] = field(default_factory=dict)
    # Reasoning effort hint — passed to providers that support it (e.g. OpenAI).
    # Example: {"effort": "low"} | {"effort": "medium"} | {"effort": "high"}
    reasoning: dict[str, Any] | None = None
    # Extended thinking — Anthropic only.
    # Example: {"type": "enabled", "budget_tokens": 10000}
    #          {"type": "adaptive"}  (Opus 4.6 only)
    thinking: dict[str, Any] | None = None
    # Optional provider hint for models not in the registry (passthrough mode).
    # Must be one of the known provider names: anthropic, openai, google,
    # together, perplexity, grok.
    provider: str | None = None
    # Extra params forwarded as-is in passthrough mode (no filtering applied).
    # Example: {"temperature": 0.7, "top_p": 0.9}
    extra_params: dict[str, Any] = field(default_factory=dict)


@dataclass
class Credentials:
    """API credentials passed at call time. Hopper never reads environment variables."""
    api_key: str
    base_url: str | None = None  # override the provider's default endpoint


@dataclass
class RetryConfig:
    """Opt-in retry behaviour. Disabled by default (max_attempts=1)."""
    max_attempts: int = 1
    base_delay: float = 1.0   # seconds; doubles on each attempt
    max_delay: float = 60.0   # seconds; caps the backoff


@dataclass
class TokenUsage:
    input_tokens: int
    output_tokens: int
    total_tokens: int


@dataclass
class ModelResponse:
    content: str
    finish_reason: str


@dataclass
class StreamChunk:
    delta: str
    finish_reason: str | None = None
    usage: TokenUsage | None = None  # populated on the final chunk


@dataclass
class ModelEntry:
    """Internal registry entry. Not part of the public API."""
    id: str           # exact model string sent to the provider API
    provider: str
    aliases: list[str] = field(default_factory=list)
    supported_params: list[str] = field(default_factory=list)
    unsupported_params: list[str] = field(default_factory=list)
    defaults: dict[str, Any] = field(default_factory=dict)


@dataclass
class ResponseEnvelope:
    """
    Every call returns an envelope. The caller always knows exactly what was
    sent, what came back, and what parameter decisions were made.

    to_dict() / to_json() are fully serializable without any Hopper imports.
    The opt-in `raw` field is excluded from serialization; access it directly.
    """
    response: ModelResponse
    request_sent: dict[str, Any]
    param_resolution_log: list[str]
    provider: str
    model_id: str
    latency_ms: float
    timestamp: str          # ISO 8601
    usage: TokenUsage | None = None
    raw: Any = None         # original provider response object; opt-in via include_raw=True

    def to_dict(self) -> dict[str, Any]:
        return {
            "response": {
                "content": self.response.content,
                "finish_reason": self.response.finish_reason,
            },
            "request_sent": self.request_sent,
            "param_resolution_log": self.param_resolution_log,
            "provider": self.provider,
            "model_id": self.model_id,
            "latency_ms": self.latency_ms,
            "timestamp": self.timestamp,
            "usage": {
                "input_tokens": self.usage.input_tokens,
                "output_tokens": self.usage.output_tokens,
                "total_tokens": self.usage.total_tokens,
            } if self.usage else None,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())
