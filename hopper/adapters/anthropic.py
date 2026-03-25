from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import AsyncIterator

from hopper.types import (
    CanonicalRequest,
    Credentials,
    ImagePart,
    ModelEntry,
    ModelResponse,
    ResponseEnvelope,
    StreamChunk,
    TextPart,
    TokenUsage,
)

try:
    import anthropic as _sdk
except ImportError:
    _sdk = None  # type: ignore[assignment]

_BASE_URL = "https://api.anthropic.com"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_messages(request: CanonicalRequest) -> list[dict]:
    messages = []
    for msg in request.messages:
        if isinstance(msg.content, str):
            messages.append({"role": msg.role, "content": msg.content})
        else:
            parts: list[dict] = []
            for part in msg.content:
                if isinstance(part, TextPart):
                    parts.append({"type": "text", "text": part.text})
                elif isinstance(part, ImagePart):
                    if part.data:
                        parts.append({
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": part.media_type,
                                "data": part.data,
                            },
                        })
                    else:
                        parts.append({
                            "type": "image",
                            "source": {"type": "url", "url": part.url},
                        })
            messages.append({"role": msg.role, "content": parts})
    return messages


def _is_retryable(error: Exception) -> bool:
    if _sdk is None:
        return False
    return isinstance(
        error,
        (_sdk.RateLimitError, _sdk.InternalServerError, _sdk.APIConnectionError),
    )


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

class AnthropicAdapter:

    async def complete(
        self,
        request: CanonicalRequest,
        model_entry: ModelEntry,
        credentials: Credentials,
        params: dict,
        resolution_log: list[str],
        include_raw: bool,
    ) -> ResponseEnvelope:
        if _sdk is None:
            raise ImportError("Install the anthropic package: pip install anthropic")

        client = _sdk.AsyncAnthropic(
            api_key=credentials.api_key,
            base_url=credentials.base_url or _BASE_URL,
        )
        messages = _build_messages(request)

        payload: dict = {
            "model": model_entry.id,
            "messages": messages,
            **params,
            **request.provider_options,
        }
        if request.system:
            payload["system"] = request.system

        start = time.monotonic()
        timestamp = datetime.now(timezone.utc).isoformat()

        resp = await client.messages.create(**payload)
        latency_ms = (time.monotonic() - start) * 1000

        content = next(
            (block.text for block in resp.content if block.type == "text"), ""
        )
        usage = TokenUsage(
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
            total_tokens=resp.usage.input_tokens + resp.usage.output_tokens,
        )
        request_sent = {"model": model_entry.id, "messages": messages, **params}
        if request.system:
            request_sent["system"] = request.system

        return ResponseEnvelope(
            response=ModelResponse(
                content=content,
                finish_reason=resp.stop_reason or "end_turn",
            ),
            request_sent=request_sent,
            param_resolution_log=resolution_log,
            provider="anthropic",
            model_id=model_entry.id,
            latency_ms=latency_ms,
            timestamp=timestamp,
            usage=usage,
            raw=resp.model_dump() if include_raw else None,
        )

    async def stream(
        self,
        request: CanonicalRequest,
        model_entry: ModelEntry,
        credentials: Credentials,
        params: dict,
    ) -> AsyncIterator[StreamChunk]:
        if _sdk is None:
            raise ImportError("Install the anthropic package: pip install anthropic")

        client = _sdk.AsyncAnthropic(
            api_key=credentials.api_key,
            base_url=credentials.base_url or _BASE_URL,
        )
        messages = _build_messages(request)

        payload: dict = {
            "model": model_entry.id,
            "messages": messages,
            **params,
            **request.provider_options,
        }
        if request.system:
            payload["system"] = request.system

        async with client.messages.stream(**payload) as stream:
            async for text in stream.text_stream:
                yield StreamChunk(delta=text)

            final = await stream.get_final_message()
            usage = TokenUsage(
                input_tokens=final.usage.input_tokens,
                output_tokens=final.usage.output_tokens,
                total_tokens=final.usage.input_tokens + final.usage.output_tokens,
            )
            yield StreamChunk(delta="", finish_reason=final.stop_reason, usage=usage)

    def is_retryable(self, error: Exception) -> bool:
        return _is_retryable(error)


ADAPTER = AnthropicAdapter()
