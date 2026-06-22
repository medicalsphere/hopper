"""
Fugu (Sakana AI) adapter — uses the OpenAI Responses API via Sakana's endpoint.

Install: pip install openai
Docs: https://console.sakana.ai/models

Fugu exposes an OpenAI-compatible Responses API. Key differences from OpenAI:
- Fixed base URL: https://api.sakana.ai/v1
- System prompt goes in `instructions` (same as Perplexity/OpenAI).
- `max_tokens` is renamed to `max_output_tokens`.
- `temperature` is accepted by the server but silently ignored; the router
  strips it automatically because it is listed as unsupported in fugu.yaml.
- `reasoning` is supported: {"effort": "high" | "xhigh" | "max"}.
"""
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
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None  # type: ignore[assignment,misc]

_PROVIDER = "fugu"
_BASE_URL = "https://api.sakana.ai/v1"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_input(request: CanonicalRequest) -> list[dict]:
    """Build the Responses API `input` array. System prompt goes in `instructions`."""
    messages: list[dict] = []
    for msg in request.messages:
        if isinstance(msg.content, str):
            messages.append({"role": msg.role, "content": msg.content})
        else:
            parts: list[dict] = []
            for part in msg.content:
                if isinstance(part, TextPart):
                    parts.append({"type": "input_text", "text": part.text})
                elif isinstance(part, ImagePart):
                    url = (
                        part.url
                        if part.url
                        else f"data:{part.media_type};base64,{part.data}"
                    )
                    parts.append({"type": "input_image", "image_url": url})
            messages.append({"role": msg.role, "content": parts})
    return messages


def _translate_params(params: dict) -> dict:
    """max_tokens → max_output_tokens for the Responses API."""
    translated: dict = {}
    for key, value in params.items():
        if key == "max_tokens":
            translated["max_output_tokens"] = value
        else:
            translated[key] = value
    return translated


def _finish_reason(resp) -> str:
    if resp.status == "completed":
        return "stop"
    if resp.status == "incomplete" and resp.incomplete_details:
        return getattr(resp.incomplete_details, "reason", "incomplete")
    return resp.status or "stop"


def _is_retryable(error: Exception) -> bool:
    try:
        import openai
        return isinstance(
            error,
            (openai.RateLimitError, openai.InternalServerError, openai.APIConnectionError),
        )
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

class FuguAdapter:

    async def complete(
        self,
        request: CanonicalRequest,
        model_entry: ModelEntry,
        credentials: Credentials,
        params: dict,
        resolution_log: list[str],
        include_raw: bool,
    ) -> ResponseEnvelope:
        if AsyncOpenAI is None:
            raise ImportError("Install the openai package: pip install openai")

        client = AsyncOpenAI(
            api_key=credentials.api_key,
            base_url=_BASE_URL,
        )
        try:
            input_messages = _build_input(request)
            api_params = _translate_params(params)

            payload: dict = {
                "model": model_entry.id,
                "input": input_messages,
                **api_params,
                **request.provider_options,
            }
            if request.system:
                payload["instructions"] = request.system

            start = time.monotonic()
            timestamp = datetime.now(timezone.utc).isoformat()

            resp = await client.responses.create(**payload)
            latency_ms = (time.monotonic() - start) * 1000

            usage = (
                TokenUsage(
                    input_tokens=resp.usage.input_tokens,
                    output_tokens=resp.usage.output_tokens,
                    total_tokens=resp.usage.total_tokens,
                )
                if resp.usage
                else None
            )

            request_sent: dict = {"model": model_entry.id, "input": input_messages, **api_params}
            if request.system:
                request_sent["instructions"] = request.system

            return ResponseEnvelope(
                response=ModelResponse(
                    content=resp.output_text or "",
                    finish_reason=_finish_reason(resp),
                ),
                request_sent=request_sent,
                param_resolution_log=resolution_log,
                provider=_PROVIDER,
                model_id=model_entry.id,
                latency_ms=latency_ms,
                timestamp=timestamp,
                usage=usage,
                raw=resp.model_dump() if include_raw else None,
            )
        finally:
            await client.close()

    async def stream(
        self,
        request: CanonicalRequest,
        model_entry: ModelEntry,
        credentials: Credentials,
        params: dict,
    ) -> AsyncIterator[StreamChunk]:
        if AsyncOpenAI is None:
            raise ImportError("Install the openai package: pip install openai")

        client = AsyncOpenAI(
            api_key=credentials.api_key,
            base_url=_BASE_URL,
        )
        try:
            input_messages = _build_input(request)
            api_params = _translate_params(params)

            stream_params: dict = {
                "model": model_entry.id,
                "input": input_messages,
                **api_params,
                **request.provider_options,
            }
            if request.system:
                stream_params["instructions"] = request.system

            async with client.responses.stream(**stream_params) as stream:
                async for event in stream:
                    if event.type == "response.output_text.delta":
                        yield StreamChunk(delta=event.delta)

                final = await stream.get_final_response()
                final_usage = (
                    TokenUsage(
                        input_tokens=final.usage.input_tokens,
                        output_tokens=final.usage.output_tokens,
                        total_tokens=final.usage.total_tokens,
                    )
                    if final.usage
                    else None
                )
                yield StreamChunk(
                    delta="",
                    finish_reason=_finish_reason(final),
                    usage=final_usage,
                )
        finally:
            await client.close()

    def is_retryable(self, error: Exception) -> bool:
        return _is_retryable(error)


ADAPTER = FuguAdapter()
