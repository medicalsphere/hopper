"""
xAI Grok adapter — uses the OpenAI Responses API via xAI's compatible endpoint.

Install: pip install openai   (xAI exposes an OpenAI-compatible Responses API)
Docs: https://docs.x.ai/developers/quickstart

xAI differs from OpenAI's own Responses API in one key way: the system prompt
is embedded directly in the `input` array as a {"role":"system"} message,
not passed via a separate `instructions` parameter.
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

_BASE_URL = "https://api.x.ai/v1"
_PROVIDER = "grok"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_input(request: CanonicalRequest) -> list[dict]:
    """Build the Responses API `input` array for xAI.

    System prompt is embedded as the first message with role 'system' —
    this is xAI's convention, distinct from OpenAI's separate `instructions`
    parameter.
    """
    messages: list[dict] = []

    if request.system:
        messages.append({"role": "system", "content": request.system})

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

class GrokAdapter:

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
            base_url=credentials.base_url or _BASE_URL,
        )

        input_messages = _build_input(request)
        api_params = _translate_params(params)

        payload: dict = {
            "model": model_entry.id,
            "input": input_messages,
            **api_params,
            **request.provider_options,
        }

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
            base_url=credentials.base_url or _BASE_URL,
        )

        input_messages = _build_input(request)
        api_params = _translate_params(params)

        stream_params: dict = {
            "model": model_entry.id,
            "input": input_messages,
            **api_params,
            **request.provider_options,
        }

        async with client.responses.stream(**stream_params) as stream:
            async for text in stream.text_deltas:
                yield StreamChunk(delta=text)

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

    def is_retryable(self, error: Exception) -> bool:
        return _is_retryable(error)


ADAPTER = GrokAdapter()
