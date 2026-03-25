"""
OpenAI-compatible adapter helpers.

Used by: openai.py, together.py, perplexity.py, grok.py
These four providers all speak the OpenAI chat completions API; they differ
only in base URL and the models they offer.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import AsyncIterator

try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None  # type: ignore[assignment,misc]

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


# ---------------------------------------------------------------------------
# Message building
# ---------------------------------------------------------------------------

def build_openai_messages(request: CanonicalRequest) -> list[dict]:
    """Convert a CanonicalRequest into the OpenAI messages array.

    System prompt is prepended as a system-role message when present.
    Multimodal content (ImagePart) is translated to the image_url format.
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
                    parts.append({"type": "text", "text": part.text})
                elif isinstance(part, ImagePart):
                    url = (
                        part.url
                        if part.url
                        else f"data:{part.media_type};base64,{part.data}"
                    )
                    parts.append({"type": "image_url", "image_url": {"url": url}})
            messages.append({"role": msg.role, "content": parts})

    return messages


# ---------------------------------------------------------------------------
# Retryable error classification
# ---------------------------------------------------------------------------

def is_retryable_openai(error: Exception) -> bool:
    try:
        import openai
        return isinstance(
            error,
            (openai.RateLimitError, openai.InternalServerError, openai.APIConnectionError),
        )
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# One-shot completion
# ---------------------------------------------------------------------------

async def openai_compat_complete(
    request: CanonicalRequest,
    model_entry: ModelEntry,
    credentials: Credentials,
    params: dict,
    resolution_log: list[str],
    include_raw: bool,
    provider_name: str,
    default_base_url: str,
) -> ResponseEnvelope:
    if AsyncOpenAI is None:
        raise ImportError("Install the openai package: pip install openai")

    client = AsyncOpenAI(
        api_key=credentials.api_key,
        base_url=credentials.base_url or default_base_url,
    )
    messages = build_openai_messages(request)
    payload: dict = {
        "model": model_entry.id,
        "messages": messages,
        **params,
        **request.provider_options,
    }

    start = time.monotonic()
    timestamp = datetime.now(timezone.utc).isoformat()

    resp = await client.chat.completions.create(**payload)
    latency_ms = (time.monotonic() - start) * 1000

    choice = resp.choices[0]
    usage = (
        TokenUsage(
            input_tokens=resp.usage.prompt_tokens,
            output_tokens=resp.usage.completion_tokens,
            total_tokens=resp.usage.total_tokens,
        )
        if resp.usage
        else None
    )

    # request_sent excludes provider_options (already merged) to stay canonical
    request_sent = {"model": model_entry.id, "messages": messages, **params}

    return ResponseEnvelope(
        response=ModelResponse(
            content=choice.message.content or "",
            finish_reason=choice.finish_reason or "stop",
        ),
        request_sent=request_sent,
        param_resolution_log=resolution_log,
        provider=provider_name,
        model_id=model_entry.id,
        latency_ms=latency_ms,
        timestamp=timestamp,
        usage=usage,
        raw=resp.model_dump() if include_raw else None,
    )


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------

async def openai_compat_stream(
    request: CanonicalRequest,
    model_entry: ModelEntry,
    credentials: Credentials,
    params: dict,
    default_base_url: str,
) -> AsyncIterator[StreamChunk]:
    """Async generator — call with `async for chunk in openai_compat_stream(...)`."""
    if AsyncOpenAI is None:
        raise ImportError("Install the openai package: pip install openai")

    client = AsyncOpenAI(
        api_key=credentials.api_key,
        base_url=credentials.base_url or default_base_url,
    )
    messages = build_openai_messages(request)
    payload: dict = {
        "model": model_entry.id,
        "messages": messages,
        "stream": True,
        **params,
        **request.provider_options,
    }

    async with client.chat.completions.create(**payload) as stream:
        async for chunk in stream:
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            delta = choice.delta.content or ""
            yield StreamChunk(delta=delta, finish_reason=choice.finish_reason)
