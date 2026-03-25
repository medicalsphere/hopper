"""
Together AI adapter — uses the dedicated `together` Python SDK.

Install: pip install together
Docs: https://docs.together.ai/docs/inference-python

The Together SDK mirrors the OpenAI chat completions interface, so message
format and response parsing are the same. The client is AsyncTogether, not
AsyncOpenAI. build_openai_messages() from shared.py is still used since the
message format is identical.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import AsyncIterator

from hopper.adapters.shared import build_openai_messages
from hopper.types import (
    CanonicalRequest,
    Credentials,
    ModelEntry,
    ModelResponse,
    ResponseEnvelope,
    StreamChunk,
    TokenUsage,
)

try:
    from together import AsyncTogether
except ImportError:
    AsyncTogether = None  # type: ignore[assignment,misc]

_PROVIDER = "together"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_retryable(error: Exception) -> bool:
    # Check HTTP status code for provider-agnostic retry logic
    status = getattr(getattr(error, "response", None), "status_code", None)
    if status is not None:
        return status in (429, 500, 502, 503, 529)
    return False


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

class TogetherAdapter:

    async def complete(
        self,
        request: CanonicalRequest,
        model_entry: ModelEntry,
        credentials: Credentials,
        params: dict,
        resolution_log: list[str],
        include_raw: bool,
    ) -> ResponseEnvelope:
        if AsyncTogether is None:
            raise ImportError("Install the together package: pip install together")

        client = AsyncTogether(api_key=credentials.api_key)
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

        request_sent = {"model": model_entry.id, "messages": messages, **params}

        return ResponseEnvelope(
            response=ModelResponse(
                content=choice.message.content or "",
                finish_reason=choice.finish_reason or "stop",
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
        if AsyncTogether is None:
            raise ImportError("Install the together package: pip install together")

        client = AsyncTogether(api_key=credentials.api_key)
        messages = build_openai_messages(request)

        payload: dict = {
            "model": model_entry.id,
            "messages": messages,
            "stream": True,
            **params,
            **request.provider_options,
        }

        response = await client.chat.completions.create(**payload)
        async for chunk in response:
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            yield StreamChunk(delta=choice.delta.content or "", finish_reason=choice.finish_reason)

    def is_retryable(self, error: Exception) -> bool:
        return _is_retryable(error)


ADAPTER = TogetherAdapter()
