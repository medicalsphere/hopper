"""
Z.AI adapter — uses the OpenAI SDK against Z.AI's OpenAI-compatible
Chat Completions endpoint.

Install: pip install openai
Docs: https://docs.z.ai/api-reference/introduction

Z.AI's API mirrors the OpenAI chat completions request/response shape,
so build_openai_messages() and choices[0].message/usage parsing apply
directly. One difference from plain OpenAI:

- The `thinking` param is not a recognised kwarg on
  client.chat.completions.create(), so it is moved into `extra_body`.
  (max_tokens is accepted as-is — no renaming needed.)
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import AsyncIterator

from hopper.adapters.shared import build_openai_messages, is_retryable_openai
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
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None  # type: ignore[assignment,misc]

_BASE_URL = "https://api.z.ai/api/paas/v4/"
_PROVIDER = "zai"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _translate_params(params: dict) -> tuple[dict, dict]:
    """Split canonical params into (api_params, extra_body).

    thinking is moved to extra_body since it is not part of the
    OpenAI SDK's chat.completions.create() signature.
    max_tokens is passed through unchanged (Z.AI accepts it natively).
    """
    api_params: dict = {}
    extra_body: dict = {}
    for key, value in params.items():
        if key == "thinking":
            extra_body["thinking"] = value
        else:
            api_params[key] = value
    return api_params, extra_body


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

class ZaiAdapter:

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
        try:
            messages = build_openai_messages(request)
            api_params, extra_body = _translate_params(params)

            payload: dict = {
                "model": model_entry.id,
                "messages": messages,
                **api_params,
                **request.provider_options,
            }
            if extra_body:
                payload["extra_body"] = extra_body

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

            request_sent: dict = {"model": model_entry.id, "messages": messages, **api_params}
            if extra_body:
                request_sent["extra_body"] = extra_body

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
            base_url=credentials.base_url or _BASE_URL,
        )
        try:
            messages = build_openai_messages(request)
            api_params, extra_body = _translate_params(params)

            payload: dict = {
                "model": model_entry.id,
                "messages": messages,
                "stream": True,
                **api_params,
                **request.provider_options,
            }
            if extra_body:
                payload["extra_body"] = extra_body

            response = await client.chat.completions.create(**payload)
            async for chunk in response:
                if not chunk.choices:
                    continue
                choice = chunk.choices[0]
                if choice.delta.content is not None:
                    yield StreamChunk(delta=choice.delta.content, finish_reason=choice.finish_reason)
                elif choice.finish_reason:
                    yield StreamChunk(delta="", finish_reason=choice.finish_reason)
        finally:
            await client.close()

    def is_retryable(self, error: Exception) -> bool:
        return is_retryable_openai(error)


ADAPTER = ZaiAdapter()
