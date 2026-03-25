"""
Hopper — unified interface for AI model API calls.

Named after Grace Hopper, who built the first compiler: the original
abstraction layer between human intent and machine execution.

Public API surface:
    complete(request, credentials, ...) -> ResponseEnvelope
    stream(request, credentials)        -> AsyncIterator[StreamChunk]

All types a caller needs are re-exported here.
"""
from __future__ import annotations

import asyncio
from typing import AsyncIterator

from hopper import router
from hopper.types import (
    CanonicalMessage,
    CanonicalRequest,
    Credentials,
    ImagePart,
    ResponseEnvelope,
    RetryConfig,
    StreamChunk,
    TextPart,
    TokenUsage,
)

__all__ = [
    "complete",
    "stream",
    "CanonicalRequest",
    "CanonicalMessage",
    "TextPart",
    "ImagePart",
    "Credentials",
    "RetryConfig",
    "ResponseEnvelope",
    "StreamChunk",
    "TokenUsage",
]


async def complete(
    request: CanonicalRequest,
    credentials: Credentials,
    retry_config: RetryConfig | None = None,
    include_raw: bool = False,
) -> ResponseEnvelope:
    """
    Call a model and return a fully-populated ResponseEnvelope.

    Parameters
    ----------
    request:      The canonical request. Use request.model to specify a model
                  ID or alias (e.g. "claude-sonnet", "gpt-4o").
    credentials:  API key and optional base URL override. Hopper never reads
                  environment variables — secret management stays in the caller.
    retry_config: Opt-in retry behaviour. Disabled by default (max_attempts=1).
                  Only transient provider errors are retried; the adapter's
                  is_retryable() method decides which errors qualify.
    include_raw:  Store the original provider response in envelope.raw.
                  Excluded from to_dict() / to_json() to guarantee serialisability.
    """
    adapter, model_entry, resolution_log = router.resolve(request)
    params, param_log = router.apply_defaults_and_filter(request, model_entry)
    full_log = resolution_log + param_log

    max_attempts = retry_config.max_attempts if retry_config else 1

    for attempt in range(max_attempts):
        try:
            return await adapter.complete(
                request=request,
                model_entry=model_entry,
                credentials=credentials,
                params=params,
                resolution_log=full_log,
                include_raw=include_raw,
            )
        except Exception as exc:
            if attempt == max_attempts - 1 or not adapter.is_retryable(exc):
                raise
            delay = min(
                (retry_config.base_delay * (2 ** attempt)),  # type: ignore[union-attr]
                retry_config.max_delay,                      # type: ignore[union-attr]
            )
            await asyncio.sleep(delay)

    # Unreachable — loop always raises or returns — but satisfies the type checker.
    raise RuntimeError("Retry loop exited without returning or raising")


async def stream(
    request: CanonicalRequest,
    credentials: Credentials,
) -> AsyncIterator[StreamChunk]:
    """
    Call a model and yield StreamChunks as they arrive.

    Usage:
        async for chunk in hopper.stream(request, credentials):
            print(chunk.delta, end="", flush=True)

    The final chunk carries finish_reason and token usage.
    Streaming does not support retries; callers should implement retry logic
    at a higher level when streaming is required.
    """
    adapter, model_entry, resolution_log = router.resolve(request)
    params, param_log = router.apply_defaults_and_filter(request, model_entry)
    _ = resolution_log + param_log  # log is available to the adapter via params path

    async for chunk in adapter.stream(request, model_entry, credentials, params):
        yield chunk
