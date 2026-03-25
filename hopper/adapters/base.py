from __future__ import annotations

from typing import AsyncIterator, Protocol

from hopper.types import (
    CanonicalRequest,
    Credentials,
    ModelEntry,
    ResponseEnvelope,
    StreamChunk,
)


class ProviderAdapter(Protocol):
    """
    Protocol every provider adapter must satisfy.

    Invariant: always use model_entry.id (the exact API model string) when
    calling the provider, never request.model, which may be an alias.

    complete() is a standard async method — await it.

    stream() is declared as a plain `def` returning AsyncIterator[StreamChunk].
    Implementations should be async generator functions (async def + yield),
    which satisfy this protocol because AsyncGenerator is a subtype of
    AsyncIterator. Callers do: `async for chunk in adapter.stream(...)`.
    Do NOT await adapter.stream().

    is_retryable() classifies whether an exception from the provider SDK is a
    transient error worth retrying. The retry loop lives in hopper/__init__.py.
    """

    async def complete(
        self,
        request: CanonicalRequest,
        model_entry: ModelEntry,
        credentials: Credentials,
        params: dict,
        resolution_log: list[str],
        include_raw: bool,
    ) -> ResponseEnvelope: ...

    def stream(
        self,
        request: CanonicalRequest,
        model_entry: ModelEntry,
        credentials: Credentials,
        params: dict,
    ) -> AsyncIterator[StreamChunk]: ...

    def is_retryable(self, error: Exception) -> bool: ...
