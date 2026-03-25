"""
Google Gemini adapter — uses the google-genai SDK (pip install google-genai).

API reference: https://ai.google.dev/gemini-api/docs/quickstart
"""
from __future__ import annotations

import base64
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
    from google import genai as _sdk
    from google.genai import types as _types
except ImportError:
    _sdk = None   # type: ignore[assignment]
    _types = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_contents(request: CanonicalRequest) -> list:
    """Convert canonical messages to google-genai Content objects.

    Gemini uses "model" role instead of "assistant".
    Images are passed as Part.from_bytes (base64) or Part.from_uri (URL).
    """
    contents = []
    for msg in request.messages:
        role = "model" if msg.role == "assistant" else "user"
        if isinstance(msg.content, str):
            contents.append(
                _types.Content(
                    role=role,
                    parts=[_types.Part.from_text(text=msg.content)],
                )
            )
        else:
            parts = []
            for part in msg.content:
                if isinstance(part, TextPart):
                    parts.append(_types.Part.from_text(text=part.text))
                elif isinstance(part, ImagePart):
                    if part.data:
                        parts.append(
                            _types.Part.from_bytes(
                                data=base64.b64decode(part.data),
                                mime_type=part.media_type,
                            )
                        )
                    else:
                        parts.append(
                            _types.Part.from_uri(
                                file_uri=part.url,
                                mime_type=part.media_type,
                            )
                        )
            contents.append(_types.Content(role=role, parts=parts))
    return contents


def _build_config(params: dict, system: str | None):
    """Build a GenerateContentConfig from canonical params.

    Translates canonical max_tokens → max_output_tokens.
    System instruction lives inside the config object (not a top-level param).
    Returns None if there is nothing to configure.
    """
    kwargs: dict = {}
    for key, value in params.items():
        if key == "max_tokens":
            kwargs["max_output_tokens"] = value
        else:
            kwargs[key] = value
    if system:
        kwargs["system_instruction"] = system
    return _types.GenerateContentConfig(**kwargs) if kwargs else None


def _finish_reason(resp) -> str:
    if resp.candidates:
        return str(resp.candidates[0].finish_reason)
    return "STOP"


def _is_retryable(error: Exception) -> bool:
    try:
        from google.api_core.exceptions import (
            InternalServerError,
            ServiceUnavailable,
            TooManyRequests,
        )
        return isinstance(error, (InternalServerError, ServiceUnavailable, TooManyRequests))
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

class GoogleAdapter:

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
            raise ImportError(
                "Install the google-genai package: pip install google-genai"
            )

        client = _sdk.Client(api_key=credentials.api_key)
        contents = _build_contents(request)
        config = _build_config(params, request.system)

        start = time.monotonic()
        timestamp = datetime.now(timezone.utc).isoformat()

        resp = await client.aio.models.generate_content(
            model=model_entry.id,
            contents=contents,
            config=config,
        )
        latency_ms = (time.monotonic() - start) * 1000

        usage = None
        if resp.usage_metadata:
            usage = TokenUsage(
                input_tokens=resp.usage_metadata.prompt_token_count or 0,
                output_tokens=resp.usage_metadata.candidates_token_count or 0,
                total_tokens=resp.usage_metadata.total_token_count or 0,
            )

        # Build a JSON-serialisable request_sent from canonical data, not SDK objects
        translated_params = {
            ("max_output_tokens" if k == "max_tokens" else k): v
            for k, v in params.items()
        }
        request_sent: dict = {"model": model_entry.id, "config": translated_params}
        if request.system:
            request_sent["system_instruction"] = request.system

        raw = None
        if include_raw:
            try:
                raw = resp.model_dump()
            except AttributeError:
                raw = str(resp)

        return ResponseEnvelope(
            response=ModelResponse(
                content=resp.text or "",
                finish_reason=_finish_reason(resp),
            ),
            request_sent=request_sent,
            param_resolution_log=resolution_log,
            provider="google",
            model_id=model_entry.id,
            latency_ms=latency_ms,
            timestamp=timestamp,
            usage=usage,
            raw=raw,
        )

    async def stream(
        self,
        request: CanonicalRequest,
        model_entry: ModelEntry,
        credentials: Credentials,
        params: dict,
    ) -> AsyncIterator[StreamChunk]:
        if _sdk is None:
            raise ImportError(
                "Install the google-genai package: pip install google-genai"
            )

        client = _sdk.Client(api_key=credentials.api_key)
        contents = _build_contents(request)
        config = _build_config(params, request.system)

        async for chunk in await client.aio.models.generate_content_stream(
            model=model_entry.id,
            contents=contents,
            config=config,
        ):
            yield StreamChunk(delta=chunk.text or "")

    def is_retryable(self, error: Exception) -> bool:
        return _is_retryable(error)


ADAPTER = GoogleAdapter()
