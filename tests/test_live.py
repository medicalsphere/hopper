"""
Live integration tests — make real API calls using keys from .env.

Run with:
    uv run pytest -m live -v               # all providers
    uv run pytest -m live -v -k anthropic  # one provider

Excluded from the default test run (no -m flag). Providers without a
key in .env are skipped automatically.
"""
from __future__ import annotations

import os
import pathlib

import pytest


# ---------------------------------------------------------------------------
# Load .env before any imports that might need env vars
# ---------------------------------------------------------------------------

def _load_dotenv() -> None:
    for candidate in (
        pathlib.Path(__file__).parent.parent / ".env",
        pathlib.Path(__file__).parent / ".env",
    ):
        if not candidate.exists():
            continue
        with candidate.open() as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip().strip("'\""))
        break


_load_dotenv()

import hopper
from hopper import CanonicalMessage, CanonicalRequest, Credentials

# ---------------------------------------------------------------------------
# Provider table: (provider_label, model_id, api_key_env, base_url_env)
# ---------------------------------------------------------------------------

PROVIDERS = [
    ("anthropic",  "claude-sonnet-4-6",     "ANTHROPIC_API_KEY",  None),
    ("openai",     "gpt-5.4-2026-03-05",    "OPENAI_API_KEY",     None),
    ("google",     "gemini-3-flash-preview", "GEMINI_API_KEY",     None),
    ("together",   "zai-org/GLM-5",          "TOGETHER_API_KEY",   None),
    ("perplexity", "perplexity/sonar",        "PERPLEXITY_API_KEY", None),
    ("grok",       "grok-4.20",              "XAI_API_KEY",        None),
    ("kimi",       "kimi-k2.6",              "KIMI_API_KEY",       None),
    ("zai",        "glm-5.2",               "ZAI_API_KEY",        None),
    ("fugu",       "fugu-ultra",             "FUGU_API_KEY",       None),
    ("openrouter", "openrouter/fusion",       "OPENROUTER_API_KEY", None),
]


def _creds(api_key_env: str, base_url_env: str | None) -> Credentials | None:
    key = os.environ.get(api_key_env)
    if not key:
        return None
    base_url = os.environ.get(base_url_env) if base_url_env else None
    return Credentials(api_key=key, base_url=base_url)


def _ids(providers):
    return [p[0] for p in providers]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(params=PROVIDERS, ids=_ids(PROVIDERS))
def provider(request):
    label, model, key_env, base_url_env = request.param
    creds = _creds(key_env, base_url_env)
    if creds is None:
        pytest.skip(f"{key_env} not set")
    return label, model, creds


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

pytestmark = pytest.mark.live

PROMPT = "Reply with the single word: hello"


@pytest.mark.live
async def test_complete(provider):
    label, model, creds = provider
    req = CanonicalRequest(
        model=model,
        messages=[CanonicalMessage(role="user", content=PROMPT)],
    )
    envelope = await hopper.complete(req, creds)

    assert envelope.provider == label
    assert envelope.model_id == model
    assert isinstance(envelope.response.content, str)
    assert len(envelope.response.content) > 0
    assert envelope.response.finish_reason is not None
    assert envelope.usage is not None
    assert envelope.usage.input_tokens > 0
    assert envelope.usage.output_tokens > 0


@pytest.mark.live
async def test_stream(provider):
    label, model, creds = provider
    req = CanonicalRequest(
        model=model,
        messages=[CanonicalMessage(role="user", content=PROMPT)],
    )

    chunks = []
    async for chunk in hopper.stream(req, creds):
        chunks.append(chunk)

    assert len(chunks) > 0

    text_chunks = [c for c in chunks if c.delta]
    assert len(text_chunks) > 0, "No text chunks received"

    full_text = "".join(c.delta for c in chunks)
    assert len(full_text) > 0

    # Every delta in a text chunk must be a non-empty string (no None leaked through)
    for c in text_chunks:
        assert isinstance(c.delta, str)
        assert c.delta != ""

    # Exactly one chunk should carry finish_reason
    finish_chunks = [c for c in chunks if c.finish_reason]
    assert len(finish_chunks) == 1, f"Expected 1 finish chunk, got {len(finish_chunks)}"
    assert finish_chunks[0].finish_reason is not None
