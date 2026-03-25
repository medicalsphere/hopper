# Hopper

A unified Python library for AI model API calls.

Named after Grace Hopper — the original abstraction layer between human intent and machine execution.

## Supported providers

Anthropic, OpenAI, Google Gemini, Together AI, Perplexity, xAI Grok.

## Installation

```bash
git clone <repo>
cd hopper
uv sync --all-extras
```

Or install only the providers you need:

```bash
uv pip install -e ".[anthropic,openai,google]"
```

## Usage

```python
import asyncio
import hopper
from hopper import CanonicalRequest, CanonicalMessage, Credentials

request = CanonicalRequest(
    model="claude-sonnet",   # model ID or alias
    messages=[CanonicalMessage(role="user", content="Hello!")],
    system="You are a helpful assistant.",
)

credentials = Credentials(api_key="sk-ant-...")

# single response
envelope = asyncio.run(hopper.complete(request, credentials))
print(envelope.response.content)

# streaming
async def stream():
    async for chunk in hopper.stream(request, credentials):
        print(chunk.delta, end="", flush=True)

asyncio.run(stream())
```

### Image input

```python
from hopper import ImagePart, TextPart

request = CanonicalRequest(
    model="claude-sonnet",
    messages=[
        CanonicalMessage(
            role="user",
            content=[
                ImagePart(data="<base64>", media_type="image/jpeg"),
                TextPart(text="What is in this image?"),
            ],
        )
    ],
)
```

### Multi-turn conversations

```python
messages = [
    CanonicalMessage(role="user",      content="My name is Alice."),
    CanonicalMessage(role="assistant", content="Got it, Alice!"),
    CanonicalMessage(role="user",      content="What's my name?"),
]
request = CanonicalRequest(model="claude-sonnet", messages=messages)
```

### Model aliases

Every model has short aliases so you don't need to remember full IDs:

```
"claude-sonnet"  →  claude-sonnet-4-6
"claude-haiku"   →  claude-haiku-4-5-20251001
"gemini-3-flash" →  gemini-3-flash-preview
"gpt-5.4-mini"   →  gpt-5.4-mini-2026-03-17
"grok"           →  grok-4.20
"sonar"          →  perplexity/sonar
```

### Calling models not in the registry

Hopper ships with a curated model registry, but providers release new models
frequently. You can call any model from a supported provider without waiting
for the registry to be updated — just pass `provider=`:

```python
request = CanonicalRequest(
    model="claude-sonnet-5-new",   # not in the registry yet
    provider="anthropic",          # tells Hopper which adapter to use
    messages=[CanonicalMessage(role="user", content="Hello!")],
)
```

Use `extra_params` to pass any parameters alongside it:

```python
request = CanonicalRequest(
    model="claude-sonnet-5-new",
    provider="anthropic",
    messages=[...],
    extra_params={"temperature": 0.7, "top_p": 0.9},
)
```

`extra_params` works for registered models too — anything in there is forwarded
to the provider API without filtering.

## Smoke tests

Hopper never reads API keys from the environment — credentials are always passed explicitly by the caller. This keeps secret management entirely outside the library.

The smoke test is the one exception: it's a developer tool for verifying real API connectivity, so it reads keys from a local `.env` file that is never committed.

**Setup:**

```bash
cp .env.example .env
# fill in keys for the providers you want to test:
#   ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY,
#   TOGETHER_API_KEY, PERPLEXITY_API_KEY, XAI_API_KEY
```

Providers without a key are skipped automatically.

**Run:**

```bash
uv run python tests/smoke_test.py              # all sections (basic + image + multi-turn)
uv run python tests/smoke_test.py --stream     # streaming mode
uv run python tests/smoke_test.py --no-image   # skip image tests
uv run python tests/smoke_test.py --no-multi   # skip multi-turn tests
```

The image test uses `tests/assets/image_example.jpeg` and verifies that models can count the five asterisk markers in the image.

## Unit tests

```bash
uv run pytest
```

## Adding a provider

1. Add `hopper/models/<provider>.yaml`
2. Add `hopper/adapters/<provider>.py` exposing an `ADAPTER` instance

The router picks them up automatically — no other files need to change.
