"""
Smoke test — makes real API calls to each provider whose key is set.

Usage:
    1. Copy .env.example to .env in the project root and fill in your keys.
    2. Run: uv run python tests/smoke_test.py [--stream]

Flags:
    --stream      Test the streaming path instead of complete().
    --no-image    Skip the image-input test.
    --no-multi    Skip the multi-turn conversation test.

Keys can also be set as environment variables directly. A .env file in the
project root (or the tests/ directory) is loaded automatically if present.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import os
import pathlib
import sys
import time


def _load_dotenv() -> None:
    """Load key=value pairs from a .env file into os.environ (no-op if absent)."""
    for candidate in (
        pathlib.Path(__file__).parent.parent / ".env",  # project root
        pathlib.Path(__file__).parent / ".env",         # tests/
    ):
        if not candidate.exists():
            continue
        with candidate.open() as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip("'\"")
                os.environ.setdefault(key, value)
        break


_load_dotenv()

import hopper
from hopper import (
    CanonicalMessage,
    CanonicalRequest,
    Credentials,
    ImagePart,
    TextPart,
)

# ---------------------------------------------------------------------------
# Test image — CT scan with 5 white asterisk (*) markers
# ---------------------------------------------------------------------------
_IMAGE_PATH      = pathlib.Path(__file__).parent / "assets" / "image_example.jpeg"
_IMAGE_B64       = base64.b64encode(_IMAGE_PATH.read_bytes()).decode() if _IMAGE_PATH.exists() else None
_IMAGE_MEDIA     = "image/jpeg"
_IMAGE_QUESTION  = "How many star symbols (asterisks) are visible in this image? Answer with a single number only."
_IMAGE_EXPECTED  = "5"

# ---------------------------------------------------------------------------
# Provider table
# (label, model_id, env_var, supports_vision)
# ---------------------------------------------------------------------------
PROVIDERS = [
    ("anthropic",  "claude-sonnet-4-6",       "ANTHROPIC_API_KEY", True),
    ("openai",     "gpt-5.4-2026-03-05", "OPENAI_API_KEY",    True),
    ("google",     "gemini-3-flash-preview",   "GEMINI_API_KEY",    True),
    ("together",   "zai-org/GLM-5",            "TOGETHER_API_KEY",  True),
    ("perplexity", "perplexity/sonar",           "PERPLEXITY_API_KEY", True),
    ("grok",       "grok-4.20",                "XAI_API_KEY",       True),
]

GREEN  = "\033[32m"
RED    = "\033[31m"
YELLOW = "\033[33m"
CYAN   = "\033[36m"
RESET  = "\033[0m"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _token_info(usage) -> str:
    return f"  ({usage.input_tokens}→{usage.output_tokens} tokens)" if usage else ""


def _print_ok(tag: str, reply: str, elapsed: float, usage=None) -> None:
    print(f"{GREEN}[OK]{RESET}   {tag}: {reply!r}  {elapsed:.0f}ms{_token_info(usage)}")


def _print_fail(tag: str, exc: Exception, elapsed: float) -> None:
    print(f"{RED}[FAIL]{RESET} {tag}: {exc}  {elapsed:.0f}ms")


def _print_skip(tag: str, reason: str) -> None:
    print(f"{YELLOW}[SKIP]{RESET} {tag} — {reason}")


def _section(title: str) -> None:
    print(f"\n{CYAN}{title}{RESET}")


# ---------------------------------------------------------------------------
# Test: basic complete / stream
# ---------------------------------------------------------------------------

def _basic_req(model: str, label: str = "") -> CanonicalRequest:
    kwargs = {}
    if label == "openai":
        kwargs["reasoning"] = {"effort": "high"}
    if label == "anthropic":
        kwargs["thinking"] = {"type": "enabled", "budget_tokens": 5000}
    return CanonicalRequest(
        model=model,
        messages=[CanonicalMessage(role="user", content="Say hello in one word. No punctuation.")],
        **kwargs,
    )


async def _test_complete(tag: str, label: str, model: str, creds: Credentials) -> bool:
    start = time.monotonic()
    try:
        env = await hopper.complete(_basic_req(model, label), creds)
        elapsed = (time.monotonic() - start) * 1000
        _print_ok(tag, env.response.content.strip(), elapsed, env.usage)
        return True
    except Exception as exc:
        _print_fail(tag, exc, (time.monotonic() - start) * 1000)
        return False


async def _test_stream(tag: str, label: str, model: str, creds: Credentials) -> bool:
    start = time.monotonic()
    try:
        chunks = []
        async for chunk in hopper.stream(_basic_req(model, label), creds):
            chunks.append(chunk)
        elapsed = (time.monotonic() - start) * 1000
        reply = "".join(c.delta for c in chunks).strip()
        final_usage = next((c.usage for c in reversed(chunks) if c.usage), None)
        _print_ok(f"{tag} (stream)", reply, elapsed, final_usage)
        return True
    except Exception as exc:
        _print_fail(f"{tag} (stream)", exc, (time.monotonic() - start) * 1000)
        return False


# ---------------------------------------------------------------------------
# Test: image input
# ---------------------------------------------------------------------------

async def _test_image(tag: str, model: str, creds: Credentials) -> bool:
    if _IMAGE_B64 is None:
        _print_skip(tag, "image_example.jpeg not found in tests/assets/")
        return True  # not a provider failure

    req = CanonicalRequest(
        model=model,
        messages=[
            CanonicalMessage(
                role="user",
                content=[
                    ImagePart(data=_IMAGE_B64, media_type=_IMAGE_MEDIA),
                    TextPart(text=_IMAGE_QUESTION),
                ],
            )
        ],
    )

    start = time.monotonic()
    try:
        env = await hopper.complete(req, creds)
        elapsed = (time.monotonic() - start) * 1000
        reply = env.response.content.strip()
        correct = _IMAGE_EXPECTED in reply
        status = "✓ correct" if correct else f"✗ expected {_IMAGE_EXPECTED!r}"
        _print_ok(f"{tag} (image)", f"{reply}  [{status}]", elapsed, env.usage)
        return correct
    except Exception as exc:
        _print_fail(f"{tag} (image)", exc, (time.monotonic() - start) * 1000)
        return False


# ---------------------------------------------------------------------------
# Test: multi-turn conversation
# ---------------------------------------------------------------------------

async def _test_multiturn(tag: str, model: str, creds: Credentials) -> bool:
    # Turn 1: plant a fact
    turn1_req = CanonicalRequest(
        model=model,
        messages=[
            CanonicalMessage(role="user", content="My name is Alice. Just say 'Got it.' and nothing else."),
        ],
    )

    start = time.monotonic()
    try:
        turn1 = await hopper.complete(turn1_req, creds)
    except Exception as exc:
        _print_fail(f"{tag} (multi-turn turn-1)", exc, (time.monotonic() - start) * 1000)
        return False

    # Turn 2: recall the fact
    turn2_req = CanonicalRequest(
        model=model,
        messages=[
            CanonicalMessage(role="user",      content="My name is Alice. Just say 'Got it.' and nothing else."),
            CanonicalMessage(role="assistant", content=turn1.response.content),
            CanonicalMessage(role="user",      content="What is my name? Answer with my name only, no punctuation."),
        ],
    )

    try:
        turn2 = await hopper.complete(turn2_req, creds)
        elapsed = (time.monotonic() - start) * 1000
        reply = turn2.response.content.strip()
        recalled = "alice" in reply.lower()
        status = "✓ recalled" if recalled else "✗ forgot"
        _print_ok(f"{tag} (multi-turn)", f"{reply}  [{status}]", elapsed, turn2.usage)
        return recalled
    except Exception as exc:
        _print_fail(f"{tag} (multi-turn)", exc, (time.monotonic() - start) * 1000)
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main(use_stream: bool, skip_image: bool, skip_multi: bool) -> None:
    mode = "streaming" if use_stream else "complete"
    print(f"Hopper smoke test — {mode} mode\n")

    ok = skipped = failed = 0

    def record(success: bool) -> None:
        nonlocal ok, skipped, failed
        if success:
            ok += 1
        else:
            failed += 1

    # --- basic ---
    _section("Basic completion")
    for label, model, env_var, _ in PROVIDERS:
        api_key = os.environ.get(env_var)
        tag = f"{label}/{model}"
        if not api_key:
            _print_skip(tag, f"{env_var} not set")
            skipped += 1
            continue
        creds = Credentials(api_key=api_key)
        if use_stream:
            record(await _test_stream(tag, label, model, creds))
        else:
            record(await _test_complete(tag, label, model, creds))

    # --- image ---
    if not skip_image:
        _section("Image input")
        for label, model, env_var, supports_vision in PROVIDERS:
            api_key = os.environ.get(env_var)
            tag = f"{label}/{model}"
            if not api_key:
                _print_skip(tag, f"{env_var} not set")
                skipped += 1
                continue
            if not supports_vision:
                _print_skip(tag, "provider does not support image input")
                skipped += 1
                continue
            record(await _test_image(tag, model, Credentials(api_key=api_key)))

    # --- multi-turn ---
    if not skip_multi:
        _section("Multi-turn conversation")
        for label, model, env_var, _ in PROVIDERS:
            api_key = os.environ.get(env_var)
            tag = f"{label}/{model}"
            if not api_key:
                _print_skip(tag, f"{env_var} not set")
                skipped += 1
                continue
            record(await _test_multiturn(tag, model, Credentials(api_key=api_key)))

    print(f"\n{ok} passed, {failed} failed, {skipped} skipped")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stream",     action="store_true", help="Test streaming instead of complete()")
    parser.add_argument("--no-image",   action="store_true", help="Skip image input tests")
    parser.add_argument("--no-multi",   action="store_true", help="Skip multi-turn conversation tests")
    args = parser.parse_args()
    asyncio.run(main(args.stream, args.no_image, args.no_multi))
