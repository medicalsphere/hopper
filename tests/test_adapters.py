"""
Adapter tests — each test specifies what a correctly-implemented adapter must do.

Provider SDKs are mocked so these tests never make real API calls.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hopper.types import (
    CanonicalMessage,
    CanonicalRequest,
    Credentials,
    ImagePart,
    ModelEntry,
    ResponseEnvelope,
    RetryConfig,
    TextPart,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _entry(provider: str, model_id: str, **kwargs) -> ModelEntry:
    return ModelEntry(
        id=model_id,
        provider=provider,
        supported_params=kwargs.get("supported_params", ["temperature", "max_tokens"]),
        unsupported_params=kwargs.get("unsupported_params", []),
        defaults=kwargs.get("defaults", {}),
    )


def _req(model: str, content: str = "Hello", **kwargs) -> CanonicalRequest:
    return CanonicalRequest(
        model=model,
        messages=[CanonicalMessage(role="user", content=content)],
        **kwargs,
    )


CREDS = Credentials(api_key="test-key")


# ---------------------------------------------------------------------------
# Anthropic adapter
# ---------------------------------------------------------------------------

class TestAnthropicAdapter:

    def _mock_response(self, text: str = "Hi there!") -> MagicMock:
        block = MagicMock()
        block.type = "text"
        block.text = text
        resp = MagicMock()
        resp.content = [block]
        resp.stop_reason = "end_turn"
        resp.usage = MagicMock(input_tokens=10, output_tokens=4)
        resp.model_dump.return_value = {"id": "msg_abc"}
        return resp

    async def test_returns_response_envelope(self):
        from hopper.adapters.anthropic import ADAPTER

        mock_sdk = MagicMock()
        mock_sdk.AsyncAnthropic.return_value.messages.create = AsyncMock(
            return_value=self._mock_response()
        )

        with patch("hopper.adapters.anthropic._sdk", mock_sdk):
            envelope = await ADAPTER.complete(
                request=_req("claude-sonnet-4-6"),
                model_entry=_entry("anthropic", "claude-sonnet-4-6"),
                credentials=CREDS,
                params={"max_tokens": 1024},
                resolution_log=[],
                include_raw=False,
            )

        assert isinstance(envelope, ResponseEnvelope)
        assert envelope.response.content == "Hi there!"
        assert envelope.provider == "anthropic"
        assert envelope.model_id == "claude-sonnet-4-6"

    async def test_uses_model_entry_id_not_request_model(self):
        """The adapter must send model_entry.id to the API, never request.model."""
        from hopper.adapters.anthropic import ADAPTER

        mock_sdk = MagicMock()
        create_mock = AsyncMock(return_value=self._mock_response())
        mock_sdk.AsyncAnthropic.return_value.messages.create = create_mock

        with patch("hopper.adapters.anthropic._sdk", mock_sdk):
            await ADAPTER.complete(
                request=_req("claude-sonnet"),   # alias
                model_entry=_entry("anthropic", "claude-sonnet-4-6"),
                credentials=CREDS,
                params={},
                resolution_log=[],
                include_raw=False,
            )

        call_kwargs = create_mock.call_args.kwargs
        assert call_kwargs["model"] == "claude-sonnet-4-6"  # API string, not alias

    async def test_injects_system_prompt(self):
        from hopper.adapters.anthropic import ADAPTER

        mock_sdk = MagicMock()
        create_mock = AsyncMock(return_value=self._mock_response())
        mock_sdk.AsyncAnthropic.return_value.messages.create = create_mock

        with patch("hopper.adapters.anthropic._sdk", mock_sdk):
            await ADAPTER.complete(
                request=_req("claude-sonnet-4-6", system="You are a doctor."),
                model_entry=_entry("anthropic", "claude-sonnet-4-6"),
                credentials=CREDS,
                params={},
                resolution_log=[],
                include_raw=False,
            )

        assert create_mock.call_args.kwargs.get("system") == "You are a doctor."

    async def test_envelope_is_json_serialisable(self):
        from hopper.adapters.anthropic import ADAPTER

        mock_sdk = MagicMock()
        mock_sdk.AsyncAnthropic.return_value.messages.create = AsyncMock(
            return_value=self._mock_response()
        )

        with patch("hopper.adapters.anthropic._sdk", mock_sdk):
            envelope = await ADAPTER.complete(
                request=_req("claude-sonnet-4-6"),
                model_entry=_entry("anthropic", "claude-sonnet-4-6"),
                credentials=CREDS,
                params={"max_tokens": 512},
                resolution_log=["Applied default max_tokens=8096."],
                include_raw=False,
            )

        import json
        json.loads(envelope.to_json())  # must not raise

    async def test_raw_populated_when_include_raw_true(self):
        from hopper.adapters.anthropic import ADAPTER

        mock_sdk = MagicMock()
        mock_sdk.AsyncAnthropic.return_value.messages.create = AsyncMock(
            return_value=self._mock_response()
        )

        with patch("hopper.adapters.anthropic._sdk", mock_sdk):
            envelope = await ADAPTER.complete(
                request=_req("claude-sonnet-4-6"),
                model_entry=_entry("anthropic", "claude-sonnet-4-6"),
                credentials=CREDS,
                params={},
                resolution_log=[],
                include_raw=True,
            )

        assert envelope.raw is not None

    async def test_raw_absent_by_default(self):
        from hopper.adapters.anthropic import ADAPTER

        mock_sdk = MagicMock()
        mock_sdk.AsyncAnthropic.return_value.messages.create = AsyncMock(
            return_value=self._mock_response()
        )

        with patch("hopper.adapters.anthropic._sdk", mock_sdk):
            envelope = await ADAPTER.complete(
                request=_req("claude-sonnet-4-6"),
                model_entry=_entry("anthropic", "claude-sonnet-4-6"),
                credentials=CREDS,
                params={},
                resolution_log=[],
                include_raw=False,
            )

        assert envelope.raw is None

    async def test_raises_import_error_when_sdk_missing(self):
        from hopper.adapters.anthropic import ADAPTER

        with patch("hopper.adapters.anthropic._sdk", None):
            with pytest.raises(ImportError, match="anthropic"):
                await ADAPTER.complete(
                    request=_req("claude-sonnet-4-6"),
                    model_entry=_entry("anthropic", "claude-sonnet-4-6"),
                    credentials=CREDS,
                    params={},
                    resolution_log=[],
                    include_raw=False,
                )

    async def test_builds_multimodal_message(self):
        from hopper.adapters.anthropic import ADAPTER

        mock_sdk = MagicMock()
        create_mock = AsyncMock(return_value=self._mock_response())
        mock_sdk.AsyncAnthropic.return_value.messages.create = create_mock

        msg = CanonicalMessage(
            role="user",
            content=[
                TextPart(text="What is in this image?"),
                ImagePart(data="abc123", media_type="image/jpeg"),
            ],
        )

        with patch("hopper.adapters.anthropic._sdk", mock_sdk):
            await ADAPTER.complete(
                request=CanonicalRequest(
                    model="claude-sonnet-4-6",
                    messages=[msg],
                ),
                model_entry=_entry("anthropic", "claude-sonnet-4-6"),
                credentials=CREDS,
                params={},
                resolution_log=[],
                include_raw=False,
            )

        sent_messages = create_mock.call_args.kwargs["messages"]
        assert len(sent_messages) == 1
        parts = sent_messages[0]["content"]
        assert any(p.get("type") == "text" for p in parts)
        assert any(p.get("type") == "image" for p in parts)

    async def test_thinking_param_forwarded_to_api(self):
        """thinking dict must be passed straight through to the Messages API."""
        from hopper.adapters.anthropic import ADAPTER

        mock_sdk = MagicMock()
        create_mock = AsyncMock(return_value=self._mock_response())
        mock_sdk.AsyncAnthropic.return_value.messages.create = create_mock

        thinking = {"type": "adaptive"}
        with patch("hopper.adapters.anthropic._sdk", mock_sdk):
            await ADAPTER.complete(
                request=_req("claude-sonnet-4-6"),
                model_entry=_entry("anthropic", "claude-sonnet-4-6"),
                credentials=CREDS,
                params={"thinking": thinking},
                resolution_log=[],
                include_raw=False,
            )

        call_kwargs = create_mock.call_args.kwargs
        assert call_kwargs.get("thinking") == thinking

    async def test_thinking_blocks_ignored_in_response(self):
        """Thinking blocks in the response must not appear in envelope content."""
        from hopper.adapters.anthropic import ADAPTER

        thinking_block = MagicMock()
        thinking_block.type = "thinking"
        thinking_block.thinking = "Let me reason through this..."
        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "The answer is 42."

        mock_resp = MagicMock()
        mock_resp.content = [thinking_block, text_block]
        mock_resp.stop_reason = "end_turn"
        mock_resp.usage = MagicMock(input_tokens=100, output_tokens=50)
        mock_resp.model_dump.return_value = {}

        mock_sdk = MagicMock()
        mock_sdk.AsyncAnthropic.return_value.messages.create = AsyncMock(return_value=mock_resp)

        with patch("hopper.adapters.anthropic._sdk", mock_sdk):
            envelope = await ADAPTER.complete(
                request=_req("claude-sonnet-4-6"),
                model_entry=_entry("anthropic", "claude-sonnet-4-6"),
                credentials=CREDS,
                params={},
                resolution_log=[],
                include_raw=False,
            )

        assert envelope.response.content == "The answer is 42."

    def test_is_retryable_returns_bool(self):
        from hopper.adapters.anthropic import ADAPTER
        result = ADAPTER.is_retryable(ValueError("random error"))
        assert isinstance(result, bool)
        assert result is False


# ---------------------------------------------------------------------------
# OpenAI adapter — Responses API
# ---------------------------------------------------------------------------

class TestOpenAIAdapter:
    """
    OpenAI uses the Responses API (client.responses.create / .stream).
    This is distinct from the Chat Completions API used by Together/Perplexity/Grok.
    """

    def _mock_response(self, text: str = "Response") -> MagicMock:
        resp = MagicMock()
        resp.output_text = text
        resp.status = "completed"
        resp.incomplete_details = None
        resp.usage = MagicMock(input_tokens=8, output_tokens=3, total_tokens=11)
        resp.model_dump.return_value = {}
        return resp

    async def test_complete_returns_envelope(self):
        from hopper.adapters.openai import ADAPTER

        mock_client_cls = MagicMock()
        mock_client_cls.return_value.responses.create = AsyncMock(
            return_value=self._mock_response()
        )

        with patch("hopper.adapters.openai.AsyncOpenAI", mock_client_cls):
            envelope = await ADAPTER.complete(
                request=_req("gpt-4o"),
                model_entry=_entry("openai", "gpt-4o"),
                credentials=CREDS,
                params={"max_tokens": 256},
                resolution_log=[],
                include_raw=False,
            )

        assert envelope.provider == "openai"
        assert envelope.model_id == "gpt-4o"
        assert envelope.response.content == "Response"
        assert envelope.response.finish_reason == "stop"

    async def test_translates_max_tokens_to_max_output_tokens(self):
        """Responses API uses max_output_tokens, not max_tokens."""
        from hopper.adapters.openai import ADAPTER

        mock_client_cls = MagicMock()
        create_mock = AsyncMock(return_value=self._mock_response())
        mock_client_cls.return_value.responses.create = create_mock

        with patch("hopper.adapters.openai.AsyncOpenAI", mock_client_cls):
            await ADAPTER.complete(
                request=_req("gpt-4o"),
                model_entry=_entry("openai", "gpt-4o"),
                credentials=CREDS,
                params={"max_tokens": 512},
                resolution_log=[],
                include_raw=False,
            )

        call_kwargs = create_mock.call_args.kwargs
        assert call_kwargs.get("max_output_tokens") == 512
        assert "max_tokens" not in call_kwargs

    async def test_uses_model_entry_id_not_alias(self):
        from hopper.adapters.openai import ADAPTER

        mock_client_cls = MagicMock()
        create_mock = AsyncMock(return_value=self._mock_response())
        mock_client_cls.return_value.responses.create = create_mock

        with patch("hopper.adapters.openai.AsyncOpenAI", mock_client_cls):
            await ADAPTER.complete(
                request=_req("gpt-4o-latest"),   # alias
                model_entry=_entry("openai", "gpt-4o"),
                credentials=CREDS,
                params={},
                resolution_log=[],
                include_raw=False,
            )

        call_kwargs = create_mock.call_args.kwargs
        assert call_kwargs["model"] == "gpt-4o"

    async def test_system_prompt_sent_as_instructions(self):
        """Responses API uses `instructions`, not a system role message."""
        from hopper.adapters.openai import ADAPTER

        mock_client_cls = MagicMock()
        create_mock = AsyncMock(return_value=self._mock_response())
        mock_client_cls.return_value.responses.create = create_mock

        with patch("hopper.adapters.openai.AsyncOpenAI", mock_client_cls):
            await ADAPTER.complete(
                request=_req("gpt-4o", system="You are helpful."),
                model_entry=_entry("openai", "gpt-4o"),
                credentials=CREDS,
                params={},
                resolution_log=[],
                include_raw=False,
            )

        call_kwargs = create_mock.call_args.kwargs
        assert call_kwargs.get("instructions") == "You are helpful."
        # System prompt must NOT appear as a message in `input`
        for msg in call_kwargs.get("input", []):
            assert msg.get("role") != "system"

    async def test_uses_input_not_messages(self):
        """Responses API uses `input`, not `messages`."""
        from hopper.adapters.openai import ADAPTER

        mock_client_cls = MagicMock()
        create_mock = AsyncMock(return_value=self._mock_response())
        mock_client_cls.return_value.responses.create = create_mock

        with patch("hopper.adapters.openai.AsyncOpenAI", mock_client_cls):
            await ADAPTER.complete(
                request=_req("gpt-4o", content="Hello"),
                model_entry=_entry("openai", "gpt-4o"),
                credentials=CREDS,
                params={},
                resolution_log=[],
                include_raw=False,
            )

        call_kwargs = create_mock.call_args.kwargs
        assert "input" in call_kwargs
        assert "messages" not in call_kwargs

    async def test_uses_openai_com_base_url(self):
        from hopper.adapters.openai import ADAPTER

        mock_client_cls = MagicMock()
        mock_client_cls.return_value.responses.create = AsyncMock(
            return_value=self._mock_response()
        )

        with patch("hopper.adapters.openai.AsyncOpenAI", mock_client_cls):
            await ADAPTER.complete(
                request=_req("gpt-4o"),
                model_entry=_entry("openai", "gpt-4o"),
                credentials=CREDS,
                params={},
                resolution_log=[],
                include_raw=False,
            )

        init_kwargs = mock_client_cls.call_args.kwargs
        assert "openai.com" in init_kwargs.get("base_url", "")

    async def test_reasoning_param_forwarded_to_api(self):
        """reasoning dict must be passed straight through to the Responses API."""
        from hopper.adapters.openai import ADAPTER

        mock_client_cls = MagicMock()
        create_mock = AsyncMock(return_value=self._mock_response())
        mock_client_cls.return_value.responses.create = create_mock

        with patch("hopper.adapters.openai.AsyncOpenAI", mock_client_cls):
            await ADAPTER.complete(
                request=_req("gpt-4o", reasoning={"effort": "high"}),
                model_entry=_entry("openai", "gpt-4o"),
                credentials=CREDS,
                params={"reasoning": {"effort": "high"}},
                resolution_log=[],
                include_raw=False,
            )

        call_kwargs = create_mock.call_args.kwargs
        assert call_kwargs.get("reasoning") == {"effort": "high"}

    async def test_raises_import_error_when_sdk_missing(self):
        from hopper.adapters.openai import ADAPTER

        with patch("hopper.adapters.openai.AsyncOpenAI", None):
            with pytest.raises(ImportError, match="openai"):
                await ADAPTER.complete(
                    request=_req("gpt-4o"),
                    model_entry=_entry("openai", "gpt-4o"),
                    credentials=CREDS,
                    params={},
                    resolution_log=[],
                    include_raw=False,
                )


# ---------------------------------------------------------------------------
# Perplexity adapter — OpenAI SDK against Perplexity's Responses-compatible endpoint
# ---------------------------------------------------------------------------

class TestPerplexityAdapter:
    """
    Perplexity exposes a /v1/responses endpoint compatible with the OpenAI Responses API.
    Uses AsyncOpenAI with base_url="https://api.perplexity.ai/v1".
    System prompt → instructions=; max_tokens → max_output_tokens.
    """

    def _mock_response(self, text: str = "Perplexity reply") -> MagicMock:
        resp = MagicMock()
        resp.output_text = text
        resp.status = "completed"
        resp.incomplete_details = None
        resp.usage = MagicMock(input_tokens=8, output_tokens=3, total_tokens=11)
        resp.model_dump.return_value = {}
        return resp

    async def test_complete_returns_envelope(self):
        from hopper.adapters.perplexity import ADAPTER

        mock_client_cls = MagicMock()
        mock_client_cls.return_value.responses.create = AsyncMock(
            return_value=self._mock_response()
        )

        with patch("hopper.adapters.perplexity.AsyncOpenAI", mock_client_cls):
            envelope = await ADAPTER.complete(
                request=_req("perplexity/sonar"),
                model_entry=_entry("perplexity", "perplexity/sonar"),
                credentials=CREDS,
                params={"max_tokens": 256},
                resolution_log=[],
                include_raw=False,
            )

        assert envelope.provider == "perplexity"
        assert envelope.model_id == "perplexity/sonar"
        assert envelope.response.content == "Perplexity reply"

    async def test_uses_perplexity_base_url(self):
        """Client must be initialised with the Perplexity base URL."""
        from hopper.adapters.perplexity import ADAPTER, _BASE_URL

        mock_client_cls = MagicMock()
        mock_client_cls.return_value.responses.create = AsyncMock(
            return_value=self._mock_response()
        )

        with patch("hopper.adapters.perplexity.AsyncOpenAI", mock_client_cls):
            await ADAPTER.complete(
                request=_req("perplexity/sonar"),
                model_entry=_entry("perplexity", "perplexity/sonar"),
                credentials=CREDS,
                params={},
                resolution_log=[],
                include_raw=False,
            )

        init_kwargs = mock_client_cls.call_args.kwargs
        assert init_kwargs.get("base_url") == _BASE_URL

    async def test_system_prompt_sent_as_instructions(self):
        """System prompt must be passed as instructions=, not embedded in input."""
        from hopper.adapters.perplexity import ADAPTER

        mock_client_cls = MagicMock()
        create_mock = AsyncMock(return_value=self._mock_response())
        mock_client_cls.return_value.responses.create = create_mock

        with patch("hopper.adapters.perplexity.AsyncOpenAI", mock_client_cls):
            await ADAPTER.complete(
                request=_req("perplexity/sonar", system="Be concise."),
                model_entry=_entry("perplexity", "perplexity/sonar"),
                credentials=CREDS,
                params={},
                resolution_log=[],
                include_raw=False,
            )

        call_kwargs = create_mock.call_args.kwargs
        assert call_kwargs.get("instructions") == "Be concise."
        for msg in call_kwargs.get("input", []):
            assert msg.get("role") != "system"

    async def test_max_tokens_translated(self):
        """max_tokens must be sent as max_output_tokens."""
        from hopper.adapters.perplexity import ADAPTER

        mock_client_cls = MagicMock()
        create_mock = AsyncMock(return_value=self._mock_response())
        mock_client_cls.return_value.responses.create = create_mock

        with patch("hopper.adapters.perplexity.AsyncOpenAI", mock_client_cls):
            await ADAPTER.complete(
                request=_req("perplexity/sonar"),
                model_entry=_entry("perplexity", "perplexity/sonar"),
                credentials=CREDS,
                params={"max_tokens": 512},
                resolution_log=[],
                include_raw=False,
            )

        call_kwargs = create_mock.call_args.kwargs
        assert call_kwargs.get("max_output_tokens") == 512
        assert "max_tokens" not in call_kwargs


# ---------------------------------------------------------------------------
# Together AI adapter — dedicated `together` SDK
# ---------------------------------------------------------------------------

class TestTogetherAdapter:
    """
    Together AI has its own SDK (pip install together) with AsyncTogether.
    Message format is identical to OpenAI chat completions.
    """

    def _mock_together_response(self, text: str = "Together reply") -> MagicMock:
        choice = MagicMock()
        choice.message.content = text
        choice.finish_reason = "stop"
        resp = MagicMock()
        resp.choices = [choice]
        resp.usage = MagicMock(prompt_tokens=8, completion_tokens=3, total_tokens=11)
        resp.model_dump.return_value = {}
        return resp

    async def test_complete_returns_envelope(self):
        from hopper.adapters.together import ADAPTER

        mock_client_cls = MagicMock()
        mock_client_cls.return_value.chat.completions.create = AsyncMock(
            return_value=self._mock_together_response()
        )

        with patch("hopper.adapters.together.AsyncTogether", mock_client_cls):
            envelope = await ADAPTER.complete(
                request=_req("meta-llama/Llama-3.3-70B-Instruct-Turbo"),
                model_entry=_entry("together", "meta-llama/Llama-3.3-70B-Instruct-Turbo"),
                credentials=CREDS,
                params={"max_tokens": 256},
                resolution_log=[],
                include_raw=False,
            )

        assert envelope.provider == "together"
        assert envelope.response.content == "Together reply"

    async def test_uses_together_client_not_openai(self):
        """Must use AsyncTogether, not AsyncOpenAI or shared helper client."""
        from hopper.adapters.together import ADAPTER

        mock_together_cls = MagicMock()
        mock_together_cls.return_value.chat.completions.create = AsyncMock(
            return_value=self._mock_together_response()
        )
        mock_openai_cls = MagicMock()

        with patch("hopper.adapters.together.AsyncTogether", mock_together_cls), \
             patch("hopper.adapters.shared.AsyncOpenAI", mock_openai_cls):
            await ADAPTER.complete(
                request=_req("meta-llama/Llama-3.3-70B-Instruct-Turbo"),
                model_entry=_entry("together", "meta-llama/Llama-3.3-70B-Instruct-Turbo"),
                credentials=CREDS,
                params={},
                resolution_log=[],
                include_raw=False,
            )

        mock_together_cls.assert_called_once()
        mock_openai_cls.assert_not_called()

    async def test_system_prompt_prepended_as_system_message(self):
        from hopper.adapters.together import ADAPTER

        mock_client_cls = MagicMock()
        create_mock = AsyncMock(return_value=self._mock_together_response())
        mock_client_cls.return_value.chat.completions.create = create_mock

        with patch("hopper.adapters.together.AsyncTogether", mock_client_cls):
            await ADAPTER.complete(
                request=_req("meta-llama/Llama-3.3-70B-Instruct-Turbo", system="Be concise."),
                model_entry=_entry("together", "meta-llama/Llama-3.3-70B-Instruct-Turbo"),
                credentials=CREDS,
                params={},
                resolution_log=[],
                include_raw=False,
            )

        messages = create_mock.call_args.kwargs["messages"]
        assert messages[0] == {"role": "system", "content": "Be concise."}

    async def test_raises_import_error_when_sdk_missing(self):
        from hopper.adapters.together import ADAPTER

        with patch("hopper.adapters.together.AsyncTogether", None):
            with pytest.raises(ImportError, match="together"):
                await ADAPTER.complete(
                    request=_req("meta-llama/Llama-3.3-70B-Instruct-Turbo"),
                    model_entry=_entry("together", "meta-llama/Llama-3.3-70B-Instruct-Turbo"),
                    credentials=CREDS,
                    params={},
                    resolution_log=[],
                    include_raw=False,
                )


# ---------------------------------------------------------------------------
# Grok adapter — xAI Responses API via openai package
# ---------------------------------------------------------------------------

class TestGrokAdapter:
    """
    xAI uses the Responses API (client.responses.create) via the openai package
    pointed at https://api.x.ai/v1. System prompt is embedded in the input
    array as a system-role message, not passed via `instructions`.
    """

    def _mock_response(self, text: str = "Grok reply") -> MagicMock:
        resp = MagicMock()
        resp.output_text = text
        resp.status = "completed"
        resp.incomplete_details = None
        resp.usage = MagicMock(input_tokens=8, output_tokens=3, total_tokens=11)
        resp.model_dump.return_value = {}
        return resp

    async def test_complete_returns_envelope(self):
        from hopper.adapters.grok import ADAPTER

        mock_client_cls = MagicMock()
        mock_client_cls.return_value.responses.create = AsyncMock(
            return_value=self._mock_response()
        )

        with patch("hopper.adapters.grok.AsyncOpenAI", mock_client_cls):
            envelope = await ADAPTER.complete(
                request=_req("grok-3"),
                model_entry=_entry("grok", "grok-3"),
                credentials=CREDS,
                params={"max_tokens": 256},
                resolution_log=[],
                include_raw=False,
            )

        assert envelope.provider == "grok"
        assert envelope.response.content == "Grok reply"
        assert envelope.response.finish_reason == "stop"

    async def test_uses_x_ai_base_url(self):
        from hopper.adapters.grok import ADAPTER

        mock_client_cls = MagicMock()
        mock_client_cls.return_value.responses.create = AsyncMock(
            return_value=self._mock_response()
        )

        with patch("hopper.adapters.grok.AsyncOpenAI", mock_client_cls):
            await ADAPTER.complete(
                request=_req("grok-3"),
                model_entry=_entry("grok", "grok-3"),
                credentials=CREDS,
                params={},
                resolution_log=[],
                include_raw=False,
            )

        init_kwargs = mock_client_cls.call_args.kwargs
        assert "x.ai" in init_kwargs.get("base_url", "")

    async def test_uses_responses_api_not_chat_completions(self):
        """Must call client.responses.create(), not client.chat.completions.create()."""
        from hopper.adapters.grok import ADAPTER

        mock_client_cls = MagicMock()
        responses_create = AsyncMock(return_value=self._mock_response())
        mock_client_cls.return_value.responses.create = responses_create

        with patch("hopper.adapters.grok.AsyncOpenAI", mock_client_cls):
            await ADAPTER.complete(
                request=_req("grok-3"),
                model_entry=_entry("grok", "grok-3"),
                credentials=CREDS,
                params={},
                resolution_log=[],
                include_raw=False,
            )

        responses_create.assert_called_once()
        mock_client_cls.return_value.chat.completions.create.assert_not_called()

    async def test_uses_input_not_messages(self):
        """Responses API uses `input`, not `messages`."""
        from hopper.adapters.grok import ADAPTER

        mock_client_cls = MagicMock()
        create_mock = AsyncMock(return_value=self._mock_response())
        mock_client_cls.return_value.responses.create = create_mock

        with patch("hopper.adapters.grok.AsyncOpenAI", mock_client_cls):
            await ADAPTER.complete(
                request=_req("grok-3"),
                model_entry=_entry("grok", "grok-3"),
                credentials=CREDS,
                params={},
                resolution_log=[],
                include_raw=False,
            )

        call_kwargs = create_mock.call_args.kwargs
        assert "input" in call_kwargs
        assert "messages" not in call_kwargs

    async def test_system_prompt_embedded_in_input_not_instructions(self):
        """xAI embeds system as a role in `input`, unlike OpenAI's `instructions`."""
        from hopper.adapters.grok import ADAPTER

        mock_client_cls = MagicMock()
        create_mock = AsyncMock(return_value=self._mock_response())
        mock_client_cls.return_value.responses.create = create_mock

        with patch("hopper.adapters.grok.AsyncOpenAI", mock_client_cls):
            await ADAPTER.complete(
                request=_req("grok-3", system="You are Grok."),
                model_entry=_entry("grok", "grok-3"),
                credentials=CREDS,
                params={},
                resolution_log=[],
                include_raw=False,
            )

        call_kwargs = create_mock.call_args.kwargs
        assert "instructions" not in call_kwargs
        assert call_kwargs["input"][0] == {"role": "system", "content": "You are Grok."}

    async def test_translates_max_tokens_to_max_output_tokens(self):
        from hopper.adapters.grok import ADAPTER

        mock_client_cls = MagicMock()
        create_mock = AsyncMock(return_value=self._mock_response())
        mock_client_cls.return_value.responses.create = create_mock

        with patch("hopper.adapters.grok.AsyncOpenAI", mock_client_cls):
            await ADAPTER.complete(
                request=_req("grok-3"),
                model_entry=_entry("grok", "grok-3"),
                credentials=CREDS,
                params={"max_tokens": 512},
                resolution_log=[],
                include_raw=False,
            )

        call_kwargs = create_mock.call_args.kwargs
        assert call_kwargs.get("max_output_tokens") == 512
        assert "max_tokens" not in call_kwargs

    async def test_raises_import_error_when_sdk_missing(self):
        from hopper.adapters.grok import ADAPTER

        with patch("hopper.adapters.grok.AsyncOpenAI", None):
            with pytest.raises(ImportError, match="openai"):
                await ADAPTER.complete(
                    request=_req("grok-3"),
                    model_entry=_entry("grok", "grok-3"),
                    credentials=CREDS,
                    params={},
                    resolution_log=[],
                    include_raw=False,
                )


# ---------------------------------------------------------------------------
# Google adapter — google-genai SDK
# ---------------------------------------------------------------------------

class TestGoogleAdapter:
    """
    Google uses the google-genai SDK (pip install google-genai).
    client.aio.models.generate_content() with GenerateContentConfig.
    """

    def _mock_response(self, text: str = "Gemini reply") -> MagicMock:
        candidate = MagicMock()
        candidate.finish_reason = "STOP"
        resp = MagicMock()
        resp.text = text
        resp.candidates = [candidate]
        resp.usage_metadata = MagicMock(
            prompt_token_count=6,
            candidates_token_count=3,
            total_token_count=9,
        )
        return resp

    def _mock_sdk_and_types(self, response: MagicMock):
        mock_sdk = MagicMock()
        mock_sdk.Client.return_value.aio.models.generate_content = AsyncMock(
            return_value=response
        )
        mock_types = MagicMock()
        return mock_sdk, mock_types

    async def test_returns_envelope_with_google_provider(self):
        from hopper.adapters.google import ADAPTER

        resp = self._mock_response()
        mock_sdk, mock_types = self._mock_sdk_and_types(resp)

        with patch("hopper.adapters.google._sdk", mock_sdk), \
             patch("hopper.adapters.google._types", mock_types):
            envelope = await ADAPTER.complete(
                request=_req("gemini-2.0-flash"),
                model_entry=_entry("google", "gemini-2.0-flash"),
                credentials=CREDS,
                params={"max_tokens": 512, "temperature": 0.3},
                resolution_log=[],
                include_raw=False,
            )

        assert envelope.provider == "google"
        assert envelope.response.content == "Gemini reply"

    async def test_translates_max_tokens_to_max_output_tokens(self):
        """GenerateContentConfig uses max_output_tokens, not max_tokens."""
        from hopper.adapters.google import ADAPTER

        resp = self._mock_response()
        mock_sdk, mock_types = self._mock_sdk_and_types(resp)

        with patch("hopper.adapters.google._sdk", mock_sdk), \
             patch("hopper.adapters.google._types", mock_types):
            await ADAPTER.complete(
                request=_req("gemini-2.0-flash"),
                model_entry=_entry("google", "gemini-2.0-flash"),
                credentials=CREDS,
                params={"max_tokens": 1024},
                resolution_log=[],
                include_raw=False,
            )

        # GenerateContentConfig must be called with max_output_tokens, not max_tokens
        config_call_kwargs = mock_types.GenerateContentConfig.call_args.kwargs
        assert config_call_kwargs.get("max_output_tokens") == 1024
        assert "max_tokens" not in config_call_kwargs

    async def test_system_instruction_passed_in_config(self):
        """System prompt goes in GenerateContentConfig.system_instruction."""
        from hopper.adapters.google import ADAPTER

        resp = self._mock_response()
        mock_sdk, mock_types = self._mock_sdk_and_types(resp)

        with patch("hopper.adapters.google._sdk", mock_sdk), \
             patch("hopper.adapters.google._types", mock_types):
            await ADAPTER.complete(
                request=_req("gemini-2.0-flash", system="You are a doctor."),
                model_entry=_entry("google", "gemini-2.0-flash"),
                credentials=CREDS,
                params={},
                resolution_log=[],
                include_raw=False,
            )

        config_call_kwargs = mock_types.GenerateContentConfig.call_args.kwargs
        assert config_call_kwargs.get("system_instruction") == "You are a doctor."

    async def test_uses_aio_client_for_async(self):
        """Must use client.aio.models.generate_content, not sync client."""
        from hopper.adapters.google import ADAPTER

        resp = self._mock_response()
        mock_sdk, mock_types = self._mock_sdk_and_types(resp)
        aio_mock = mock_sdk.Client.return_value.aio.models.generate_content

        with patch("hopper.adapters.google._sdk", mock_sdk), \
             patch("hopper.adapters.google._types", mock_types):
            await ADAPTER.complete(
                request=_req("gemini-2.0-flash"),
                model_entry=_entry("google", "gemini-2.0-flash"),
                credentials=CREDS,
                params={},
                resolution_log=[],
                include_raw=False,
            )

        aio_mock.assert_called_once()

    async def test_raises_import_error_when_sdk_missing(self):
        from hopper.adapters.google import ADAPTER

        with patch("hopper.adapters.google._sdk", None):
            with pytest.raises(ImportError, match="google-genai"):
                await ADAPTER.complete(
                    request=_req("gemini-2.0-flash"),
                    model_entry=_entry("google", "gemini-2.0-flash"),
                    credentials=CREDS,
                    params={},
                    resolution_log=[],
                    include_raw=False,
                )


# ---------------------------------------------------------------------------
# hopper.complete() — integration with retry logic
# ---------------------------------------------------------------------------

class TestCompleteRetryLoop:

    async def test_retries_on_transient_error(self):
        """complete() retries when adapter.is_retryable() returns True."""
        import hopper
        from hopper.adapters.anthropic import ADAPTER

        mock_sdk = MagicMock()
        good_resp = MagicMock()
        good_block = MagicMock()
        good_block.type = "text"
        good_block.text = "OK"
        good_resp.content = [good_block]
        good_resp.stop_reason = "end_turn"
        good_resp.usage = MagicMock(input_tokens=5, output_tokens=2)
        good_resp.model_dump.return_value = {}

        retryable_error = MagicMock(spec=Exception)

        call_count = 0

        async def flaky_create(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise retryable_error
            return good_resp

        mock_sdk.AsyncAnthropic.return_value.messages.create = flaky_create

        # Patch is_retryable to return True for our mock error
        with patch("hopper.adapters.anthropic._sdk", mock_sdk), \
             patch.object(ADAPTER, "is_retryable", return_value=True), \
             patch("asyncio.sleep", new=AsyncMock()):

            envelope = await hopper.complete(
                request=CanonicalRequest(
                    model="claude-sonnet-4-6",
                    messages=[CanonicalMessage(role="user", content="hello")],
                ),
                credentials=CREDS,
                retry_config=RetryConfig(max_attempts=2, base_delay=0.01),
            )

        assert call_count == 2
        assert envelope.response.content == "OK"

    async def test_does_not_retry_non_retryable_error(self):
        import hopper
        from hopper.adapters.anthropic import ADAPTER

        mock_sdk = MagicMock()
        mock_sdk.AsyncAnthropic.return_value.messages.create = AsyncMock(
            side_effect=ValueError("bad request")
        )

        call_count = 0
        original_create = mock_sdk.AsyncAnthropic.return_value.messages.create

        async def counting_create(**kwargs):
            nonlocal call_count
            call_count += 1
            raise ValueError("bad request")

        mock_sdk.AsyncAnthropic.return_value.messages.create = counting_create

        with patch("hopper.adapters.anthropic._sdk", mock_sdk), \
             patch.object(ADAPTER, "is_retryable", return_value=False):

            with pytest.raises(ValueError, match="bad request"):
                await hopper.complete(
                    request=CanonicalRequest(
                        model="claude-sonnet-4-6",
                        messages=[CanonicalMessage(role="user", content="hello")],
                    ),
                    credentials=CREDS,
                    retry_config=RetryConfig(max_attempts=3),
                )

        assert call_count == 1  # no retries
