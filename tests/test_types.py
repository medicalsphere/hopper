"""
Type tests — specifications for canonical types and ResponseEnvelope serialisation.
"""
import json

import pytest

from hopper.types import (
    CanonicalMessage,
    CanonicalRequest,
    ImagePart,
    ModelResponse,
    ResponseEnvelope,
    TextPart,
    TokenUsage,
)


# ---------------------------------------------------------------------------
# ImagePart validation
# ---------------------------------------------------------------------------

def test_image_part_requires_url_or_data():
    with pytest.raises(ValueError, match="requires either url or data"):
        ImagePart()


def test_image_part_rejects_both_url_and_data():
    with pytest.raises(ValueError, match="not both"):
        ImagePart(url="https://example.com/img.jpg", data="abc123")


def test_image_part_url_only():
    part = ImagePart(url="https://example.com/img.jpg")
    assert part.url == "https://example.com/img.jpg"
    assert part.data is None


def test_image_part_data_only():
    part = ImagePart(data="base64encodeddata", media_type="image/png")
    assert part.data == "base64encodeddata"
    assert part.media_type == "image/png"


# ---------------------------------------------------------------------------
# CanonicalMessage — multimodal content
# ---------------------------------------------------------------------------

def test_message_with_string_content():
    msg = CanonicalMessage(role="user", content="Hello")
    assert msg.content == "Hello"


def test_message_with_multimodal_content():
    msg = CanonicalMessage(
        role="user",
        content=[
            TextPart(text="Describe this image"),
            ImagePart(url="https://example.com/photo.jpg"),
        ],
    )
    assert len(msg.content) == 2
    assert isinstance(msg.content[0], TextPart)
    assert isinstance(msg.content[1], ImagePart)


# ---------------------------------------------------------------------------
# ResponseEnvelope serialisation
# ---------------------------------------------------------------------------

def _make_envelope(**kwargs) -> ResponseEnvelope:
    defaults = dict(
        response=ModelResponse(content="Hello!", finish_reason="stop"),
        request_sent={"model": "claude-sonnet-4-6", "messages": []},
        param_resolution_log=["Applied default max_tokens=8096 for 'claude-sonnet-4-6'."],
        provider="anthropic",
        model_id="claude-sonnet-4-6",
        latency_ms=123.4,
        timestamp="2026-03-25T12:00:00+00:00",
        usage=TokenUsage(input_tokens=10, output_tokens=5, total_tokens=15),
    )
    defaults.update(kwargs)
    return ResponseEnvelope(**defaults)


def test_to_dict_returns_plain_dict():
    envelope = _make_envelope()
    d = envelope.to_dict()
    assert isinstance(d, dict)
    assert d["response"]["content"] == "Hello!"
    assert d["response"]["finish_reason"] == "stop"
    assert d["provider"] == "anthropic"
    assert d["model_id"] == "claude-sonnet-4-6"


def test_to_dict_includes_usage():
    envelope = _make_envelope()
    d = envelope.to_dict()
    assert d["usage"]["input_tokens"] == 10
    assert d["usage"]["output_tokens"] == 5
    assert d["usage"]["total_tokens"] == 15


def test_to_dict_usage_is_none_when_not_set():
    envelope = _make_envelope(usage=None)
    d = envelope.to_dict()
    assert d["usage"] is None


def test_to_dict_excludes_raw():
    # raw is opt-in and must not appear in the serialised dict to guarantee
    # JSON-serialisability without Hopper imports.
    envelope = _make_envelope(raw={"some": "provider-object"})
    d = envelope.to_dict()
    assert "raw" not in d


def test_to_json_produces_valid_json():
    envelope = _make_envelope()
    serialised = envelope.to_json()
    parsed = json.loads(serialised)
    assert parsed["provider"] == "anthropic"


def test_to_json_requires_no_hopper_imports():
    # The JSON output must be loadable with the standard library alone —
    # no Hopper types in the output.
    envelope = _make_envelope()
    serialised = envelope.to_json()
    parsed = json.loads(serialised)
    # All values are plain Python types
    assert isinstance(parsed["response"], dict)
    assert isinstance(parsed["usage"], dict)
    assert isinstance(parsed["param_resolution_log"], list)


def test_param_resolution_log_is_included():
    envelope = _make_envelope()
    d = envelope.to_dict()
    assert len(d["param_resolution_log"]) == 1
    assert "max_tokens" in d["param_resolution_log"][0]
