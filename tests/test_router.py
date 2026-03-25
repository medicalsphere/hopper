"""
Router tests — these read as specifications for what the router guarantees.
"""
import pytest

from hopper import router
from hopper.types import CanonicalMessage, CanonicalRequest, ModelEntry


def _req(model: str, **kwargs) -> CanonicalRequest:
    return CanonicalRequest(
        model=model,
        messages=[CanonicalMessage(role="user", content="hello")],
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Model resolution
# ---------------------------------------------------------------------------

def test_resolves_exact_model_id():
    _, entry, _ = router.resolve(_req("claude-sonnet-4-6"))
    assert entry.id == "claude-sonnet-4-6"
    assert entry.provider == "anthropic"


def test_resolves_alias_to_canonical_api_id():
    # The alias "claude-sonnet" should resolve to the real API model string.
    _, entry, _ = router.resolve(_req("claude-sonnet"))
    assert entry.provider == "anthropic"
    assert entry.id == "claude-sonnet-4-6"  # exact API string, not the alias


def test_alias_and_id_are_the_same_model_entry_object():
    # Aliases must share the same ModelEntry instance, not copies.
    _, entry_by_alias, _ = router.resolve(_req("claude-sonnet"))
    _, entry_by_id, _ = router.resolve(_req("claude-sonnet-4-6"))
    assert entry_by_alias is entry_by_id


def test_raises_for_unknown_model():
    with pytest.raises(ValueError, match="Unknown model"):
        router.resolve(_req("not-a-real-model"))


def test_all_six_providers_have_at_least_one_model():
    providers_found = {entry.provider for entry in router._MODELS.values()}
    assert providers_found == {"anthropic", "openai", "google", "together", "perplexity", "grok"}


def test_registry_covers_expected_aliases():
    for alias in ["claude-sonnet", "claude-opus",
                  "gpt-5.4", "gpt-5.4-mini",
                  "gemini-3-flash", "gemini-3.1-pro", "GLM-5",
                  "grok", "sonar"]:
        assert alias in router._MODELS, f"Alias {alias!r} missing from registry"


# ---------------------------------------------------------------------------
# Parameter filtering
# ---------------------------------------------------------------------------

def test_applies_model_default_when_caller_omits_max_tokens():
    entry = router._MODELS["claude-sonnet-4-6"]
    _, log = router.apply_defaults_and_filter(_req("claude-sonnet-4-6"), entry)
    # Default max_tokens should be applied
    assert any("max_tokens" in msg for msg in log)


def test_caller_value_overrides_model_default():
    entry = router._MODELS["claude-sonnet-4-6"]
    filtered, log = router.apply_defaults_and_filter(_req("claude-sonnet-4-6", max_tokens=100), entry)
    assert filtered["max_tokens"] == 100
    # No "applied default" log entry for max_tokens since caller supplied it
    assert not any("Applied default max_tokens" in msg for msg in log)


def test_drops_unsupported_param_and_logs_warning():
    # gpt-5.4 has temperature in unsupported_params
    entry = router._MODELS["gpt-5.4-2026-03-05"]
    filtered, log = router.apply_defaults_and_filter(
        _req("gpt-5.4-2026-03-05", temperature=0.7), entry
    )
    assert "temperature" not in filtered
    assert any("Dropped unsupported param" in msg and "temperature" in msg for msg in log)


def test_passes_supported_param_without_logging():
    entry = router._MODELS["claude-sonnet-4-6"]
    filtered, log = router.apply_defaults_and_filter(
        _req("claude-sonnet-4-6", temperature=0.5, max_tokens=512), entry
    )
    assert filtered["temperature"] == 0.5
    assert filtered["max_tokens"] == 512
    # No drop warnings
    assert not any("Dropped" in msg for msg in log)


def test_unknown_param_passes_through_with_notice():
    # Create a synthetic entry where supported_params is non-empty but doesn't
    # include a param the caller supplies — simulates a future canonical param.
    entry = ModelEntry(
        id="test-model",
        provider="openai",
        supported_params=["max_tokens"],   # temperature not listed
        unsupported_params=[],
        defaults={},
    )
    request = _req("gpt-4o", temperature=0.5)
    filtered, log = router.apply_defaults_and_filter(request, entry)
    assert "temperature" in filtered   # passed through
    assert any("not listed" in msg for msg in log)


def test_provider_options_are_not_touched_by_filter():
    # provider_options bypass filtering; the adapter reads them directly.
    entry = router._MODELS["claude-sonnet-4-6"]
    request = CanonicalRequest(
        model="claude-sonnet-4-6",
        messages=[CanonicalMessage(role="user", content="hi")],
        provider_options={"response_format": {"type": "json_object"}},
    )
    filtered, _ = router.apply_defaults_and_filter(request, entry)
    # provider_options do not appear in the filtered dict
    assert "response_format" not in filtered


# ---------------------------------------------------------------------------
# Passthrough mode (unregistered model + provider=)
# ---------------------------------------------------------------------------

def test_passthrough_resolves_with_provider_hint():
    _, entry, log = router.resolve(_req("claude-new-model-x", provider="anthropic"))
    assert entry.id == "claude-new-model-x"
    assert entry.provider == "anthropic"
    assert any("Passthrough" in msg for msg in log)


def test_passthrough_raises_without_provider_hint():
    with pytest.raises(ValueError, match="Unknown model"):
        router.resolve(_req("claude-new-model-x"))


def test_extra_params_pass_through_filter():
    entry = router._MODELS["claude-sonnet-4-6"]
    request = _req("claude-sonnet-4-6", extra_params={"top_p": 0.9, "top_k": 40})
    filtered, _ = router.apply_defaults_and_filter(request, entry)
    assert filtered["top_p"] == 0.9
    assert filtered["top_k"] == 40


def test_extra_params_override_canonical_params():
    entry = router._MODELS["claude-sonnet-4-6"]
    # If caller passes max_tokens both ways, extra_params wins (merged last)
    request = _req("claude-sonnet-4-6", max_tokens=512, extra_params={"max_tokens": 999})
    filtered, _ = router.apply_defaults_and_filter(request, entry)
    assert filtered["max_tokens"] == 999


# ---------------------------------------------------------------------------
# Adapter loading
# ---------------------------------------------------------------------------

def test_resolve_returns_adapter_with_correct_methods():
    adapter, _, _ = router.resolve(_req("claude-sonnet-4-6"))
    assert callable(getattr(adapter, "complete", None))
    assert callable(getattr(adapter, "stream", None))
    assert callable(getattr(adapter, "is_retryable", None))
