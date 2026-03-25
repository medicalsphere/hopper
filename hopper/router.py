from __future__ import annotations

import importlib
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from hopper.types import CanonicalRequest, ModelEntry

if TYPE_CHECKING:
    from hopper.adapters.base import ProviderAdapter

# ---------------------------------------------------------------------------
# Registry — loaded once at import time from models/*.yaml
# ---------------------------------------------------------------------------

_MODELS: dict[str, ModelEntry] = {}  # keyed by both model id and every alias


def _load_registry() -> None:
    models_dir = Path(__file__).parent / "models"
    seen: set[str] = set()

    for yaml_file in sorted(models_dir.glob("*.yaml")):
        with yaml_file.open() as f:
            data = yaml.safe_load(f)

        provider = data["provider"]
        for entry_data in data["models"]:
            entry = ModelEntry(
                id=entry_data["id"],
                provider=provider,
                aliases=entry_data.get("aliases") or [],
                supported_params=entry_data.get("supported_params") or [],
                unsupported_params=entry_data.get("unsupported_params") or [],
                defaults=entry_data.get("defaults") or {},
            )

            for key in [entry.id] + entry.aliases:
                if key in seen:
                    raise ValueError(
                        f"Duplicate model ID or alias {key!r} found in {yaml_file.name}"
                    )
                seen.add(key)
                _MODELS[key] = entry  # all keys point to the same ModelEntry object


_load_registry()


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------

def resolve(request: CanonicalRequest) -> tuple[ProviderAdapter, ModelEntry, list[str]]:
    """
    Resolve a model ID (or alias) to the correct adapter and registry entry.

    Uses dynamic import so that adding a new provider requires only creating
    one adapter file and one YAML file — this function never changes.
    Each adapter module must export a module-level ADAPTER instance.
    """
    model_entry = _MODELS.get(request.model)
    if model_entry is None:
        available = sorted(_MODELS)
        raise ValueError(
            f"Unknown model: {request.model!r}. Available models: {available}"
        )

    module = importlib.import_module(f"hopper.adapters.{model_entry.provider}")
    adapter: ProviderAdapter = module.ADAPTER
    return adapter, model_entry, []


def apply_defaults_and_filter(
    request: CanonicalRequest,
    model_entry: ModelEntry,
) -> tuple[dict, list[str]]:
    """
    Build the filtered parameter dict for the adapter.

    Three outcomes for each canonical param (temperature, max_tokens):
    1. In unsupported_params  → drop it, log a warning.
    2. Not in supported_params (unknown) → pass through, log a notice.
    3. In supported_params    → pass through silently.

    Model defaults are applied for params the caller left unset.
    provider_options bypass this filter entirely; adapters read them directly.
    """
    log: list[str] = []
    raw: dict = {}

    if request.max_tokens is not None:
        raw["max_tokens"] = request.max_tokens
    if request.temperature is not None:
        raw["temperature"] = request.temperature

    # Apply model-level defaults for params the caller did not supply
    for param, default_value in model_entry.defaults.items():
        if param not in raw:
            raw[param] = default_value
            log.append(
                f"Applied default {param}={default_value!r} for {model_entry.id!r}."
            )

    filtered: dict = {}
    for param, value in raw.items():
        if param in model_entry.unsupported_params:
            log.append(
                f"Dropped unsupported param {param!r} for model {model_entry.id!r}."
            )
        elif model_entry.supported_params and param not in model_entry.supported_params:
            log.append(
                f"Param {param!r} not listed for {model_entry.id!r} — passing through."
            )
            filtered[param] = value
        else:
            filtered[param] = value

    return filtered, log
