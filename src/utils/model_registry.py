from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple


@dataclass(frozen=True)
class ModelCapabilities:
    text: bool = True
    vision: bool = False
    video: bool = False


@dataclass(frozen=True)
class ReasoningConfig:
    # mode: none | openai_reasoning_effort | budget_tokens | model_variant_only
    mode: str = "none"
    effort: Optional[str] = None  # low|medium|high
    budget_tokens: Optional[int] = None


@dataclass(frozen=True)
class RecommendedParams:
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    reasoning: Optional[ReasoningConfig] = None


@dataclass(frozen=True)
class ModelSpec:
    alias: str
    openrouter_id: str
    capabilities: ModelCapabilities
    recommended_params: RecommendedParams


def default_registry_path() -> str:
    return os.path.join(os.path.dirname(__file__), "model_registry.json")


def _parse_reasoning(obj: Optional[Dict[str, Any]]) -> Optional[ReasoningConfig]:
    if not obj:
        return None
    return ReasoningConfig(
        mode=str(obj.get("mode", "none")),
        effort=obj.get("effort"),
        budget_tokens=obj.get("budget_tokens"),
    )


def _parse_model_spec(entry: Dict[str, Any]) -> ModelSpec:
    alias = entry["alias"]
    openrouter_id = entry["openrouter_id"]
    caps = entry.get("capabilities", {}) or {}
    rec = entry.get("recommended_params", {}) or {}
    return ModelSpec(
        alias=alias,
        openrouter_id=openrouter_id,
        capabilities=ModelCapabilities(
            text=bool(caps.get("text", True)),
            vision=bool(caps.get("vision", False)),
            video=bool(caps.get("video", False)),
        ),
        recommended_params=RecommendedParams(
            temperature=rec.get("temperature"),
            max_tokens=rec.get("max_tokens"),
            reasoning=_parse_reasoning(rec.get("reasoning")),
        ),
    )


def load_model_registry(registry_path: Optional[str] = None) -> Dict[str, ModelSpec]:
    path = registry_path or default_registry_path()
    with open(path, "r") as f:
        data = json.load(f)

    models = data.get("models")
    if not isinstance(models, list):
        raise ValueError(f"Invalid model registry format at {path}: expected key 'models' as a list")

    by_alias: Dict[str, ModelSpec] = {}
    for entry in models:
        spec = _parse_model_spec(entry)
        if spec.alias in by_alias:
            raise ValueError(f"Duplicate model alias in registry: {spec.alias}")
        by_alias[spec.alias] = spec
    return by_alias


def resolve_model(
    model_or_alias: str,
    registry_path: Optional[str] = None,
    allow_raw: bool = True,
) -> Tuple[str, Optional[ModelSpec], bool]:
    """
    Resolve a CLI model argument into an OpenRouter model id.

    Returns: (openrouter_id, spec_or_none, found_in_registry)
    """
    registry = load_model_registry(registry_path)
    if model_or_alias in registry:
        spec = registry[model_or_alias]
        return spec.openrouter_id, spec, True

    if not allow_raw:
        raise ValueError(f"Unknown model alias: {model_or_alias!r}. Add it to the model registry.")

    # Treat as a raw OpenRouter model id.
    return model_or_alias, None, False


