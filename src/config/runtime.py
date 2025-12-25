from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Dict


def _get_env(name: str, default: Optional[str] = None) -> Optional[str]:
    value = os.getenv(name)
    if value is None:
        return default
    value = value.strip()
    return value if value != "" else default


@dataclass(frozen=True)
class OpenRouterConfig:
    api_key: Optional[str]
    api_base: str
    site_url: Optional[str]
    app_name: Optional[str]

    @property
    def extra_headers(self) -> Dict[str, str]:
        headers: Dict[str, str] = {}
        # OpenRouter recommends these headers when available.
        if self.site_url:
            headers["HTTP-Referer"] = self.site_url
        if self.app_name:
            headers["X-Title"] = self.app_name
        return headers


@dataclass(frozen=True)
class RuntimeConfig:
    provider: str
    openrouter: OpenRouterConfig


def load_runtime_config() -> RuntimeConfig:
    provider = _get_env("TEA_PROVIDER", "openrouter")
    openrouter = OpenRouterConfig(
        api_key=_get_env("OPENROUTER_API_KEY"),
        api_base=_get_env("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1"),
        site_url=_get_env("OPENROUTER_SITE_URL"),
        app_name=_get_env("OPENROUTER_APP_NAME", "TheoremExplainAgent"),
    )
    return RuntimeConfig(provider=provider, openrouter=openrouter)


