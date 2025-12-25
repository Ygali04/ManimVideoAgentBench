from __future__ import annotations

import base64
import mimetypes
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import requests

from src.config.runtime import load_runtime_config


def _guess_mime_type(file_path: str) -> str:
    mime_type, _ = mimetypes.guess_type(file_path)
    if not mime_type:
        raise ValueError(f"Could not determine mime type for: {file_path}")
    return mime_type


def file_to_data_url(file_path: str, mime_type: Optional[str] = None) -> str:
    mime = mime_type or _guess_mime_type(file_path)
    with open(file_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def image_url_part(url_or_data_url: str, detail: str = "high") -> Dict[str, Any]:
    return {"type": "image_url", "image_url": {"url": url_or_data_url, "detail": detail}}


def video_url_part(url_or_data_url: str) -> Dict[str, Any]:
    return {"type": "video_url", "video_url": {"url": url_or_data_url}}


@dataclass(frozen=True)
class OpenRouterResponse:
    text: str
    raw: Dict[str, Any]


class OpenRouterClient:
    """
    Minimal direct HTTP client for OpenRouter's OpenAI-compatible endpoints.

    We use direct HTTP here (instead of strict OpenAI client schemas) to stay tolerant
    to multimodal `video_url` schema variations across client libraries.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        site_url: Optional[str] = None,
        app_name: Optional[str] = None,
        timeout_s: int = 180,
    ):
        runtime = load_runtime_config()
        self.api_key = api_key or runtime.openrouter.api_key
        self.api_base = (api_base or runtime.openrouter.api_base).rstrip("/")
        self.site_url = site_url or runtime.openrouter.site_url
        self.app_name = app_name or runtime.openrouter.app_name
        self.timeout_s = timeout_s

        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY is not set.")

    def _headers(self) -> Dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.site_url:
            headers["HTTP-Referer"] = self.site_url
        if self.app_name:
            headers["X-Title"] = self.app_name
        return headers

    def chat_completions(
        self,
        *,
        model: str,
        messages: List[Dict[str, Any]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> OpenRouterResponse:
        url = f"{self.api_base}/chat/completions"
        payload: Dict[str, Any] = {"model": model, "messages": messages}
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if extra_body:
            payload.update(extra_body)

        resp = requests.post(url, headers=self._headers(), json=payload, timeout=self.timeout_s)
        if resp.status_code >= 400:
            raise RuntimeError(f"OpenRouter error {resp.status_code}: {resp.text}")

        data = resp.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content")
        if content is None:
            raise RuntimeError(f"OpenRouter returned no content. Raw: {data}")

        # OpenAI-compatible responses usually return a string content.
        if isinstance(content, str):
            text = content
        else:
            # Defensive: join text parts if content is structured.
            text_parts: List[str] = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text" and "text" in part:
                    text_parts.append(str(part["text"]))
            text = "\n".join(text_parts) if text_parts else str(content)

        return OpenRouterResponse(text=text, raw=data)

    def embeddings(
        self,
        *,
        model: str,
        input: Union[str, List[str]],
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        url = f"{self.api_base}/embeddings"
        payload: Dict[str, Any] = {"model": model, "input": input}
        if extra_body:
            payload.update(extra_body)
        resp = requests.post(url, headers=self._headers(), json=payload, timeout=self.timeout_s)
        if resp.status_code >= 400:
            raise RuntimeError(f"OpenRouter embeddings error {resp.status_code}: {resp.text}")
        return resp.json()


def build_user_message(parts: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {"role": "user", "content": parts}


def local_video_path_to_part(video_path: str) -> Dict[str, Any]:
    if not os.path.exists(video_path):
        raise FileNotFoundError(video_path)
    return video_url_part(file_to_data_url(video_path))


