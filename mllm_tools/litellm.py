import json
import re
import time
from typing import List, Dict, Any, Union, Optional
import io
import os
import base64
from PIL import Image
import mimetypes
import litellm
from litellm import completion, completion_cost
from litellm.exceptions import APIConnectionError, APIError, RateLimitError, ServiceUnavailableError
from dotenv import load_dotenv

load_dotenv()

from src.config.runtime import load_runtime_config
from src.utils.model_registry import default_registry_path, load_model_registry, ModelSpec


_MODEL_REGISTRY_CACHE: Optional[Dict[str, ModelSpec]] = None


def _get_model_registry() -> Dict[str, ModelSpec]:
    global _MODEL_REGISTRY_CACHE
    if _MODEL_REGISTRY_CACHE is None:
        try:
            _MODEL_REGISTRY_CACHE = load_model_registry(default_registry_path())
        except Exception:
            # Registry is an optional convenience; don’t hard-fail model calls if it’s missing/broken.
            _MODEL_REGISTRY_CACHE = {}
    return _MODEL_REGISTRY_CACHE


def _lookup_model_spec(model_name: str) -> Optional[ModelSpec]:
    registry = _get_model_registry()
    if not registry:
        return None

    # Direct alias match.
    if model_name in registry:
        return registry[model_name]

    # Match against OpenRouter ids (with or without openrouter/ prefix).
    candidate = model_name[len("openrouter/"):] if model_name.startswith("openrouter/") else model_name
    for spec in registry.values():
        if spec.openrouter_id == candidate:
            return spec
    return None


class LiteLLMWrapper:
    """Wrapper for LiteLLM to support multiple models and logging"""
    
    def __init__(
        self,
        model_name: str = "gpt-4-vision-preview",
        temperature: float = 0.7,
        print_cost: bool = False,
        verbose: bool = False,
        use_langfuse: bool = True,
        provider: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ):
        """
        Initialize the LiteLLM wrapper
        
        Args:
            model_name: Name of the model to use (e.g. "azure/gpt-4", "vertex_ai/gemini-pro")
            temperature: Temperature for completion
            print_cost: Whether to print the cost of the completion
            verbose: Whether to print verbose output
            use_langfuse: Whether to enable Langfuse logging
        """
        self.model_name = model_name
        self.temperature = temperature
        self.print_cost = print_cost
        self.verbose = verbose
        self.accumulated_cost = 0
        self.provider = provider
        self.max_tokens = max_tokens

        if self.verbose:
            os.environ['LITELLM_LOG'] = 'DEBUG'
        
        # Set langfuse callback only if enabled
        if use_langfuse:
            litellm.success_callback = ["langfuse"]
            litellm.failure_callback = ["langfuse"]

    def _encode_file(self, file_path: Union[str, Image.Image]) -> str:
        """
        Encode local file or PIL Image to base64 string
        
        Args:
            file_path: Path to local file or PIL Image object
            
        Returns:
            Base64 encoded file string
        """
        if isinstance(file_path, Image.Image):
            buffered = io.BytesIO()
            file_path.save(buffered, format="PNG")
            return base64.b64encode(buffered.getvalue()).decode("utf-8")
        else:
            with open(file_path, "rb") as file:
                return base64.b64encode(file.read()).decode("utf-8")

    def _get_mime_type(self, file_path: str) -> str:
        """
        Get the MIME type of a file based on its extension
        
        Args:
            file_path: Path to the file
            
        Returns:
            MIME type as a string (e.g., "image/jpeg", "audio/mp3")
        """
        mime_type, _ = mimetypes.guess_type(file_path)
        if mime_type is None:
            raise ValueError(f"Unsupported file type: {file_path}")
        return mime_type

    def __call__(self, messages: List[Dict[str, Any]], metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Process messages and return completion
        
        Args:
            messages: List of message dictionaries with 'type' and 'content' keys
            metadata: Optional metadata to pass to litellm completion, e.g. for Langfuse tracking
        
        Returns:
            Generated text response
        """
        if metadata is None:
            print("No metadata provided, using empty metadata")
            metadata = {}
        metadata["trace_name"] = f"litellm-completion-{self.model_name}"
        runtime = load_runtime_config()
        provider = (self.provider or runtime.provider or "openrouter").strip().lower()

        model_spec = _lookup_model_spec(self.model_name)

        # Convert messages to OpenAI-style content parts (works well with OpenRouter).
        content_parts: List[Dict[str, Any]] = []
        for msg in messages:
            msg_type = msg["type"]
            if msg_type == "text":
                content_parts.append({"type": "text", "text": msg["content"]})
                continue

            if msg_type not in ["image", "audio", "video"]:
                raise ValueError(f"Unsupported message type: {msg_type!r}")

            if msg_type == "image" and model_spec and not model_spec.capabilities.vision:
                raise ValueError(
                    f"Model {model_spec.alias} ({model_spec.openrouter_id}) does not support vision, "
                    f"but an image input was provided."
                )
            if msg_type == "video" and model_spec and not model_spec.capabilities.video:
                raise ValueError(
                    f"Model {model_spec.alias} ({model_spec.openrouter_id}) does not support video, "
                    f"but a video input was provided."
                )

            # Convert local files / PIL to data: URLs.
            if isinstance(msg["content"], Image.Image) or (isinstance(msg["content"], str) and os.path.isfile(msg["content"])):
                if isinstance(msg["content"], Image.Image):
                    mime_type = "image/png"
                else:
                    mime_type = self._get_mime_type(msg["content"])
                base64_data = self._encode_file(msg["content"])
                data_url = f"data:{mime_type};base64,{base64_data}"
            else:
                data_url = msg["content"]

            if msg_type == "image":
                content_parts.append(
                    {"type": "image_url", "image_url": {"url": data_url, "detail": "high"}}
                )
            elif msg_type == "video":
                # Note: most of this repo routes video eval through a dedicated OpenRouter client.
                content_parts.append({"type": "video_url", "video_url": {"url": data_url}})
            else:
                raise ValueError("Audio inputs are not currently supported by LiteLLMWrapper in this repo.")

        formatted_messages = [{"role": "user", "content": content_parts}]

        try:
            completion_kwargs: Dict[str, Any] = {
                "messages": formatted_messages,
                "temperature": self.temperature,
                "metadata": metadata,
                "max_retries": 99,
            }

            model_for_completion = self.model_name
            if provider == "openrouter" or self.model_name.startswith("openrouter/"):
                if not runtime.openrouter.api_key:
                    raise ValueError("OPENROUTER_API_KEY is not set, but TEA_PROVIDER=openrouter.")
                model_for_completion = self.model_name if self.model_name.startswith("openrouter/") else f"openrouter/{self.model_name}"
                completion_kwargs["api_base"] = runtime.openrouter.api_base
                completion_kwargs["api_key"] = runtime.openrouter.api_key
                if runtime.openrouter.extra_headers:
                    completion_kwargs["extra_headers"] = runtime.openrouter.extra_headers

            # Apply registry-driven defaults if the wrapper didn't explicitly set them.
            if model_spec and model_spec.recommended_params.temperature is not None and completion_kwargs.get("temperature") is None:
                completion_kwargs["temperature"] = model_spec.recommended_params.temperature

            max_tokens_to_use = self.max_tokens
            if max_tokens_to_use is None and model_spec and model_spec.recommended_params.max_tokens is not None:
                max_tokens_to_use = model_spec.recommended_params.max_tokens
            if max_tokens_to_use is not None:
                completion_kwargs["max_tokens"] = max_tokens_to_use

            # If it's OpenAI o-series, set temperature None + reasoning_effort default.
            if (re.match(r"^o\\d+.*$", self.model_name) or re.match(r"^openai/o.*$", self.model_name)):
                completion_kwargs["temperature"] = None
                completion_kwargs["reasoning_effort"] = "medium"

            # Registry-driven reasoning controls (best-effort; omit if unsupported).
            if model_spec and model_spec.recommended_params.reasoning:
                reasoning = model_spec.recommended_params.reasoning
                if reasoning.mode == "openai_reasoning_effort" and reasoning.effort:
                    completion_kwargs.setdefault("reasoning_effort", reasoning.effort)
                elif reasoning.mode != "none":
                    print(
                        f"Warning: reasoning mode {reasoning.mode!r} is not mapped in LiteLLMWrapper yet; omitting."
                    )

            # Retry configuration for transient API errors
            max_retries = 5
            base_delay = 2.0  # seconds
            transient_errors = (APIConnectionError, APIError, RateLimitError, ServiceUnavailableError)
            
            def _attempt_completion(kwargs: Dict[str, Any]) -> Any:
                """Attempt completion with retry logic for transient errors."""
                last_exception = None
                for attempt in range(max_retries):
                    try:
                        return completion(model=model_for_completion, **kwargs)
                    except transient_errors as e:
                        last_exception = e
                        if attempt < max_retries - 1:
                            delay = base_delay * (2 ** attempt)  # Exponential backoff
                            print(f"Warning: API error on attempt {attempt + 1}/{max_retries}: {type(e).__name__}. "
                                  f"Retrying in {delay:.1f}s...")
                            time.sleep(delay)
                        else:
                            print(f"Error: All {max_retries} attempts failed with transient errors.")
                            raise
                    except Exception as e:
                        # Non-transient error, don't retry
                        raise
                raise last_exception  # Should not reach here, but just in case
            
            try:
                response = _attempt_completion(completion_kwargs)
            except Exception as e:
                # If optional params break a provider/model, retry once without them.
                retryable_keys = ["reasoning_effort", "max_tokens"]
                if any(k in completion_kwargs for k in retryable_keys) and not isinstance(e, transient_errors):
                    stripped = dict(completion_kwargs)
                    for k in retryable_keys:
                        if k in stripped:
                            print(f"Warning: completion failed; retrying without {k} (model={model_for_completion}).")
                            stripped.pop(k, None)
                    response = _attempt_completion(stripped)
                else:
                    raise e

            if self.print_cost:
                # pass your response from completion to completion_cost
                try:
                    cost = completion_cost(completion_response=response)
                    self.accumulated_cost += cost
                    print(f"Accumulated Cost: ${self.accumulated_cost:.10f}")
                except Exception as cost_err:
                    # Cost calculation may fail for unmapped models - not critical
                    print(f"Warning: Could not calculate cost: {cost_err}")
                
            content = response.choices[0].message.content
            if content is None:
                print(f"Got null response from model. Full response: {response}")
            return content
        
        except Exception as e:
            print(f"Error in model completion: {e}")
            raise
        
if __name__ == "__main__":
    pass