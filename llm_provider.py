"""Configurable LLM provider adapter for AI Chef.

Local development keeps using Ollama by default:

    AI_PROVIDER=ollama
    OLLAMA_BASE_URL=http://localhost:11434
    OLLAMA_MODEL=phi3:3.8b

The public deployment uses a hosted, OpenAI-compatible provider (Groq by
default) so it does not depend on the developer's laptop:

    AI_PROVIDER=groq
    GROQ_API_KEY=...
    GROQ_MODEL=openai/gpt-oss-20b

Any other OpenAI-compatible endpoint can be used with AI_PROVIDER=hosted and
HOSTED_LLM_BASE_URL / HOSTED_LLM_MODEL / HOSTED_LLM_API_KEY.

Rules enforced here:
  * no silent provider fallback,
  * secrets are read only from the server environment,
  * bounded retries and timeouts,
  * clear, non-secret error messages.
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import aiohttp


class ProviderError(RuntimeError):
    """The configured provider could not answer."""


class ProviderConfigError(ProviderError):
    """The provider is not configured correctly."""


@dataclass(frozen=True)
class ProviderInfo:
    provider: str
    base_url: str
    model: str
    api_key: str = ""
    timeout: float = 60.0
    retries: int = 2

    @property
    def is_ollama(self) -> bool:
        return self.provider == "ollama"

    @property
    def configured(self) -> bool:
        if not self.base_url or not self.model:
            return False
        return self.is_ollama or bool(self.api_key)

    @property
    def display_name(self) -> str:
        return "Ollama" if self.is_ollama else "hosted AI"

    def public(self) -> dict:
        return {
            "provider": self.provider,
            "model": self.model,
            "configured": self.configured,
        }

    def fingerprint(self) -> Tuple:
        return (self.provider, self.base_url, self.model, self.api_key,
                self.timeout, self.retries)


_CHECK_CACHE: Dict[Tuple, Tuple[float, bool, str]] = {}
# The free Groq plan has a daily request cap. Health/status probes must not
# consume it; one verification every five minutes is plenty.
_CHECK_TTL = 300.0


def _env(name: str, default: str = "") -> str:
    return (os.environ.get(name) or default).strip()


def _env_float(name: str, default: float) -> float:
    try:
        return float(_env(name, str(default)))
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)))
    except ValueError:
        return default


def load_provider() -> ProviderInfo:
    """Read the active provider from the environment.

    Raises ProviderConfigError for an unknown provider or a hosted provider
    with missing settings. It never falls back to another provider.
    """
    provider = _env("AI_PROVIDER", "ollama").lower()
    if provider in ("local", "ollama", ""):
        base = _env("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
        model = _env("OLLAMA_MODEL", "phi3:3.8b")
        timeout = _env_float("LLM_TIMEOUT", 120.0)
        retries = max(0, min(_env_int("LLM_MAX_RETRIES", 2), 4))
        if not base or not model:
            raise ProviderConfigError(
                "Ollama is not configured. Set OLLAMA_BASE_URL and OLLAMA_MODEL."
            )
        return ProviderInfo("ollama", base, model, "", timeout, retries)

    if provider in ("groq",):
        base = _env("GROQ_BASE_URL", "https://api.groq.com/openai/v1").rstrip("/")
        model = _env("GROQ_MODEL", "openai/gpt-oss-20b")
        key = _env("GROQ_API_KEY") or _env("HOSTED_LLM_API_KEY")
    elif provider in ("hosted", "openai", "openai_compatible", "openai-compatible"):
        base = _env("HOSTED_LLM_BASE_URL").rstrip("/")
        model = _env("HOSTED_LLM_MODEL")
        key = _env("HOSTED_LLM_API_KEY") or _env("LLM_API_KEY")
    else:
        raise ProviderConfigError(
            f"Unknown AI_PROVIDER {provider!r}. Use 'ollama', 'groq', or 'hosted'."
        )

    if not base:
        raise ProviderConfigError(
            "The hosted AI endpoint is missing. Set HOSTED_LLM_BASE_URL."
        )
    if not model:
        raise ProviderConfigError(
            "The hosted AI model is missing. Set HOSTED_LLM_MODEL."
        )
    if not key:
        raise ProviderConfigError(
            "The hosted AI API key is missing. Set HOSTED_LLM_API_KEY "
            "(or GROQ_API_KEY for AI_PROVIDER=groq)."
        )
    timeout = _env_float("LLM_TIMEOUT", 60.0)
    retries = max(0, min(_env_int("LLM_MAX_RETRIES", 2), 4))
    return ProviderInfo(provider, base, model, key, timeout, retries)


def _public_error(exc: Exception) -> str:
    """Remove secrets and cap error text before it reaches logs/UI."""
    text = str(exc) or exc.__class__.__name__
    try:
        info = load_provider()
        if info.api_key:
            text = text.replace(info.api_key, "[redacted]")
    except Exception:
        pass
    return text[:300]


def _headers(info: ProviderInfo) -> dict:
    headers = {"Content-Type": "application/json"}
    if not info.is_ollama and info.api_key:
        headers["Authorization"] = f"Bearer {info.api_key}"
    return headers


def _payload(info: ProviderInfo, messages: List[dict], json_mode: bool,
             max_tokens: int, temperature: float) -> dict:
    if info.is_ollama:
        payload = {
            "model": info.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        if json_mode:
            payload["format"] = "json"
        return payload
    payload = {
        "model": info.model,
        "messages": messages,
        "stream": False,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    return payload


def _chat_url(info: ProviderInfo) -> str:
    return f"{info.base_url}/api/chat" if info.is_ollama else f"{info.base_url}/chat/completions"


def _models_url(info: ProviderInfo) -> str:
    return f"{info.base_url}/api/tags" if info.is_ollama else f"{info.base_url}/models"


async def _post_once(info: ProviderInfo, messages: List[dict], json_mode: bool,
                     max_tokens: int, temperature: float) -> str:
    async with aiohttp.ClientSession() as client:
        async with client.post(
            _chat_url(info),
            json=_payload(info, messages, json_mode, max_tokens, temperature),
            headers=_headers(info),
            timeout=aiohttp.ClientTimeout(total=info.timeout),
        ) as resp:
            body = await resp.text()
            if resp.status != 200:
                raise _HttpStatus(resp.status, body, resp.headers.get("Retry-After"))
            try:
                data = await resp.json(content_type=None)
            except Exception:
                raise ProviderError("The AI provider returned invalid JSON.")

    if info.is_ollama:
        content = (data.get("message") or {}).get("content", "")
    else:
        choices = data.get("choices") or []
        content = ((choices[0] or {}).get("message") or {}).get("content", "") if choices else ""
    content = (content or "").strip()
    if not content:
        raise ProviderError("The AI provider returned an empty answer.")
    return content


class _HttpStatus(ProviderError):
    def __init__(self, status: int, body: str, retry_after: Optional[str]):
        self.status = status
        self.body = (body or "")[:300]
        self.retry_after = retry_after
        super().__init__(f"AI provider HTTP {status}")


async def llm_chat(messages: List[dict], json_mode: bool = False,
                   max_tokens: int = 300, temperature: float = 0.4) -> str:
    """Call the configured provider with bounded retries."""
    info = load_provider()
    if not info.configured:
        raise ProviderConfigError("The AI provider is not configured.")

    last_error: Exception = ProviderError("The AI provider did not answer.")
    attempts = info.retries + 1
    for attempt in range(attempts):
        try:
            return await _post_once(info, messages, json_mode, max_tokens, temperature)
        except _HttpStatus as exc:
            last_error = exc
            retryable = exc.status == 429 or 500 <= exc.status < 600
            if not retryable or attempt >= attempts - 1:
                if exc.status in (401, 403):
                    raise ProviderConfigError(
                        "The hosted AI rejected the API key. Check the server environment."
                    ) from None
                if exc.status == 429:
                    raise ProviderError(
                        "The free AI quota is busy right now. Please try again shortly."
                    ) from None
                raise ProviderError(
                    f"The AI provider returned HTTP {exc.status}."
                ) from None
            delay = 1.0
            if exc.retry_after:
                try:
                    delay = max(0.5, min(float(exc.retry_after), 8.0))
                except ValueError:
                    pass
            await asyncio.sleep(delay * (2 ** attempt))
        except asyncio.TimeoutError:
            last_error = ProviderError("The AI provider timed out.")
            if attempt >= attempts - 1:
                break
            await asyncio.sleep(min(4.0, 0.6 * (2 ** attempt)))
        except aiohttp.ClientError as exc:
            last_error = exc
            if attempt >= attempts - 1:
                break
            await asyncio.sleep(min(4.0, 0.6 * (2 ** attempt)))
        except ProviderError:
            raise
        except Exception as exc:
            last_error = exc
            break

    raise ProviderError(
        f"The AI provider is unavailable right now ({_public_error(last_error)})."
    )


async def check_provider(force: bool = False) -> Tuple[bool, str]:
    """Verify the configured provider and model without generating tokens."""
    try:
        info = load_provider()
    except ProviderConfigError as exc:
        return False, str(exc)
    if not info.configured:
        return False, "The AI provider is not configured."

    key = info.fingerprint()
    now = time.time()
    cached = _CHECK_CACHE.get(key)
    if not force and cached and now - cached[0] < _CHECK_TTL:
        return cached[1], cached[2]

    result: Tuple[bool, str]
    try:
        async with aiohttp.ClientSession() as client:
            async with client.get(
                _models_url(info),
                headers=_headers(info),
                timeout=aiohttp.ClientTimeout(total=min(15.0, info.timeout)),
            ) as resp:
                if resp.status in (401, 403):
                    result = (False, "The hosted AI rejected the API key.")
                elif resp.status != 200:
                    result = (False, f"AI provider health check returned HTTP {resp.status}.")
                else:
                    data = await resp.json(content_type=None)
                    if info.is_ollama:
                        names = [m.get("name", "") for m in data.get("models", [])]
                        found = any(info.model in n for n in names)
                    else:
                        names = [m.get("id", "") for m in data.get("data", [])]
                        found = info.model in names
                    result = (found, "ready" if found else "Configured AI model is not available.")
    except asyncio.TimeoutError:
        result = (False, "AI provider health check timed out.")
    except aiohttp.ClientError:
        result = (False, "AI provider is unreachable.")
    except Exception:
        result = (False, "AI provider health check failed.")

    _CHECK_CACHE[key] = (now, result[0], result[1])
    return result


def reset_provider_cache() -> None:
    """Test helper: forget cached health results."""
    _CHECK_CACHE.clear()


def provider_status() -> dict:
    """Public, secret-free provider metadata for /health and the UI."""
    try:
        info = load_provider()
        return {**info.public(), "display": info.display_name}
    except ProviderConfigError as exc:
        return {
            "provider": _env("AI_PROVIDER", "ollama"),
            "model": "",
            "configured": False,
            "display": "AI",
            "error": str(exc),
        }
