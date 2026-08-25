"""LLM interface for the baselines.

The baselines depend only on `complete(system, user) -> str`, so the harness is
fully testable with `FakeLLM` (deterministic, scriptable) and needs no API key.
`KubeAstraLLM` is the real adapter — it reuses KubeAstra's provider factory, so
whichever provider is configured (Claude / Gemini / OpenAI / Ollama) works via
one env switch. Model choice is a run-time decision, not baked in here.
"""
from __future__ import annotations

import re
import sys
import time
from typing import Callable, Protocol, Sequence, Union

from .base import find_mcp


def _is_rate_limit(exc: Exception) -> bool:
    s = str(exc).lower()
    return "429" in s or "resource_exhausted" in s or "rate limit" in s or "quota" in s


def _is_transient(exc: Exception) -> bool:
    """A transient *service/network* fault (the endpoint is momentarily down or
    the socket dropped), as opposed to the model refusing or a client error.

    These must be retried, not scored: a 503 or a reset connection is the
    service being unavailable, and counting it as a baseline "refusal" both
    misattributes the failure and understates the baseline. Deterministic
    client errors (400/401/403 — a bad request or bad key) are excluded so a
    genuinely broken call fails fast instead of burning the retry budget.
    """
    s = str(exc).lower()
    if any(c in s for c in ("400", "401", "403", "invalid", "permission")):
        return False
    return any(t in s for t in (
        "503", "500", "unavailable", "internal error", "internal server",
        "connection reset", "connection aborted", "connection error",
        "timed out", "timeout", "temporarily", "[errno",
    ))


def _is_retryable(exc: Exception) -> bool:
    return _is_rate_limit(exc) or _is_transient(exc)


def _retry_after_seconds(exc: Exception) -> float | None:
    """Pull a retry delay out of a Google rate-limit error string, if present
    (e.g. 'retry in 38.7s' or "retryDelay': '38s'")."""
    for pat in (r"retry in ([\d.]+)s", r"retryDelay['\"]?:\s*['\"]?(\d+)s"):
        m = re.search(pat, str(exc))
        if m:
            return float(m.group(1))
    return None


class LLM(Protocol):
    def complete(self, system: str, user: str) -> str: ...


class FakeLLM:
    """Deterministic test double. `responses` is either a callable
    (system, user) -> str, or a sequence returned in order (last repeats), which
    is how B3's retry loop is exercised."""

    def __init__(self, responses: Union[Callable[[str, str], str], Sequence[str]]):
        self.responses = responses
        self.calls: list[tuple[str, str]] = []
        self._i = 0

    def complete(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        if callable(self.responses):
            return self.responses(system, user)
        r = self.responses[min(self._i, len(self.responses) - 1)]
        self._i += 1
        return r


class KubeAstraLLM:
    """Real adapter over KubeAstra's provider factory (`services.llm`).

    Honors an explicit `model` for every provider — including Gemini, whose
    factory otherwise hardcodes flash-lite — and retries on rate-limit (429)
    and transient service faults (503 / dropped connection) with backoff so
    throttling or a brief outage slows a run instead of corrupting it. A
    rate-limit honors the server's `retryDelay`; a transient fault just backs
    off exponentially.

    On the Anthropic provider it calls the client directly so the `system`
    block (which carries the manifest, see prompts.py) is sent with
    `cache_control` — prompt caching makes the reused manifest ~90% cheaper.
    Other providers keep the factory path unchanged (no caching).
    """

    def __init__(self, model: str | None = None, temperature: float = 0.2,
                 max_tokens: int = 4000, max_retries: int = 6,
                 base_delay: float = 10.0, max_delay: float = 75.0):
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._max_retries = max_retries
        self._base_delay = base_delay
        self._max_delay = max_delay
        self._provider = None
        self._anthropic = None     # (client, model) — the cached-request path
        self._setup_done = False

    def _get_provider(self):
        """Set up the backend once and return the provider (None on the
        Anthropic path, which uses the direct client instead)."""
        if self._setup_done:
            return self._provider
        mcp = find_mcp()
        if mcp and str(mcp) not in sys.path:
            sys.path.insert(0, str(mcp))
        from config.settings import get_settings
        settings = get_settings()
        pname = (settings.llm_provider or "gemini").lower()
        if pname == "anthropic":
            # Use the Anthropic client directly so we can attach cache_control —
            # KubeAstra's provider abstraction can't. The manifest lives in a
            # cached `system` block (see prompts.py), so the ~80 tasks that share
            # a manifest read it from cache instead of re-billing ~5.7K tokens.
            import anthropic
            model = self._model or settings.anthropic_model
            self._anthropic = (anthropic.Anthropic(api_key=settings.anthropic_api_key), model)
        elif pname == "gemini" and self._model:
            # Bypass the factory's hardcoded flash-lite so --model is honored.
            from services.llm.gemini_provider import GeminiProvider
            self._provider = GeminiProvider(
                api_key=settings.gemini_api_key, model=self._model,
                timeout=settings.gemini_timeout_seconds)
        else:
            from services.llm import get_provider
            self._provider = get_provider(self._model)
        self._setup_done = True
        return self._provider

    def _single_call(self, system: str, user: str) -> str:
        """One request. Anthropic goes through the direct client with a cached
        system block; every other provider through the factory as before."""
        if self._anthropic is not None:
            client, model = self._anthropic
            resp = client.messages.create(
                model=model,
                max_tokens=self._max_tokens,
                system=[{"type": "text", "text": system,
                         "cache_control": {"type": "ephemeral"}}],
                messages=[{"role": "user", "content": user}],
            )
            text = "".join(getattr(b, "text", "") for b in resp.content
                           if getattr(b, "type", None) == "text").strip()
            if not text:
                raise RuntimeError("Anthropic returned an empty response")
            return text
        provider = self._get_provider()
        return provider.generate(user, system=system,
                                 temperature=self._temperature,
                                 max_tokens=self._max_tokens)

    def complete(self, system: str, user: str) -> str:
        self._get_provider()                      # ensure backend is set up
        delay = self._base_delay
        last: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                return self._single_call(system, user)
            except Exception as exc:
                last = exc
                if not _is_retryable(exc) or attempt == self._max_retries:
                    raise
                # A rate-limit carries the server's own delay; a transient
                # service/network fault has none, so fall back to the backoff.
                server_wait = _retry_after_seconds(exc) if _is_rate_limit(exc) else None
                # A *daily*-quota 429 asks us to wait hours (retryDelay far
                # beyond max_delay). Sleeping max_delay and retrying is pointless
                # thrash — fail fast so the caller sees the quota wall at once
                # instead of a run that limps for days producing all-refusals.
                if server_wait is not None and server_wait > self._max_delay:
                    raise
                wait = server_wait or delay
                time.sleep(min(wait + 2.0, self._max_delay))   # small buffer
                delay = min(delay * 2, self._max_delay)
        raise last if last else RuntimeError("unreachable")
