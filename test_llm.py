"""Unit tests for the real-LLM adapter's error classification and retry loop.

No API key or network: the retry loop is exercised with a fake provider that
raises a scripted sequence of exceptions. Only the classification and backoff
logic is under test — the provider factory (`_get_provider`) is bypassed by
pre-seeding `_provider`.
"""
from __future__ import annotations
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import pytest  # noqa: E402
from systems.llm import (  # noqa: E402
    KubeAstraLLM, _is_rate_limit, _is_transient, _is_retryable,
)

# The two faults seen in the wild that were wrongly scored as refusals.
UNAVAILABLE = "503 UNAVAILABLE. {'error': {'code': 503, 'status': 'UNAVAILABLE'}}"
RESET = "[Errno 54] Connection reset by peer"
RATE = "429 RESOURCE_EXHAUSTED. retryDelay: '38s'"
# A per-minute throttle with no parseable delay → retried on plain backoff.
RATE_SHORT = "429 RESOURCE_EXHAUSTED: rate limit exceeded, slow down"
# A daily-quota 429 asks us to wait hours, not seconds.
DAILY_QUOTA = ("429 RESOURCE_EXHAUSTED. Quota exceeded ... per_day, limit: 250. "
               "Please retry in 3h18m. retryDelay: '10818s'")


def test_rate_limit_classification():
    assert _is_rate_limit(RuntimeError(RATE))
    assert _is_rate_limit(RuntimeError("Rate limit exceeded"))
    assert not _is_rate_limit(RuntimeError(UNAVAILABLE))


def test_transient_classification():
    assert _is_transient(RuntimeError(UNAVAILABLE))
    assert _is_transient(RuntimeError(RESET))
    assert _is_transient(RuntimeError("500 Internal error"))


def test_client_errors_are_not_transient():
    # A bad request or bad key won't fix itself — must fail fast, not retry.
    assert not _is_transient(RuntimeError("400 INVALID_ARGUMENT"))
    assert not _is_transient(RuntimeError("403 permission denied"))
    assert not _is_retryable(RuntimeError("401 unauthorized: invalid api key"))


def test_refusal_text_is_not_an_exception_path():
    # Sanity: a baseline "refusal" is a normal string return, never an
    # exception, so it never reaches the retry classifier. A plausible model
    # sentence must not look retryable.
    assert not _is_retryable(RuntimeError("I cannot produce that diff."))


class _ScriptedProvider:
    """Raises the first `len(faults)` calls, then returns `ok`."""

    def __init__(self, faults: list[Exception], ok: str = "OK"):
        self._faults = list(faults)
        self._ok = ok
        self.calls = 0

    def generate(self, user, system=None, temperature=None, max_tokens=None):
        self.calls += 1
        if self._faults:
            raise self._faults.pop(0)
        return self._ok


def _llm(provider) -> KubeAstraLLM:
    # tiny delays so the backoff sleeps are negligible; pre-seed the provider
    # and mark setup done so the factory / settings import is never touched.
    llm = KubeAstraLLM(model="x", base_delay=0.001, max_delay=0.01, max_retries=6)
    llm._provider = provider
    llm._setup_done = True
    return llm


def test_transient_fault_is_retried_then_succeeds():
    p = _ScriptedProvider([RuntimeError(UNAVAILABLE), RuntimeError(RESET)])
    assert _llm(p).complete("sys", "user") == "OK"
    assert p.calls == 3            # two faults + one success


def test_rate_limit_is_retried_then_succeeds():
    p = _ScriptedProvider([RuntimeError(RATE_SHORT)])
    assert _llm(p).complete("sys", "user") == "OK"
    assert p.calls == 2


def test_client_error_fails_fast_without_retry():
    p = _ScriptedProvider([RuntimeError("400 INVALID_ARGUMENT: bad prompt")])
    with pytest.raises(RuntimeError):
        _llm(p).complete("sys", "user")
    assert p.calls == 1            # no retry burned on a client error


def test_daily_quota_429_fails_fast_without_thrash():
    # retryDelay is hours (>> max_delay): retrying 6x a 75s cap is pointless,
    # so give up on the first hit rather than limp for the whole run.
    p = _ScriptedProvider([RuntimeError(DAILY_QUOTA)])
    with pytest.raises(RuntimeError):
        _llm(p).complete("sys", "user")
    assert p.calls == 1


def test_gives_up_after_max_retries():
    faults = [RuntimeError(UNAVAILABLE) for _ in range(20)]
    p = _ScriptedProvider(faults)
    with pytest.raises(RuntimeError):
        _llm(p).complete("sys", "user")
    assert p.calls == 7            # max_retries(6) + 1 initial attempt


# ── Anthropic cached-request path ─────────────────────────────────────────────

from types import SimpleNamespace as _NS  # noqa: E402


class _FakeMessages:
    def __init__(self, blocks):
        self._blocks = blocks
        self.last_kwargs = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        return _NS(content=self._blocks)


class _FakeAnthropic:
    def __init__(self, blocks):
        self.messages = _FakeMessages(blocks)


def _anthropic_llm(blocks) -> KubeAstraLLM:
    llm = KubeAstraLLM(model="claude-sonnet-5", max_retries=0)
    client = _FakeAnthropic(blocks)
    llm._anthropic = (client, "claude-sonnet-5")   # pre-seed: skip real setup
    llm._setup_done = True
    return llm, client


def test_anthropic_path_marks_system_for_caching():
    llm, client = _anthropic_llm([_NS(type="text", text="EDITED FILE")])
    out = llm.complete("INSTRUCTIONS + FILE:\n<manifest>", "change replicas to 5")
    assert out == "EDITED FILE"
    kw = client.messages.last_kwargs
    assert kw["model"] == "claude-sonnet-5"
    # the manifest-bearing system block is sent with an ephemeral cache breakpoint
    assert kw["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert kw["system"][0]["text"].endswith("<manifest>")
    # the volatile instruction stays in the (uncached) user message
    assert kw["messages"] == [{"role": "user", "content": "change replicas to 5"}]


def test_anthropic_path_skips_thinking_blocks_and_flags_empty():
    # a response with only a thinking block (no text) is an empty result
    llm, _ = _anthropic_llm([_NS(type="thinking", thinking="...")])
    with pytest.raises(RuntimeError):
        llm.complete("sys", "user")
