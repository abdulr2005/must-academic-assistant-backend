"""Rotate prepared text input across authorized provider/key/model candidates."""
import json
import logging
import time
from dataclasses import dataclass, field
from urllib.parse import quote

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq

from . import config
from .rate_limit_tracker import CooldownTracker, is_quota_exhaustion, parse_retry_after_seconds

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Candidate:
    provider: str
    key_alias: str
    model: str
    client: object = field(repr=False)
    secret: str = field(default="", repr=False)

    @property
    def id(self):
        return f"{self.provider}/{self.key_alias}/{self.model}"

    @property
    def key_id(self):
        return f"{self.provider}/{self.key_alias}"


@dataclass(frozen=True)
class Failure:
    kind: str
    scope: str = "model"


def _error_chain(exc):
    """LangChain may wrap SDK status/metadata in a cause; bound traversal."""
    seen = set()
    while exc is not None and id(exc) not in seen and len(seen) < 8:
        seen.add(id(exc))
        yield exc
        exc = exc.__cause__ or exc.__context__


def _error_metadata(exc):
    # Only provider response metadata; never inspect request headers.
    return [value for error in _error_chain(exc)
            for attr in ("body", "response_json", "details")
            if isinstance(value := getattr(error, attr, None), (dict, list))]


def classify_error(exc: Exception) -> Failure:
    if isinstance(exc, (AttributeError, TypeError, AssertionError)):
        return Failure("programming")
    chain = list(_error_chain(exc))
    if any(isinstance(error, (AttributeError, TypeError, AssertionError)) for error in chain):
        return Failure("programming")
    statuses = [value for error in chain for value in (
        getattr(error, "status_code", None), getattr(error, "code", None),
        getattr(getattr(error, "response", None), "status_code", None),
    ) if isinstance(value, int)]
    status = statuses[0] if statuses else None
    metadata = _error_metadata(exc)
    text = (" ".join(type(error).__name__ + " " + str(error) for error in chain)
            + " " + json.dumps(metadata, default=str)).lower()
    compact = text.replace(" ", "").replace("_", "").replace("-", "")
    # Infrastructure overload is not evidence that any quota is exhausted.
    if status == 503 or any(s in text for s in ("overloaded", "503 unavailable", "service unavailable")):
        return Failure("transient")
    if status in (401, 403) or any(s in compact for s in (
        "authenticationerror", "invalidapikey", "invalidkey", "unauthorized",
        "apikeynotvalid", "apikeyinvalid", "permissiondenied",
    )):
        return Failure("permanent", "key")
    if status == 429 or is_quota_exhaustion(text) or any(s in compact for s in ("ratelimit", "resourceexhausted", "insufficientquota", "insufficientcredits")):
        # Explicit key/account scope wins. 'PerProjectPerModel' alone does not
        # mean all models under that project's key are exhausted.
        key_scope = any(s in compact for s in (
            "accountwide", "keywide", "accountquota", "keyquota", "apikeyquota",
            "insufficientquota", "insufficientcredits", "creditbalance",
        ))
        model_scope = any(s in compact for s in (
            "permodel", "modelquota", "modelspecific", "ratelimitreachedformodel",
        ))
        def scopes(value):
            if isinstance(value, dict):
                for key, item in value.items():
                    if key.lower() in {"scope", "quota_scope", "quota_type"} and isinstance(item, str):
                        yield item.lower()
                    yield from scopes(item)
            elif isinstance(value, list):
                for item in value:
                    yield from scopes(item)
        reported_scopes = set(scopes(metadata))
        key_scope |= bool(reported_scopes & {"account", "key", "project", "organization", "global"})
        model_scope |= "model" in reported_scopes
        return Failure("quota", "key" if key_scope or not model_scope else "model")
    if (isinstance(status, int) and 500 <= status <= 599) or any(s in compact for s in (
        "timeout", "timedout", "connection", "unavailable", "badgateway", "servererror",
    )):
        return Failure("transient")
    if status in (400, 402, 404, 410, 422) or any(s in compact for s in (
        "invalidmodel", "modelnotfound", "modeldoesnotexist", "decommissioned", "modelnotavailable",
    )):
        return Failure("permanent", "key" if status == 402 else "model")
    # Unknown exceptions may be bugs. Do not silently reinterpret them as a
    # provider outage merely because a fallback exists.
    return Failure("programming")


def _is_temporary_provider_error(exc: Exception) -> bool:
    return classify_error(exc).kind in {"quota", "transient"}


class LLMServiceUnavailableError(RuntimeError):
    """All configured candidates failed or remain in cooldown."""
    def __init__(self, provider_errors: list[tuple[str, Exception]]):
        self.provider_errors = provider_errors
        details = "; ".join(f"{provider}: {error}" for provider, error in provider_errors)
        super().__init__("All configured LLM providers are temporarily unavailable. " + details)


def _configured_candidates():
    providers = (
        ("OpenRouter", config.OPENROUTER_TEXT_API_KEYS, config.OPENROUTER_TEXT_MODELS),
        ("Gemini", config.GEMINI_API_KEYS, config.GEMINI_MODELS),
        ("Groq", config.GROQ_API_KEYS, config.GROQ_MODELS),
    )
    for provider, keys, models in providers:
        for index, key in enumerate(keys, 1):
            for model in models:
                common = dict(model=model, temperature=0, timeout=60, max_retries=0)
                if provider == "OpenRouter":
                    client = ChatOpenAI(api_key=key, base_url="https://openrouter.ai/api/v1", **common)
                elif provider == "Gemini":
                    client = ChatGoogleGenerativeAI(google_api_key=key, **common)
                else:
                    client = ChatGroq(api_key=key, **common)
                yield Candidate(provider, f"key{index}", model, client, key)


class HybridLLM:
    """Reference-script rotation adapted to invoke(), without rebuilding input.

    Quota: record cooldown and rotate immediately (no wasted retry of a known
    exhausted quota). Transient infrastructure: two attempts, one second apart.
    Provider -> key -> model order is fixed; singular env settings still work.
    """
    def __init__(self, candidates=None, tracker=None, sleep=time.sleep):
        self.candidates = list(_configured_candidates() if candidates is None else candidates)
        self.tracker = tracker if tracker is not None else CooldownTracker()
        self._sleep = sleep
        # Keep the existing primary/fallback client attributes for callers that
        # inspect them. Rotation itself always uses the prepared candidate list.
        self.openrouter = next((c.client for c in self.candidates if c.provider == "OpenRouter"), None)
        self.gemini = next((c.client for c in self.candidates if c.provider == "Gemini"), None)
        self._permanent_models = set()
        if not self.candidates:
            raise RuntimeError("No text LLM candidates configured. Set text provider API key/model environment variables.")

    def _redact(self, value):
        text = str(value)
        for secret in sorted({c.secret for c in self.candidates if c.secret}, key=len, reverse=True):
            for form in (secret, quote(secret, safe="")):
                text = text.replace(form, "[REDACTED]")
        return text

    def _safe_argument(self, value):
        if isinstance(value, dict):
            return {self._redact(key): self._safe_argument(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return type(value)(self._safe_argument(item) for item in value)
        return self._redact(value) if not isinstance(value, (int, float, bool, type(None))) else value

    def invoke(self, input, **kwargs):
        failures = []
        eligible = [c for c in self.candidates if self.tracker.is_available(c.key_id)]
        available = [c for c in eligible if self.tracker.is_available(c.id)]
        # Reference tracker escape hatch: if *all* model-quota estimates are
        # stale, permit probing in original order. Never bypass key/account or
        # invalid-model cooldowns with this escape hatch.
        if not available and eligible:
            available = [c for c in eligible if c.id not in self._permanent_models]
        available_ids = {c.id for c in available}
        for candidate in self.candidates:
            label = self._redact(candidate.id)
            if candidate.id not in available_ids or not self.tracker.is_available(candidate.key_id):
                remaining = max(self.tracker.remaining(candidate.id), self.tracker.remaining(candidate.key_id))
                failures.append((label, RuntimeError(f"cooldown ({remaining:.0f}s remaining)")))
                continue
            for attempt in range(2):
                try:
                    result = candidate.client.invoke(input, **kwargs)
                except Exception as exc:
                    failure = classify_error(exc)
                    if failure.kind == "programming":
                        # Preserve the exception class and immediate propagation,
                        # but redact credentials if a broken client echoed one.
                        exc.args = tuple(self._safe_argument(arg) for arg in exc.args)
                        exc.__cause__ = None
                        exc.__context__ = None
                        raise exc from None
                    safe_error = self._redact(f"{type(exc).__name__}: {exc}")
                    logger.warning("LLM candidate=%s attempt=%s kind=%s scope=%s error=%s", label, attempt + 1, failure.kind, failure.scope, safe_error)
                    if failure.kind == "transient" and attempt == 0:
                        self._sleep(1)
                        continue
                    target = candidate.key_id if failure.scope == "key" else candidate.id
                    if failure.kind == "quota":
                        self.tracker.mark_rate_limited(target, str(exc) + " " + json.dumps(_error_metadata(exc), default=str), retry_after=parse_retry_after_seconds(exc))
                    elif failure.kind == "permanent":
                        self.tracker.mark_permanently_broken(target)
                        if failure.scope == "model":
                            self._permanent_models.add(candidate.id)
                    failures.append((label, RuntimeError(f"{failure.kind}/{failure.scope}: {safe_error}")))
                    break
                else:
                    self.tracker.mark_success(candidate.id)
                    self.tracker.mark_success(candidate.key_id)
                    self._permanent_models.discard(candidate.id)
                    return result
        # Do not chain a raw SDK exception, which can retain request credentials.
        raise LLMServiceUnavailableError(failures) from None


llm = HybridLLM()
