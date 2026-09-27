"""Process-local provider cooldowns adapted from the supplied reference tracker.

Only confirmed quota failures belong in ``mark_rate_limited``. Transient
overloads/timeouts do not establish that a key or model has exhausted quota.
State is intentionally not persisted, and expiry uses a monotonic clock.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from datetime import timezone
from email.utils import parsedate_to_datetime
import math
import re
import threading
import time
from typing import TypeVar


_DEFAULT_COOLDOWN_SECONDS = 60.0
_PERMANENT_COOLDOWN_SECONDS = 6 * 60 * 60.0
_DAILY_QUOTA_BASE_COOLDOWN_SECONDS = 30 * 60.0
_DAILY_QUOTA_MAX_COOLDOWN_SECONDS = 4 * 60 * 60.0

_QUOTA_EXHAUSTION_MARKERS = (
    "429", "resource_exhausted", "resourceexhausted", "quota exceeded",
    "quota exhausted", "quotaid", "rate_limit_exceeded", "ratelimit",
    "rate limit reached", "rate limit exceeded", "tokens per day",
    "requests per day", "tokens per minute", "requests per minute",
)
_OVERLOAD_MARKERS = ("503", "overloaded", "service unavailable", "unavailable")
_RETRY_TEXT_RE = re.compile(
    r"(?:retry|try again)\s+in\s+"
    r"((?:\d+(?:\.\d+)?\s*(?:hours?|hrs?|h|minutes?|mins?|m|seconds?|secs?|s)\s*)+)",
    re.IGNORECASE,
)
_DURATION_PART_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(hours?|hrs?|h|minutes?|mins?|m|seconds?|secs?|s)",
    re.IGNORECASE,
)
_RETRY_DELAY_TEXT_RE = re.compile(
    r"[\"']?retry_?delay[\"']?\s*[:=]\s*[\"']?(\d+(?:\.\d+)?\s*s)",
    re.IGNORECASE,
)
_Candidate = TypeVar("_Candidate")


def is_quota_exhaustion(error_message: str) -> bool:
    """Recognize quota text, preserving the reference's overload precedence."""
    text = str(error_message).lower()
    if any(marker in text for marker in _OVERLOAD_MARKERS):
        return False
    return any(marker in text for marker in _QUOTA_EXHAUSTION_MARKERS)


def _nonnegative_number(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) and number >= 0 else None


def _duration_seconds(value: object) -> float | None:
    """Read seconds, provider duration strings, or protobuf-style durations."""
    if isinstance(value, Mapping):
        seconds = _nonnegative_number(value.get("seconds", 0))
        nanos = _nonnegative_number(value.get("nanos", 0))
        if ("seconds" in value or "nanos" in value) and seconds is not None and nanos is not None:
            return seconds + nanos / 1_000_000_000
        return None
    if not isinstance(value, (str, int, float)):
        seconds = getattr(value, "seconds", None)
        if seconds is not None:
            return _duration_seconds({"seconds": seconds, "nanos": getattr(value, "nanos", 0)})
        return None
    number = _nonnegative_number(value)
    if number is not None:
        return number
    if not isinstance(value, str):
        return None
    parts = list(_DURATION_PART_RE.finditer(value))
    if not parts or _DURATION_PART_RE.sub("", value).strip():
        return None
    total = 0.0
    for part in parts:
        number = float(part.group(1))
        unit = part.group(2).lower()[0]
        total += number * {"h": 3600, "m": 60, "s": 1}[unit]
    return total if math.isfinite(total) else None


def _retry_after_seconds(value: object) -> float | None:
    seconds = _duration_seconds(value)
    if seconds is not None:
        return seconds
    if isinstance(value, str):
        try:
            date = parsedate_to_datetime(value)
            if date.tzinfo is None:
                date = date.replace(tzinfo=timezone.utc)
            return max(0.0, date.timestamp() - time.time())
        except (TypeError, ValueError, OverflowError):
            pass
    return None


def _structured_retry_delay(
    value: object, seen: set[int] | None = None, depth: int = 0,
) -> float | None:
    if seen is None:
        seen = set()
    if id(value) in seen or depth >= 16:
        return None
    seen.add(id(value))
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key).lower().replace("_", "") == "retrydelay":
                delay = _duration_seconds(child)
                if delay is not None:
                    return delay
        children = value.values()
    elif isinstance(value, (list, tuple)):
        children = value
    else:
        for attr in ("retry_delay", "retryDelay"):
            delay = _duration_seconds(getattr(value, attr, None))
            if delay is not None:
                return delay
        return None
    for child in children:
        delay = _structured_retry_delay(child, seen, depth + 1)
        if delay is not None:
            return delay
    return None


def parse_retry_after_seconds(error: object) -> float | None:
    """Read Retry-After headers, structured retryDelay, then provider text.

    LangChain wraps Gemini errors, retaining original provider metadata in the
    exception cause. Inspect at most eight distinct linked errors, and prefer
    headers/structured delays over text even when the metadata is on a cause.
    HTTP dates require wall time only to compute a duration; stored deadlines
    still use the tracker monotonic clock. No request is made.
    """
    errors = []
    pending = [error]
    seen = set()
    while pending and len(errors) < 8:
        current = pending.pop(0)
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        errors.append(current)
        pending.extend((getattr(current, "__cause__", None), getattr(current, "__context__", None)))

    for current in errors:
        for owner in (current, getattr(current, "response", None)):
            headers = getattr(owner, "headers", None)
            if isinstance(headers, Mapping):
                for key, value in headers.items():
                    if str(key).lower() == "retry-after":
                        delay = _retry_after_seconds(value)
                        if delay is not None:
                            return delay

    for current in errors:
        for payload in (current, getattr(current, "body", None), getattr(current, "details", None)):
            delay = _structured_retry_delay(payload)
            if delay is not None:
                return delay
        response = getattr(current, "response", None)
        if response is not None and callable(getattr(response, "json", None)):
            try:
                payload = response.json()
            except ValueError:
                payload = None
            delay = _structured_retry_delay(payload)
            if delay is not None:
                return delay

    for current in errors:
        text = str(current)
        match = _RETRY_TEXT_RE.search(text) or _RETRY_DELAY_TEXT_RE.search(text)
        if match:
            delay = _duration_seconds(match.group(1))
            if delay is not None:
                return delay
    return None


class CooldownTracker:
    """Thread-safe cooldown state shared for the lifetime of its owner/process."""

    def __init__(self, clock: Callable[[], float] = time.monotonic):
        self._clock = clock
        self._lock = threading.Lock()
        self._cooldowns: dict[str, float] = {}
        self._daily_quota_failure_counts: dict[str, int] = {}

    def is_available(self, candidate_id: str) -> bool:
        return self.remaining(candidate_id) <= 0

    def remaining(self, candidate_id: str) -> float:
        with self._lock:
            until = self._cooldowns.get(candidate_id)
            return 0.0 if until is None else max(0.0, until - self._clock())

    def mark_rate_limited(
        self, candidate_id: str, error_message: str, retry_after: object = None,
    ) -> float:
        compact = re.sub(r"[\s_-]+", "", str(error_message).lower())
        daily = "perday" in compact or bool(
            re.search(r"\b(?:daily|rpd|tpd)\b", str(error_message), re.IGNORECASE)
        )
        delay = _retry_after_seconds(retry_after)
        if delay is None:
            delay = parse_retry_after_seconds(error_message)
        if delay is None:
            delay = _DEFAULT_COOLDOWN_SECONDS
        with self._lock:
            if daily:
                failure_count = self._daily_quota_failure_counts.get(candidate_id, 0) + 1
                self._daily_quota_failure_counts[candidate_id] = failure_count
                delay = min(
                    _DAILY_QUOTA_BASE_COOLDOWN_SECONDS * 2 ** min(failure_count - 1, 3),
                    _DAILY_QUOTA_MAX_COOLDOWN_SECONDS,
                )
            self._cooldowns[candidate_id] = self._clock() + delay
        return delay

    def mark_success(self, candidate_id: str) -> None:
        """A successful probe clears both the deadline and daily escalation."""
        with self._lock:
            self._cooldowns.pop(candidate_id, None)
            self._daily_quota_failure_counts.pop(candidate_id, None)

    def mark_permanently_broken(self, candidate_id: str) -> float:
        with self._lock:
            self._cooldowns[candidate_id] = self._clock() + _PERMANENT_COOLDOWN_SECONDS
        return _PERMANENT_COOLDOWN_SECONDS

    def filter_available(
        self, candidate_ids: Iterable[_Candidate], probe_if_all: bool = True,
    ) -> list[_Candidate]:
        """Keep order; optionally probe the original list if every item cools.

        ``probe_if_all=False`` lets callers exclude exhausted whole keys even
        while applying the reference's stale-cooldown escape to model rotation.
        """
        candidates = list(candidate_ids)
        with self._lock:
            now = self._clock()
            available = [c for c in candidates if self._cooldowns.get(c, now) <= now]
        return available if available or not probe_if_all else candidates

    def clear(self) -> None:
        """Reset state for tests; no production request should call this."""
        with self._lock:
            self._cooldowns.clear()
            self._daily_quota_failure_counts.clear()
