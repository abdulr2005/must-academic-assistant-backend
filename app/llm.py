import logging

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

from .config import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
    GROQ_API_KEY,
    GROQ_MODEL,
)


logger = logging.getLogger(__name__)


class LLMServiceUnavailableError(RuntimeError):
    """All configured text LLM providers are temporarily unavailable."""

    def __init__(self, provider_errors: list[tuple[str, Exception]]):
        self.provider_errors = provider_errors
        details = "; ".join(
            f"{provider}: {type(error).__name__}: {error}"
            for provider, error in provider_errors
        )
        super().__init__(
            "All configured LLM providers are temporarily unavailable. "
            + details
        )


def _is_temporary_provider_error(exc: Exception) -> bool:
    """Recognize provider/API availability failures without hiding code bugs."""

    if isinstance(exc, (AttributeError, TypeError, AssertionError)):
        return False

    status_code = getattr(exc, "status_code", None)
    if status_code == 429 or (
        isinstance(status_code, int)
        and 500 <= status_code <= 599
    ):
        return True

    error_name = type(exc).__name__.lower()
    error_text = str(exc).lower()

    availability_markers = (
        "ratelimit",
        "rate_limit",
        "resourceexhausted",
        "resource_exhausted",
        "timeout",
        "connection",
        "serviceunavailable",
        "temporarily unavailable",
        "provider unavailable",
        "429",
        "quota exhausted",
    )

    return any(
        marker in error_name or marker in error_text
        for marker in availability_markers
    )


class HybridLLM:
    """
    Groq is the primary provider for text chat.
    Gemini is used automatically as fallback.
    """

    def __init__(self):
        self.gemini = None
        self.groq = None

        if GEMINI_API_KEY:
            self.gemini = ChatGoogleGenerativeAI(
                model=GEMINI_MODEL,
                google_api_key=GEMINI_API_KEY,
            )

        if GROQ_API_KEY:
            self.groq = ChatGroq(
                api_key=GROQ_API_KEY,
                model=GROQ_MODEL,
                temperature=0,
            )

        if self.gemini is None and self.groq is None:
            raise RuntimeError(
                "No LLM provider configured. "
                "Set GEMINI_API_KEY or GROQ_API_KEY."
            )

    def invoke(self, input, **kwargs):
        provider_errors = []

        # Primary: Groq
        if self.groq is not None:
            try:
                return self.groq.invoke(
                    input,
                    **kwargs,
                )

            except Exception as exc:
                if not _is_temporary_provider_error(exc):
                    raise

                provider_errors.append(("Groq", exc))
                logger.warning(
                    "Groq invocation failed; trying Gemini | error=%s: %s",
                    type(exc).__name__,
                    exc,
                )

        # Fallback: Gemini
        if self.gemini is not None:
            try:
                return self.gemini.invoke(
                    input,
                    **kwargs,
                )

            except Exception as exc:
                if not _is_temporary_provider_error(exc):
                    raise

                provider_errors.append(("Gemini", exc))

        if provider_errors:
            error = LLMServiceUnavailableError(
                provider_errors
            )
            raise error from provider_errors[-1][1]

        raise RuntimeError(
            "No configured LLM provider was available for invocation."
        )


llm = HybridLLM()
