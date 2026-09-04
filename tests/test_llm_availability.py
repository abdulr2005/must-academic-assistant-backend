import pytest

from app.llm import HybridLLM, LLMServiceUnavailableError


class FailingProvider:
    def __init__(self, error):
        self.error = error

    def invoke(self, input, **kwargs):
        raise self.error


def hybrid_without_constructor(groq, gemini):
    hybrid = object.__new__(HybridLLM)
    hybrid.groq = groq
    hybrid.gemini = gemini
    return hybrid


def test_all_temporary_provider_failures_raise_dedicated_error():
    hybrid = hybrid_without_constructor(
        FailingProvider(RuntimeError("429 rate limit")),
        FailingProvider(RuntimeError("429 RESOURCE_EXHAUSTED")),
    )
    with pytest.raises(LLMServiceUnavailableError) as caught:
        hybrid.invoke("hello")
    assert [name for name, _ in caught.value.provider_errors] == [
        "Groq", "Gemini"
    ]


def test_programming_error_is_not_reclassified_or_hidden():
    hybrid = hybrid_without_constructor(
        FailingProvider(AttributeError("application bug")),
        FailingProvider(RuntimeError("429 RESOURCE_EXHAUSTED")),
    )
    with pytest.raises(AttributeError, match="application bug"):
        hybrid.invoke("hello")
