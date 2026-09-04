from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

from .config import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
    GROQ_API_KEY,
    GROQ_MODEL,
)


class HybridLLM:
    """
    Gemini is the primary provider.
    Groq is used automatically if Gemini fails
    because of quota, rate limit, timeout, or provider error.
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
        # Primary: Gemini
        if self.gemini is not None:
            try:
                return self.gemini.invoke(
                    input,
                    **kwargs,
                )

            except Exception as exc:
                print(
                    "[LLM] Gemini failed. "
                    f"Falling back to Groq. "
                    f"Reason: {type(exc).__name__}"
                )

        # Fallback: Groq
        if self.groq is not None:
            try:
                return self.groq.invoke(
                    input,
                    **kwargs,
                )

            except Exception as exc:
                raise RuntimeError(
                    "All LLM providers failed. "
                    f"Groq error: {type(exc).__name__}: {exc}"
                ) from exc

        raise RuntimeError(
            "Gemini failed and Groq fallback is not configured."
        )


llm = HybridLLM()