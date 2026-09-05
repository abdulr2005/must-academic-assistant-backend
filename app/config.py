import os
from pathlib import Path
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


# =========================================================
# Gemini - Text fallback
# =========================================================

GEMINI_API_KEY = os.getenv(
    "GEMINI_API_KEY",
    ""
).strip()

GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.6-flash"
).strip()


# =========================================================
# Groq - Legacy / optional
# =========================================================

GROQ_API_KEY = os.getenv(
    "GROQ_API_KEY",
    ""
).strip()

GROQ_MODEL = os.getenv(
    "GROQ_MODEL",
    "openai/gpt-oss-120b"
).strip()


# =========================================================
# OpenRouter - Text LLM
# Primary text model: MiniMax M3
# =========================================================

OPENROUTER_TEXT_API_KEY = os.getenv(
    "OPENROUTER_TEXT_API_KEY",
    ""
).strip()

OPENROUTER_TEXT_MODEL = os.getenv(
    "OPENROUTER_TEXT_MODEL",
    "minimax/minimax-m3:free"
).strip()


def _ordered_env_values(name: str, fallback: str) -> tuple[str, ...]:
    """Optional comma-separated rotation, preserving order and removing duplicates."""
    return tuple(dict.fromkeys(
        value.strip() for value in os.getenv(name, fallback).split(",")
        if value.strip()
    ))


# Plural settings replace their singular counterpart when supplied. These are
# text-only; the separate vision settings below retain their existing behavior.
OPENROUTER_TEXT_API_KEYS = _ordered_env_values("OPENROUTER_TEXT_API_KEYS", OPENROUTER_TEXT_API_KEY)
OPENROUTER_TEXT_MODELS = _ordered_env_values("OPENROUTER_TEXT_MODELS", OPENROUTER_TEXT_MODEL)
GEMINI_API_KEYS = _ordered_env_values("GEMINI_API_KEYS", GEMINI_API_KEY)
GEMINI_MODELS = _ordered_env_values("GEMINI_MODELS", GEMINI_MODEL)
GROQ_API_KEYS = _ordered_env_values("GROQ_API_KEYS", GROQ_API_KEY)
GROQ_MODELS = _ordered_env_values("GROQ_MODELS", GROQ_MODEL)


# =========================================================
# RAG API
# =========================================================

RAG_API_URL = os.getenv(
    "RAG_API_URL",
    "https://must-rag-api.onrender.com/rag/search"
).strip()


# =========================================================
# OpenRouter - Vision Parser
# DO NOT merge this with the text model configuration
# =========================================================

OPENROUTER_API_KEY = os.getenv(
    "OPENROUTER_API_KEY",
    ""
).strip()

OPENROUTER_MODEL = os.getenv(
    "OPENROUTER_MODEL",
    "openrouter/free"
).strip()


# =========================================================
# Environment validation
# =========================================================

def require_env(*names: str) -> None:
    values = {
        "GEMINI_API_KEY": GEMINI_API_KEY,
        "GEMINI_MODEL": GEMINI_MODEL,

        "GROQ_API_KEY": GROQ_API_KEY,
        "GROQ_MODEL": GROQ_MODEL,

        "OPENROUTER_TEXT_API_KEY": OPENROUTER_TEXT_API_KEY,
        "OPENROUTER_TEXT_MODEL": OPENROUTER_TEXT_MODEL,

        "OPENROUTER_API_KEY": OPENROUTER_API_KEY,
        "OPENROUTER_MODEL": OPENROUTER_MODEL,

        "RAG_API_URL": RAG_API_URL,
    }

    missing = [
        name
        for name in names
        if not values.get(name)
    ]

    if missing:
        raise RuntimeError(
            "Missing environment variable(s): "
            + ", ".join(missing)
        )
