import os
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


# Gemini
GEMINI_API_KEY = os.getenv(
    "GEMINI_API_KEY",
    ""
).strip()

GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.6-flash"
).strip()


# Groq
GROQ_API_KEY = os.getenv(
    "GROQ_API_KEY",
    ""
).strip()

GROQ_MODEL = os.getenv(
    "GROQ_MODEL",
    "openai/gpt-oss-120b"
).strip()


# RAG API
RAG_API_URL = os.getenv(
    "RAG_API_URL",
    "https://must-rag-api.onrender.com/rag/search"
).strip()


# OpenRouter Vision Parser
OPENROUTER_API_KEY = os.getenv(
    "OPENROUTER_API_KEY",
    ""
).strip()

OPENROUTER_MODEL = os.getenv(
    "OPENROUTER_MODEL",
    "openrouter/free"
).strip()


def require_env(*names: str) -> None:
    values = {
        "GEMINI_API_KEY": GEMINI_API_KEY,
        "GEMINI_MODEL": GEMINI_MODEL,
        "GROQ_API_KEY": GROQ_API_KEY,
        "GROQ_MODEL": GROQ_MODEL,
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