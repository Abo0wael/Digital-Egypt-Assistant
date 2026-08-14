"""LLM client construction for each supported provider.

Keeping provider wiring here (rather than inline in app.py) means the
Streamlit page only asks for "give me an LLM for this choice" and stays
unaware of per-provider constructor quirks. API keys come from the process
environment (populated from .env by load_dotenv() in app.py) instead of a
sidebar input, so each provider's client reads its own standard env var.
"""

import os

from langchain_groq import ChatGroq
from langsmith import tracing_context

# Only free fast models are included — no paid API keys required
MODEL_CHOICES = [
    "Groq - LLaMA 3.3 70B",
    "Groq - LLaMA 3.1 8B",
    "Groq - GPT-OSS 120B",
]

_ENV_VARS = {
    "Groq - LLaMA 3.3 70B": "GROQ_API_KEY",
    "Groq - LLaMA 3.1 8B": "GROQ_API_KEY",
    "Groq - GPT-OSS 120B": "GROQ_API_KEY",
}


def _build_groq_llama_70b() -> ChatGroq:
    return ChatGroq(model="llama-3.3-70b-versatile", temperature=0)


def _build_groq_llama_8b() -> ChatGroq:
    return ChatGroq(model="llama-3.1-8b-instant", temperature=0)


def _build_groq_gpt_oss() -> ChatGroq:
    # openai/gpt-oss-120b: available free on Groq as replacement for DeepSeek R1
    return ChatGroq(model="openai/gpt-oss-120b", temperature=0)


_BUILDERS = {
    "Groq - LLaMA 3.3 70B": _build_groq_llama_70b,
    "Groq - LLaMA 3.1 8B": _build_groq_llama_8b,
    "Groq - GPT-OSS 120B": _build_groq_gpt_oss,
}


def build_llm(model_choice: str):
    """Build and smoke-test an LLM client for the chosen provider.

    Raises a clear error if the provider's env var isn't set in .env, and
    whatever the underlying client raises on a bad key; callers are expected
    to catch it and surface a message to the user.
    """
    env_var = _ENV_VARS[model_choice]
    if not os.getenv(env_var):
        raise RuntimeError(f"{env_var} غير موجود في ملف .env")

    llm = _BUILDERS[model_choice]()
    return llm
