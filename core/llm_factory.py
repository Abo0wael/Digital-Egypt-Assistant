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


def sync_streamlit_secrets():
    """Sync Streamlit Secrets (st.secrets) to os.environ for deployment compatibility."""
    try:
        import streamlit as st
        if hasattr(st, "secrets"):
            for k, v in st.secrets.items():
                if isinstance(v, str):
                    clean_v = v.strip().replace("\n", "").replace("\r", "")
                    if clean_v:
                        os.environ[k] = clean_v
    except Exception:
        pass

    # Ensure LangSmith / LangChain tracing variables are synced across environment keys
    ls_key = os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")
    if ls_key:
        clean_key = ls_key.strip().replace("\n", "").replace("\r", "")
        os.environ["LANGSMITH_API_KEY"] = clean_key
        os.environ["LANGCHAIN_API_KEY"] = clean_key
        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ["LANGCHAIN_TRACING_V2"] = "true"

    if not os.getenv("LANGSMITH_PROJECT") and not os.getenv("LANGCHAIN_PROJECT"):
        os.environ["LANGSMITH_PROJECT"] = "digital-egypt-assistant"
        os.environ["LANGCHAIN_PROJECT"] = "digital-egypt-assistant"



def _build_groq_llama_70b() -> ChatGroq:
    key = os.getenv("GROQ_API_KEY", "").strip()
    return ChatGroq(model="llama-3.3-70b-versatile", api_key=key, temperature=0)


def _build_groq_llama_8b() -> ChatGroq:
    key = os.getenv("GROQ_API_KEY", "").strip()
    return ChatGroq(model="llama-3.1-8b-instant", api_key=key, temperature=0)


def _build_groq_gpt_oss() -> ChatGroq:
    key = os.getenv("GROQ_API_KEY", "").strip()
    return ChatGroq(model="openai/gpt-oss-120b", api_key=key, temperature=0)


_BUILDERS = {
    "Groq - LLaMA 3.3 70B": _build_groq_llama_70b,
    "Groq - LLaMA 3.1 8B": _build_groq_llama_8b,
    "Groq - GPT-OSS 120B": _build_groq_gpt_oss,
}


def build_llm(model_choice: str):
    """Build and smoke-test an LLM client for the chosen provider."""
    sync_streamlit_secrets()
    env_var = _ENV_VARS[model_choice]
    key_val = os.getenv(env_var, "").strip().replace("\n", "").replace("\r", "")
    if not key_val:
        raise RuntimeError(f"{env_var} غير موجود في ملف .env أو في Streamlit Secrets")

    os.environ[env_var] = key_val
    llm = _BUILDERS[model_choice]()
    return llm

