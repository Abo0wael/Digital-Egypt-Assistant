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

# Groq retired the two Llama models previously listed here on 2026-08-16.
# Keep this list aligned with Groq's production model IDs: an unavailable
# model is accepted by ChatGroq at construction time but fails with a 404 only
# when the first response (or query rewrite) is generated.
MODEL_CHOICES = [
    "Groq - GPT-OSS 120B",
    "Groq - GPT-OSS 20B",
    "Groq - Qwen 3.6 27B",
]

_ENV_VARS = {
    "Groq - GPT-OSS 120B": "GROQ_API_KEY",
    "Groq - GPT-OSS 20B": "GROQ_API_KEY",
    "Groq - Qwen 3.6 27B": "GROQ_API_KEY",
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
    else:
        # A tracing flag without a key causes a failed LangSmith request for
        # every LLM call. Keep tracing optional for local and Cloud deploys.
        os.environ["LANGSMITH_TRACING"] = "false"
        os.environ["LANGCHAIN_TRACING_V2"] = "false"

    if not os.getenv("LANGSMITH_PROJECT") and not os.getenv("LANGCHAIN_PROJECT"):
        os.environ["LANGSMITH_PROJECT"] = "digital-egypt-assistant"
        os.environ["LANGCHAIN_PROJECT"] = "digital-egypt-assistant"



def _build_groq_gpt_oss_20b() -> ChatGroq:
    key = os.getenv("GROQ_API_KEY", "").strip()
    return ChatGroq(model="openai/gpt-oss-20b", api_key=key, temperature=0)


def _build_groq_gpt_oss() -> ChatGroq:
    key = os.getenv("GROQ_API_KEY", "").strip()
    return ChatGroq(model="openai/gpt-oss-120b", api_key=key, temperature=0)


def _build_groq_qwen() -> ChatGroq:
    key = os.getenv("GROQ_API_KEY", "").strip()
    return ChatGroq(model="qwen/qwen3.6-27b", api_key=key, temperature=0)


_BUILDERS = {
    "Groq - GPT-OSS 120B": _build_groq_gpt_oss,
    "Groq - GPT-OSS 20B": _build_groq_gpt_oss_20b,
    "Groq - Qwen 3.6 27B": _build_groq_qwen,
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

