"""Process-wide provider for the active chat LLM.

Wraps core.llm_factory.build_llm in Streamlit's resource cache, keyed by
model_choice. Without this, Streamlit's rerun-the-whole-script-per-interaction
model meant every click/question rebuilt the client and re-ran its smoke-test
call — burning an extra API call per rerun and flooding LangSmith with
duplicate "human: مرحبا" traces unrelated to any real chat turn.
"""

import os
import streamlit as st

from core.llm_factory import build_llm, sync_streamlit_secrets


@st.cache_resource(show_spinner=False)
def _get_cached_llm(model_choice: str, api_key: str):
    return build_llm(model_choice)


def get_llm(model_choice: str):
    sync_streamlit_secrets()
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    return _get_cached_llm(model_choice, api_key)

