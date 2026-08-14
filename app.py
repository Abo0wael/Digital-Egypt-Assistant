"""Streamlit entry point — Digital Egypt AI Assistant (Groq Powered)."""

from dotenv import load_dotenv

load_dotenv()

from uuid import uuid4

import streamlit as st
from langchain_community.chat_message_histories import StreamlitChatMessageHistory

from bootstrap.llm_provider import get_llm
from bootstrap.vectorstore_provider import get_retriever
from config import app_config
from core.prompts import build_prompt
from core.rag_chain import build_conversational_chain
st.set_page_config(
    page_title=app_config.page_title,
    page_icon="🇪🇬",
    layout="centered",
    initial_sidebar_state="expanded",
)

from ui import (
    apply_egypt_theme,
    get_user_input,
    render_answer,
    render_footer,
    render_header,
    render_history,
    render_quick_services,
    render_sidebar,
    render_summary_action,
)

# Apply Egyptian Flag Luxury Theme
apply_egypt_theme()

if "session_id" not in st.session_state:
    st.session_state["session_id"] = str(uuid4())
session_id = st.session_state["session_id"]

model_choice = render_sidebar()
render_header()

llm = None
try:
    llm = get_llm(model_choice)
except Exception as e:
    st.error(f"⚠️ فشل تهيئة النموذج: {e}")

if llm is not None:
    retriever = get_retriever()
    prompt = build_prompt()
    memory = StreamlitChatMessageHistory()
    chain_with_history = build_conversational_chain(llm, retriever, prompt, memory)

    render_summary_action(llm, memory)
    render_quick_services()
    render_history(memory)

    user_input = get_user_input()
    if user_input:
        render_answer(chain_with_history, session_id, user_input)

render_footer()

