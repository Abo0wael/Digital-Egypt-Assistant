"""Summary agent: condenses the running chat history into an Arabic digest.

Mirrors counselling-bot's utils/summarization_service.py — a single LLM call
over the conversation so far, kept independent from the answer agent so it
can be triggered on demand (e.g. a sidebar "summarize" action) without
touching the retrieval chain. @traceable wraps it as a named run in
LangSmith so it's visible as its own trace, separate from answer_agent.
"""

from typing import List

from langsmith import traceable

from core.langsmith_config import langsmith_config

# FIX (System vs Human Message separation for Summarization Agent):
# Refactored prompt from raw string to ChatPromptTemplate with explicit SystemMessage and HumanMessage.
# This ensures LLM treats system directives separately from conversation content, avoiding accidental Q&A.
from langchain_core.prompts import ChatPromptTemplate

_SUMMARY_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "أنت مساعد متخصص في تلخيص المحادثات الحكومية.\n"
            "مهمتك: تلخيص سجل المحادثة المرفق بين المستخدم والمساعد الرقمي في نقاط موجزة وواضحة باللغة العربية، "
            "مع التركيز على الأسئلة الرئيسية التي طرحها المستخدم والإجابات الهامة المقدمة له.",
        ),
        ("human", "سجل المحادثة للتلخيص:\n\n{history}"),
    ]
)


def _format_history(messages: List) -> str:
    lines = []
    for msg in messages:
        role = "المستخدم" if msg.type == "human" else "المساعد"
        lines.append(f"{role}: {msg.content}")
    return "\n".join(lines)


@traceable(name="summary_agent", run_type="chain", project_name=langsmith_config.project)
def summarize_history(llm, messages: List) -> str:
    """Summarize ``messages`` (StreamlitChatMessageHistory.messages) via the LLM.

    Returns a fixed message if there is nothing to summarize yet, and a short
    error note (rather than raising) if the LLM call fails, so callers always
    get a displayable string back.
    """
    if not messages:
        return "لا توجد محادثة بعد لتلخيصها."

    summary_chain = _SUMMARY_PROMPT_TEMPLATE | llm

    try:
        response = summary_chain.invoke({"history": _format_history(messages)})
        return response.content if hasattr(response, "content") else str(response)
    except Exception as e:
        return f"تعذّر إنشاء الملخص: {e}"

