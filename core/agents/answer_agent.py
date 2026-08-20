"""Answer agent: runs one chat turn through the retrieval chain.

Mirrors the agent functions in counselling-bot's core/agents/agents.py —
a single-purpose callable that owns one LLM interaction and stays
independent of how the caller chooses to render it. @traceable wraps it as
a named run in LangSmith so a chat turn shows up as one trace instead of
raw chain calls.

Deliberately does NOT touch chat memory: the RunnableWithMessageHistory
built in core/rag_chain.py already writes the human/AI turn to the history
object once the stream completes. Writing it again here duplicated every
turn (each question and answer showing up twice after the following turn).
"""

import logging
from typing import Iterator

from groq import (
    APIConnectionError,
    APIStatusError,
    AuthenticationError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
)
from langsmith import traceable

from core.langsmith_config import langsmith_config

logger = logging.getLogger(__name__)


def _groq_error_message(error: Exception) -> str:
    """Return a safe Arabic message without exposing keys or API internals."""
    if isinstance(error, NotFoundError):
        return (
            "تعذّر استخدام نموذج الذكاء الاصطناعي المحدد حاليًا. "
            "يرجى اختيار نموذج آخر من القائمة أو المحاولة لاحقًا."
        )
    if isinstance(error, AuthenticationError):
        return "تعذّر الاتصال بخدمة الذكاء الاصطناعي بسبب مشكلة في مفتاح Groq."
    if isinstance(error, PermissionDeniedError):
        return "مفتاح Groq الحالي لا يملك صلاحية استخدام النموذج المحدد."
    if isinstance(error, RateLimitError):
        return "تم بلوغ حد استخدام خدمة الذكاء الاصطناعي مؤقتًا. يرجى المحاولة بعد قليل."
    if isinstance(error, APIConnectionError):
        return "تعذّر الاتصال بخدمة الذكاء الاصطناعي. يرجى التحقق من الاتصال والمحاولة لاحقًا."
    if isinstance(error, APIStatusError):
        return "حدث خطأ مؤقت في خدمة الذكاء الاصطناعي. يرجى المحاولة لاحقًا."
    return "تعذّر إنشاء الرد بسبب خطأ غير متوقع. يرجى المحاولة مرة أخرى."


@traceable(name="answer_agent", project_name=langsmith_config.project)
def stream_answer(chain_with_history, session_id: str, user_input: str) -> Iterator[str]:
    """Stream answer chunks for ``user_input``."""
    try:
        for chunk in chain_with_history.stream(
            {"input": user_input},
            config={"configurable": {"session_id": session_id}},
        ):
            yield chunk
    except Exception as error:
        logger.exception("Response generation failed")
        yield _groq_error_message(error)
