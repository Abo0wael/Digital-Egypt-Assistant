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

from typing import Iterator

from langsmith import traceable

from core.langsmith_config import langsmith_config


@traceable(name="answer_agent", project_name=langsmith_config.project)
def stream_answer(chain_with_history, session_id: str, user_input: str) -> Iterator[str]:
    """Stream answer chunks for ``user_input``."""
    for chunk in chain_with_history.stream(
        {"input": user_input},
        config={"configurable": {"session_id": session_id}},
    ):
        yield chunk
