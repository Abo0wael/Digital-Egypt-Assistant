"""Assembles the retrieval-augmented, history-aware conversational chain.

Built from plain langchain_core LCEL primitives rather than the
langchain.chains.* helpers (create_retrieval_chain, create_stuff_documents_chain)
they were dropped from the top-level `langchain` package in LangChain 1.0,
so relying on langchain_core directly keeps this module from breaking again
on the next dependency bump.
"""

from typing import List

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory


from core.prompts import build_rephrase_prompt


def _format_docs(docs: List[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def build_conversational_chain(llm, retriever, prompt, memory) -> RunnableWithMessageHistory:
    rephrase_chain = (build_rephrase_prompt() | llm | StrOutputParser()).with_config(
        run_name="rewrite_query"
    )

    def _get_search_query(x):
        chat_history = x.get("chat_history", [])
        if chat_history:
            return rephrase_chain.invoke(x)
        return x["input"]

    retrieve_context = (
        RunnableLambda(_get_search_query) | retriever | _format_docs
    ).with_config(run_name="retrieve_context")

    answer_chain = (
        RunnablePassthrough.assign(context=retrieve_context)
        | prompt
        | llm
        | StrOutputParser()
    ).with_config(run_name="generate_answer")

    return RunnableWithMessageHistory(
        answer_chain,
        lambda session_id: memory,
        input_messages_key="input",
        history_messages_key="chat_history",
    )

