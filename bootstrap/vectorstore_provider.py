"""Process-wide provider for the shared embedding model and vector store.

Mirrors the embedding_provider pattern from the counselling-bot project:
construction is expensive (downloads/loads a HF model), so it is centralized
here behind Streamlit's own resource cache instead of re-created per rerun.
"""

import streamlit as st
from langchain_chroma import Chroma
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_huggingface import HuggingFaceEmbeddings

from config import vector_store_config as config


@st.cache_resource
def get_embedding_model() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=config.embeddings_name)


@st.cache_resource
def get_vector_store() -> Chroma:
    return Chroma(
        embedding_function=get_embedding_model(),
        persist_directory=config.persist_directory,
    )


def get_retriever() -> VectorStoreRetriever:
    return get_vector_store().as_retriever(search_kwargs={"k": config.top_k})
