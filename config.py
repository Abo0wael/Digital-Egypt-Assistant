"""App-wide configuration.

Mirrors the counselling-bot pattern of small pydantic config models
(app/core/agents/vectorstore_config.py) instead of scattering literals
across the UI and chain-building code.
"""

from pydantic import BaseModel


class AppConfig(BaseModel):
    page_title: str = "المساعد الرقمي"
    app_title: str = "💬 المساعد الرقمي لخدمات مصر"
    app_subtitle: str = "اسألني عن أي خدمة رقمية متاحة على بوابة مصر الرقمية"


class VectorStoreConfig(BaseModel):
    embeddings_name: str = "intfloat/multilingual-e5-large"
    persist_directory: str = "infloat"
    top_k: int = 10


app_config = AppConfig()
vector_store_config = VectorStoreConfig()
