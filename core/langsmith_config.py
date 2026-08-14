"""LangSmith tracing configuration.

Reads the LANGCHAIN_-prefixed environment variables LangSmith's SDK expects
(LANGCHAIN_TRACING_V2, LANGCHAIN_API_KEY, LANGCHAIN_PROJECT, ...) from .env,
mirroring counselling-bot's core/src/langsmith_config.py. Agents pass
``langsmith_config.project`` explicitly to @traceable rather than relying
solely on env-var auto-detection, so trace grouping doesn't depend on
process environment alone.
"""

from typing import Optional

from pydantic_settings import BaseSettings


class LangSmithConfig(BaseSettings):
    tracing_v2: bool = False
    endpoint: str = "https://api.smith.langchain.com"
    api_key: Optional[str] = None
    project: str = "digital-egypt-assistant"

    class Config:
        env_file = ".env"
        env_prefix = "LANGCHAIN_"
        extra = "ignore"


langsmith_config = LangSmithConfig()
