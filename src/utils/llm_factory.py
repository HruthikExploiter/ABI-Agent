"""
src/utils/llm_factory.py
Builds the correct LLM client based on config.yaml provider setting.
"""

from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

from langchain_core.language_models import BaseChatModel
from loguru import logger

from src.config import get_settings


def build_llm(use_fallback: bool = False) -> BaseChatModel:
    """
    Returns a LangChain chat model based on the configured provider.
    """
    settings = get_settings()
    provider = settings.llm.provider
    model_name = (
        settings.llm.active_fallback_model
        if use_fallback
        else settings.llm.active_primary_model
    )

    # Strip "groq/" prefix if present — langchain-groq adds it automatically
    model_name = model_name.replace("groq/", "")

    logger.info(f"Building LLM | provider={provider} | model={model_name}")

    common_kwargs = {
        "model": model_name,
        "temperature": settings.llm.temperature,
        "max_tokens": settings.llm.max_tokens,
    }

    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(**common_kwargs)

    elif provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(**common_kwargs)

    else:
        raise ValueError(
            f"Unsupported provider: '{provider}'. "
            f"Valid options: google | groq"
        )