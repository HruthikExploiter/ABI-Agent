"""
src/config.py
Loads settings from config/config.yaml and makes them
available to every other file in the project.
"""

from __future__ import annotations

import yaml
from functools import lru_cache
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelPairConfig(BaseSettings):
    primary: str
    fallback: str


class LLMConfig(BaseSettings):
    provider: str = "google"
    temperature: float = 0.0
    max_tokens: int = 4096
    max_retries: int = 3
    models: dict[str, ModelPairConfig] = Field(default_factory=dict)

    @property
    def active_primary_model(self) -> str:
        return self.models[self.provider].primary

    @property
    def active_fallback_model(self) -> str:
        return self.models[self.provider].fallback


class AgentConfig(BaseSettings):
    max_iterations: int = 10
    self_heal_retries: int = 3
    enable_sql_generation: bool = True
    enable_visualization: bool = True
    verbose: bool = True


class DataConfig(BaseSettings):
    raw_path: str = "data/raw/"
    processed_path: str = "data/processed/"
    lazy_scan: bool = True
    infer_schema_length: int = 1000


class StreamlitConfig(BaseSettings):
    page_title: str = "ABI Agent"
    page_icon: str = "📦"
    layout: str = "wide"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="ABI_",
        env_nested_delimiter="__",
        case_sensitive=False,
    )
    llm: LLMConfig = Field(default_factory=LLMConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    streamlit: StreamlitConfig = Field(default_factory=StreamlitConfig)


def _load_yaml() -> dict:
    path = Path(__file__).parent.parent / "config" / "config.yaml"
    if not path.exists():
        return {}
    with path.open("r") as f:
        return yaml.safe_load(f) or {}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    raw = _load_yaml()
    llm_raw = raw.get("llm", {})
    models_raw = llm_raw.pop("models", {})
    llm_obj = LLMConfig(
        **llm_raw,
        models={k: ModelPairConfig(**v) for k, v in models_raw.items()},
    )
    return Settings(
        llm=llm_obj,
        agent=AgentConfig(**raw.get("agent", {})),
        data=DataConfig(**raw.get("data", {})),
        streamlit=StreamlitConfig(**raw.get("streamlit", {})),
    )