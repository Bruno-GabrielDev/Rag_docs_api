"""Configuração central da aplicação, carregada de variáveis de ambiente."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- LLM ---
    anthropic_api_key: str = ""
    llm_model: str = "claude-sonnet-5"
    llm_max_tokens: int = 1024
    llm_temperature: float = 0.0

    # --- Embeddings ---
    # "local" = sentence-transformers (sem custo), "voyage" ou "openai" = API
    embedding_provider: str = "local"
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    voyage_api_key: str = ""
    openai_api_key: str = ""

    # --- Chunking ---
    chunk_size: int = 400
    chunk_overlap: int = 80

    # --- Retrieval ---
    top_k: int = 4
    candidate_k: int = 12
    min_score: float = 0.25  # abaixo disso, o pipeline responde "não sei"
    rrf_k: int = 60

    # --- Persistência ---
    docs_dir: Path = PROJECT_ROOT / "data" / "docs"
    index_dir: Path = PROJECT_ROOT / "data" / "index"


@lru_cache
def get_settings() -> Settings:
    return Settings()
