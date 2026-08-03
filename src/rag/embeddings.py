"""Geração de embeddings.

A interface é um Protocol para que o resto do sistema não dependa de nenhum
fornecedor específico — trocar sentence-transformers por Voyage/OpenAI é uma
mudança de configuração, não de código.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np


class EmbeddingProvider(Protocol):
    dimension: int

    def embed_documents(self, texts: list[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """Normaliza para norma 1, assim o produto escalar vira similaridade de cosseno."""
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.clip(norms, 1e-12, None)


class LocalEmbeddings:
    """sentence-transformers rodando na própria máquina (sem custo por token)."""

    def __init__(self, model_name: str, batch_size: int = 32) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        self._batch_size = batch_size
        self.dimension = self._model.get_sentence_embedding_dimension()

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        vectors = self._model.encode(
            texts, batch_size=self._batch_size, show_progress_bar=False
        )
        return _l2_normalize(np.asarray(vectors, dtype=np.float32))

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([text])[0]


class VoyageEmbeddings:
    """Voyage AI — provedor de embeddings recomendado pela Anthropic."""

    def __init__(self, api_key: str, model: str = "voyage-3", batch_size: int = 64) -> None:
        import voyageai

        self._client = voyageai.Client(api_key=api_key)
        self._model = model
        self._batch_size = batch_size
        self.dimension = 1024

    def _embed(self, texts: list[str], input_type: str) -> np.ndarray:
        out: list[list[float]] = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            result = self._client.embed(batch, model=self._model, input_type=input_type)
            out.extend(result.embeddings)
        return _l2_normalize(np.asarray(out, dtype=np.float32))

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self._embed(texts, "document")

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed([text], "query")[0]


class OpenAIEmbeddings:
    def __init__(
        self, api_key: str, model: str = "text-embedding-3-small", batch_size: int = 64
    ) -> None:
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key)
        self._model = model
        self._batch_size = batch_size
        self.dimension = 1536

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        out: list[list[float]] = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            response = self._client.embeddings.create(model=self._model, input=batch)
            out.extend(item.embedding for item in response.data)
        return _l2_normalize(np.asarray(out, dtype=np.float32))

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([text])[0]


def build_embedding_provider(settings) -> EmbeddingProvider:
    provider = settings.embedding_provider.lower()
    if provider == "local":
        return LocalEmbeddings(settings.embedding_model)
    if provider == "voyage":
        return VoyageEmbeddings(settings.voyage_api_key)
    if provider == "openai":
        return OpenAIEmbeddings(settings.openai_api_key)
    raise ValueError(f"embedding_provider desconhecido: {settings.embedding_provider}")
