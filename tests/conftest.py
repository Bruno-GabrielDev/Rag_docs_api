"""Fixtures compartilhadas.

A suíte inteira roda sem rede, sem chave de API e sem baixar modelo: os
embeddings são substituídos por um hashing determinístico e o LLM por um
dublê. Isso mantém o CI rápido, gratuito e reprodutível — teste que depende de
resposta de LLM é teste instável.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.rag.chunking import chunk_document
from src.rag.embeddings import _l2_normalize
from src.rag.llm import FakeLLM
from src.rag.pipeline import RAGPipeline
from src.rag.retriever import HybridRetriever, tokenize
from src.rag.store import VectorStore


class HashingEmbeddings:
    """Bag-of-words projetado por hash em um espaço fixo.

    Não captura sinônimos, mas é determinístico e reproduz a propriedade que
    importa nos testes: textos com vocabulário parecido ficam próximos.
    """

    def __init__(self, dimension: int = 64) -> None:
        self.dimension = dimension

    def _vectorize(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dimension, dtype=np.float32)
        for token in tokenize(text):
            vector[hash(token) % self.dimension] += 1.0
        return vector

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return _l2_normalize(np.vstack([self._vectorize(t) for t in texts]))

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([text])[0]


SAMPLE_DOCS = {
    "politica.md": (
        "Todo pull request precisa de aprovação de dois revisores.\n\n"
        "A cobertura mínima de testes é de oitenta por cento de linhas.\n\n"
        "O deploy em produção é congelado às sextas-feiras."
    ),
    "runbook.md": (
        "Incidentes SEV1 têm tempo de resposta de quinze minutos.\n\n"
        "O post-mortem é obrigatório e sem culpados.\n\n"
        "O plantão funciona em turnos semanais."
    ),
}


@pytest.fixture
def embeddings() -> HashingEmbeddings:
    return HashingEmbeddings()


@pytest.fixture
def store(embeddings: HashingEmbeddings) -> VectorStore:
    chunks = []
    for doc_id, text in SAMPLE_DOCS.items():
        chunks.extend(chunk_document(doc_id, text, chunk_size=120, chunk_overlap=20))
    vectors = embeddings.embed_documents([c.text for c in chunks])
    store = VectorStore(dimension=embeddings.dimension)
    store.add(chunks, vectors)
    return store


@pytest.fixture
def retriever(store: VectorStore, embeddings: HashingEmbeddings) -> HybridRetriever:
    return HybridRetriever(store, embeddings)


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM(response="Segundo a política, são necessários dois revisores [1].")


@pytest.fixture
def pipeline(retriever: HybridRetriever, fake_llm: FakeLLM) -> RAGPipeline:
    return RAGPipeline(retriever=retriever, llm=fake_llm, top_k=3, min_score=0.05)
