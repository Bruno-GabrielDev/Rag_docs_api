"""Recuperação híbrida (léxica + semântica) com fusão por RRF.

Motivação: busca vetorial sozinha erra em termos raros e exatos (códigos,
siglas, nomes próprios, números de artigo), porque o embedding "dilui" esse
sinal. BM25 sozinho erra em paráfrase. Combinar os dois cobre os dois buracos,
e Reciprocal Rank Fusion faz a combinação sem precisar calibrar pesos entre
escalas de pontuação que não são comparáveis.
"""

from __future__ import annotations

import math
import re
from collections import Counter

from .embeddings import EmbeddingProvider
from .models import Chunk, RetrievedChunk
from .store import VectorStore

TOKEN_RE = re.compile(r"\w+", re.UNICODE)

# Stopwords em PT/EN — reduzem ruído no BM25 sem afetar a busca vetorial.
STOPWORDS = {
    "a", "à", "ao", "aos", "as", "às", "com", "como", "da", "das", "de", "do",
    "dos", "e", "em", "entre", "eu", "é", "na", "nas", "no", "nos", "o", "os",
    "ou", "para", "pelo", "pela", "por", "que", "se", "sem", "ser", "sobre",
    "são", "uma", "um", "and", "are", "for", "from", "in", "is", "of", "on",
    "or", "the", "to", "what", "with",
}


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


class BM25:
    """Okapi BM25 sobre o corpus de chunks."""

    def __init__(self, corpus: list[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.docs = [tokenize(doc) for doc in corpus]
        self.doc_lengths = [len(d) for d in self.docs]
        self.avg_length = (sum(self.doc_lengths) / len(self.docs)) if self.docs else 0.0
        self.term_freqs = [Counter(d) for d in self.docs]

        doc_freq: Counter[str] = Counter()
        for doc in self.docs:
            doc_freq.update(set(doc))

        n = len(self.docs)
        self.idf = {
            term: math.log(1 + (n - df + 0.5) / (df + 0.5)) for term, df in doc_freq.items()
        }

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        if not self.docs:
            return []
        terms = tokenize(query)
        scores: list[float] = []
        for i, tf in enumerate(self.term_freqs):
            score = 0.0
            length_norm = self.k1 * (
                1 - self.b + self.b * (self.doc_lengths[i] / (self.avg_length or 1))
            )
            for term in terms:
                freq = tf.get(term)
                if not freq:
                    continue
                score += self.idf.get(term, 0.0) * (freq * (self.k1 + 1)) / (freq + length_norm)
            scores.append(score)
        ranked = sorted(enumerate(scores), key=lambda x: -x[1])
        return [(i, s) for i, s in ranked[:k] if s > 0]


def reciprocal_rank_fusion(
    rankings: list[list[int]], k: int = 60
) -> list[tuple[int, float]]:
    """Funde listas ordenadas usando 1 / (k + posição).

    Só a posição importa, não a pontuação bruta — por isso funciona entre
    rankers com escalas totalmente diferentes.
    """
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(fused.items(), key=lambda x: -x[1])


class HybridRetriever:
    def __init__(
        self,
        store: VectorStore,
        embeddings: EmbeddingProvider,
        rrf_k: int = 60,
    ) -> None:
        self.store = store
        self.embeddings = embeddings
        self.rrf_k = rrf_k
        self.bm25 = BM25([c.text for c in store.chunks])

    def retrieve(
        self, query: str, top_k: int = 4, candidate_k: int = 12
    ) -> list[RetrievedChunk]:
        if len(self.store) == 0:
            return []

        query_vector = self.embeddings.embed_query(query)
        dense = self.store.search(query_vector, candidate_k)
        lexical = self.bm25.search(query, candidate_k)

        dense_scores = dict(dense)
        fused = reciprocal_rank_fusion(
            [[i for i, _ in dense], [i for i, _ in lexical]], k=self.rrf_k
        )

        results: list[RetrievedChunk] = []
        for index, _rrf_score in fused[:top_k]:
            chunk: Chunk = self.store.chunks[index]
            # Reportamos a similaridade de cosseno como score porque ela é
            # interpretável (0 a 1) e serve de limiar para o guardrail.
            score = dense_scores.get(index)
            if score is None:
                score = float(self.store.vectors[index] @ query_vector)
            results.append(RetrievedChunk(chunk=chunk, score=score))

        return sorted(results, key=lambda r: -r.score)
