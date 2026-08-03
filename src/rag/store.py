"""Vector store simples, em memória, com persistência em disco.

Implementado com numpy em vez de um banco vetorial pronto porque, nesta escala
(milhares de chunks), a busca exaustiva por cosseno é instantânea e o código
deixa explícito o que um índice vetorial faz por baixo. A interface é pequena o
suficiente para trocar por pgvector/Qdrant sem tocar no resto do pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .models import Chunk


class VectorStore:
    def __init__(self, dimension: int) -> None:
        self.dimension = dimension
        self.chunks: list[Chunk] = []
        self.vectors = np.zeros((0, dimension), dtype=np.float32)

    def __len__(self) -> int:
        return len(self.chunks)

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks e vectors precisam ter o mesmo tamanho")
        if len(chunks) == 0:
            return
        if vectors.shape[1] != self.dimension:
            raise ValueError(
                f"dimensão incompatível: esperado {self.dimension}, recebido {vectors.shape[1]}"
            )
        self.chunks.extend(chunks)
        self.vectors = np.vstack([self.vectors, vectors.astype(np.float32)])

    def search(self, query_vector: np.ndarray, k: int) -> list[tuple[int, float]]:
        """Retorna (índice, similaridade de cosseno) dos k chunks mais próximos."""
        if len(self) == 0:
            return []
        scores = self.vectors @ query_vector.astype(np.float32)
        k = min(k, len(scores))
        # argpartition evita ordenar o array inteiro: O(n) em vez de O(n log n).
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(int(i), float(scores[i])) for i in top]

    # --- persistência ---

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "vectors.npy", self.vectors)
        payload = {
            "dimension": self.dimension,
            "chunks": [c.model_dump() for c in self.chunks],
        }
        (directory / "chunks.json").write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8"
        )

    @classmethod
    def load(cls, directory: Path) -> VectorStore:
        payload = json.loads((directory / "chunks.json").read_text(encoding="utf-8"))
        store = cls(dimension=payload["dimension"])
        store.chunks = [Chunk(**c) for c in payload["chunks"]]
        store.vectors = np.load(directory / "vectors.npy")
        return store

    @staticmethod
    def exists(directory: Path) -> bool:
        return (directory / "chunks.json").exists() and (directory / "vectors.npy").exists()
