"""Métricas de qualidade do retrieval.

Separadas da avaliação de geração de propósito: quando a resposta final piora,
a primeira pergunta é sempre "o retrieval trouxe o trecho certo?". Sem medir as
duas etapas em separado não dá para saber onde mexer.
"""

from __future__ import annotations


def hit_at_k(retrieved_doc_ids: list[str], expected_doc_ids: list[str], k: int) -> float:
    """1.0 se algum documento esperado aparece nos k primeiros."""
    return float(bool(set(retrieved_doc_ids[:k]) & set(expected_doc_ids)))


def reciprocal_rank(retrieved_doc_ids: list[str], expected_doc_ids: list[str]) -> float:
    """1/posição do primeiro acerto. Penaliza o acerto que veio em último lugar."""
    expected = set(expected_doc_ids)
    for position, doc_id in enumerate(retrieved_doc_ids, start=1):
        if doc_id in expected:
            return 1.0 / position
    return 0.0


def recall_at_k(retrieved_doc_ids: list[str], expected_doc_ids: list[str], k: int) -> float:
    """Fração dos documentos esperados que foram recuperados nos k primeiros."""
    if not expected_doc_ids:
        return 1.0
    found = set(retrieved_doc_ids[:k]) & set(expected_doc_ids)
    return len(found) / len(set(expected_doc_ids))


def aggregate(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
