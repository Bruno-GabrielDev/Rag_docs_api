"""Modelos de domínio compartilhados por todas as camadas."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """Trecho indexável de um documento."""

    id: str
    doc_id: str
    text: str
    position: int
    metadata: dict = Field(default_factory=dict)


class RetrievedChunk(BaseModel):
    """Chunk recuperado, com a pontuação atribuída pelo retriever."""

    chunk: Chunk
    score: float


class Citation(BaseModel):
    marker: int
    doc_id: str
    chunk_id: str
    snippet: str


class Answer(BaseModel):
    question: str
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    grounded: bool = True
    latency_ms: int = 0
    usage: dict = Field(default_factory=dict)
