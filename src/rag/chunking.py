"""Divisão de documentos em chunks.

A estratégia é recursiva: tenta quebrar primeiro nos separadores mais
"semânticos" (parágrafo), e só desce para separadores menores (linha, frase,
espaço) quando o bloco ainda é grande demais. Isso evita cortar uma frase no
meio, que é a principal causa de chunks inúteis no retrieval.
"""

from __future__ import annotations

import hashlib
import re
from itertools import pairwise

from .models import Chunk

SEPARATORS = ["\n\n", "\n", ". ", " "]


def _split_recursive(text: str, chunk_size: int, separators: list[str]) -> list[str]:
    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    if not separators:
        # Sem separador possível: corte duro (caso raro, ex. base64 gigante).
        return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

    sep, rest = separators[0], separators[1:]
    parts = text.split(sep)

    chunks: list[str] = []
    buffer = ""
    for part in parts:
        candidate = f"{buffer}{sep}{part}" if buffer else part
        if len(candidate) <= chunk_size:
            buffer = candidate
            continue
        if buffer:
            chunks.append(buffer)
        if len(part) > chunk_size:
            chunks.extend(_split_recursive(part, chunk_size, rest))
            buffer = ""
        else:
            buffer = part
    if buffer.strip():
        chunks.append(buffer)
    return [c for c in chunks if c.strip()]


def _apply_overlap(chunks: list[str], overlap: int) -> list[str]:
    """Prefixa cada chunk com o final do anterior.

    O overlap existe para que uma resposta que "atravessa" a fronteira de dois
    chunks continue recuperável por pelo menos um deles.
    """
    if overlap <= 0 or len(chunks) < 2:
        return chunks
    out = [chunks[0]]
    for prev, current in pairwise(chunks):
        tail = prev[-overlap:].lstrip()
        out.append(f"{tail} {current}".strip())
    return out


def normalize(text: str) -> str:
    """Normaliza espaços em branco sem destruir a separação de parágrafos."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_document(
    doc_id: str,
    text: str,
    chunk_size: int = 800,
    chunk_overlap: int = 150,
    metadata: dict | None = None,
) -> list[Chunk]:
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap precisa ser menor que chunk_size")

    text = normalize(text)
    raw_chunks = _split_recursive(text, chunk_size, SEPARATORS)
    raw_chunks = _apply_overlap(raw_chunks, chunk_overlap)

    chunks: list[Chunk] = []
    for position, raw in enumerate(raw_chunks):
        digest = hashlib.sha1(f"{doc_id}:{position}:{raw}".encode()).hexdigest()[:12]
        chunks.append(
            Chunk(
                id=f"{doc_id}#{position}#{digest}",
                doc_id=doc_id,
                text=raw.strip(),
                position=position,
                metadata=metadata or {},
            )
        )
    return chunks
