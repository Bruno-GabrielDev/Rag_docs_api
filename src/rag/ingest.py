"""Ingestão: lê documentos do disco, faz chunking, gera embeddings e salva o índice.

Uso:
    python -m src.rag.ingest
    python -m src.rag.ingest --docs-dir ./data/docs --rebuild
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .chunking import chunk_document
from .config import get_settings
from .embeddings import build_embedding_provider
from .models import Chunk
from .store import VectorStore

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = {".md", ".txt", ".pdf"}


def read_document(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        return "\n\n".join(page.extract_text() or "" for page in reader.pages)
    return path.read_text(encoding="utf-8")


def load_chunks(docs_dir: Path, chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    files = sorted(p for p in docs_dir.rglob("*") if p.suffix.lower() in SUPPORTED_SUFFIXES)

    if not files:
        raise SystemExit(f"Nenhum documento encontrado em {docs_dir}")

    for path in files:
        text = read_document(path)
        if not text.strip():
            logger.warning("Documento vazio, ignorado: %s", path.name)
            continue
        doc_chunks = chunk_document(
            doc_id=path.name,
            text=text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            metadata={"source": str(path.relative_to(docs_dir))},
        )
        logger.info("%-40s %3d chunks", path.name, len(doc_chunks))
        chunks.extend(doc_chunks)

    return chunks


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Indexa documentos para o RAG")
    parser.add_argument("--docs-dir", type=Path, default=settings.docs_dir)
    parser.add_argument("--index-dir", type=Path, default=settings.index_dir)
    parser.add_argument("--rebuild", action="store_true", help="Sobrescreve o índice existente")
    args = parser.parse_args()

    if VectorStore.exists(args.index_dir) and not args.rebuild:
        raise SystemExit("Índice já existe. Use --rebuild para reconstruir.")

    chunks = load_chunks(args.docs_dir, settings.chunk_size, settings.chunk_overlap)
    logger.info("Total: %d chunks. Gerando embeddings...", len(chunks))

    embeddings = build_embedding_provider(settings)
    vectors = embeddings.embed_documents([c.text for c in chunks])

    store = VectorStore(dimension=embeddings.dimension)
    store.add(chunks, vectors)
    store.save(args.index_dir)

    logger.info("Índice salvo em %s (%d vetores, dim=%d)", args.index_dir, len(store), store.dimension)


if __name__ == "__main__":
    main()
