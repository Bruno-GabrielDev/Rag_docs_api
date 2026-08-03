"""API HTTP do RAG."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.rag.config import Settings, get_settings
from src.rag.embeddings import build_embedding_provider
from src.rag.llm import AnthropicLLM
from src.rag.models import Answer
from src.rag.pipeline import RAGPipeline
from src.rag.retriever import HybridRetriever
from src.rag.store import VectorStore

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    top_k: int | None = Field(default=None, ge=1, le=10)


def build_pipeline(settings: Settings) -> RAGPipeline:
    """Monta o pipeline uma única vez — carregar índice e modelo de embeddings
    a cada requisição custaria segundos por chamada."""
    if not VectorStore.exists(settings.index_dir):
        raise RuntimeError(
            f"Índice não encontrado em {settings.index_dir}. Rode: python -m src.rag.ingest"
        )
    store = VectorStore.load(settings.index_dir)
    embeddings = build_embedding_provider(settings)
    retriever = HybridRetriever(store, embeddings, rrf_k=settings.rrf_k)
    llm = AnthropicLLM(api_key=settings.anthropic_api_key, model=settings.llm_model)
    return RAGPipeline(
        retriever=retriever,
        llm=llm,
        top_k=settings.top_k,
        candidate_k=settings.candidate_k,
        min_score=settings.min_score,
        max_tokens=settings.llm_max_tokens,
        temperature=settings.llm_temperature,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.pipeline = build_pipeline(settings)
    logger.info("Pipeline pronto: %d chunks indexados", len(app.state.pipeline.retriever.store))
    yield
    app.state.pipeline = None


app = FastAPI(
    title="RAG Docs API",
    description="Pergunte em linguagem natural sobre uma base de documentos, com respostas citadas.",
    version="1.0.0",
    lifespan=lifespan,
)


def get_pipeline() -> RAGPipeline:
    pipeline = getattr(app.state, "pipeline", None)
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline indisponível")
    return pipeline


@app.get("/health")
def health() -> dict:
    pipeline = getattr(app.state, "pipeline", None)
    return {
        "status": "ok" if pipeline else "degraded",
        "indexed_chunks": len(pipeline.retriever.store) if pipeline else 0,
    }


@app.post("/ask", response_model=Answer)
def ask(request: AskRequest, pipeline: Annotated[RAGPipeline, Depends(get_pipeline)]) -> Answer:
    if request.top_k:
        pipeline.top_k = request.top_k
    try:
        return pipeline.answer(request.question)
    except RuntimeError as exc:
        # LLM indisponível após os retries: 502, não 500 — o erro é upstream.
        logger.error("Falha ao responder: %s", exc)
        raise HTTPException(status_code=502, detail="Serviço de LLM indisponível") from exc


@app.post("/search")
def search(request: AskRequest, pipeline: Annotated[RAGPipeline, Depends(get_pipeline)]) -> dict:
    """Retorna só o retrieval, sem gerar resposta. Útil para depurar o índice."""
    results = pipeline.retriever.retrieve(
        request.question,
        top_k=request.top_k or pipeline.top_k,
        candidate_k=pipeline.candidate_k,
    )
    return {
        "question": request.question,
        "results": [
            {"doc_id": r.chunk.doc_id, "score": round(r.score, 4), "text": r.chunk.text}
            for r in results
        ],
    }
