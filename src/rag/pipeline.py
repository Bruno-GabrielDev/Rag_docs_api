"""Orquestração do RAG: recuperar → montar prompt → gerar → validar citações."""

from __future__ import annotations

import logging
import re
import time

from .llm import LLMClient
from .models import Answer, Citation
from .prompts import SYSTEM_PROMPT, build_user_message
from .retriever import HybridRetriever

logger = logging.getLogger(__name__)

NO_ANSWER = "Não encontrei essa informação nos documentos fornecidos."
CITATION_RE = re.compile(r"\[(\d+)\]")


class RAGPipeline:
    def __init__(
        self,
        retriever: HybridRetriever,
        llm: LLMClient,
        top_k: int = 4,
        candidate_k: int = 12,
        min_score: float = 0.25,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> None:
        self.retriever = retriever
        self.llm = llm
        self.top_k = top_k
        self.candidate_k = candidate_k
        self.min_score = min_score
        self.max_tokens = max_tokens
        self.temperature = temperature

    def answer(self, question: str) -> Answer:
        started = time.perf_counter()

        retrieved = self.retriever.retrieve(
            question, top_k=self.top_k, candidate_k=self.candidate_k
        )

        # Guardrail de entrada: sem contexto suficientemente relevante, nem
        # chamamos o modelo. Isso corta a maior fonte de alucinação em RAG e,
        # de quebra, economiza tokens.
        if not retrieved or retrieved[0].score < self.min_score:
            logger.info(
                "Contexto abaixo do limiar (%.3f < %.3f) para: %s",
                retrieved[0].score if retrieved else 0.0,
                self.min_score,
                question,
            )
            return Answer(
                question=question,
                answer=NO_ANSWER,
                citations=[],
                grounded=False,
                latency_ms=int((time.perf_counter() - started) * 1000),
            )

        response = self.llm.complete(
            system=SYSTEM_PROMPT,
            user=build_user_message(question, retrieved),
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )

        citations = self._extract_citations(response.text, retrieved)
        grounded = NO_ANSWER not in response.text

        return Answer(
            question=question,
            answer=response.text.strip(),
            citations=citations,
            grounded=grounded,
            latency_ms=int((time.perf_counter() - started) * 1000),
            usage=response.usage,
        )

    @staticmethod
    def _extract_citations(text: str, retrieved) -> list[Citation]:
        """Converte marcadores [n] em citações rastreáveis.

        Marcadores fora do intervalo são descartados: o modelo às vezes inventa
        um [5] quando só existem 4 trechos, e citação inválida é pior que
        citação ausente.
        """
        citations: list[Citation] = []
        seen: set[int] = set()
        for match in CITATION_RE.finditer(text):
            marker = int(match.group(1))
            if marker in seen or not (1 <= marker <= len(retrieved)):
                continue
            seen.add(marker)
            chunk = retrieved[marker - 1].chunk
            citations.append(
                Citation(
                    marker=marker,
                    doc_id=chunk.doc_id,
                    chunk_id=chunk.id,
                    snippet=chunk.text[:240],
                )
            )
        return sorted(citations, key=lambda c: c.marker)
