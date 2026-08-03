"""Prompts versionados.

Prompt é artefato de produção: fica em módulo próprio, com versão, para que a
suíte de avaliação consiga atribuir uma variação de qualidade a uma mudança
específica de texto.
"""

from __future__ import annotations

from .models import RetrievedChunk

PROMPT_VERSION = "v2-2026-08"

SYSTEM_PROMPT = """Você é um assistente que responde perguntas EXCLUSIVAMENTE com base nos trechos de documentos fornecidos.

Regras:
1. Use apenas informação presente nos trechos. Não complete com conhecimento próprio.
2. Cite a fonte de cada afirmação com marcadores no formato [1], [2], correspondentes ao número do trecho.
3. Se os trechos não contiverem a resposta, responda exatamente: "Não encontrei essa informação nos documentos fornecidos." Não tente adivinhar.
4. Se os trechos se contradizerem, aponte a contradição em vez de escolher um lado.
5. Responda em português do Brasil, de forma direta e objetiva."""

USER_TEMPLATE = """<trechos>
{context}
</trechos>

<pergunta>
{question}
</pergunta>

Responda à pergunta usando apenas os trechos acima, com citações [n]."""


def format_context(chunks: list[RetrievedChunk]) -> str:
    blocks = []
    for i, retrieved in enumerate(chunks, start=1):
        blocks.append(
            f"[{i}] (fonte: {retrieved.chunk.doc_id})\n{retrieved.chunk.text}"
        )
    return "\n\n".join(blocks)


def build_user_message(question: str, chunks: list[RetrievedChunk]) -> str:
    return USER_TEMPLATE.format(context=format_context(chunks), question=question)
