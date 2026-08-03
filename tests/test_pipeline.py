"""Testes do pipeline RAG.

O foco aqui é o comportamento que protege o usuário: recusar responder sem
contexto, e nunca emitir uma citação que não aponte para um trecho real.
"""

from __future__ import annotations

from src.rag.llm import FakeLLM
from src.rag.pipeline import NO_ANSWER, RAGPipeline


def test_resposta_traz_citacoes_rastreaveis(pipeline):
    answer = pipeline.answer("quantos revisores aprovam um pull request")
    assert answer.citations
    assert answer.citations[0].marker == 1
    assert answer.citations[0].chunk_id
    assert answer.grounded is True


def test_guardrail_impede_chamada_ao_llm_sem_contexto_relevante(retriever):
    llm = FakeLLM()
    pipeline = RAGPipeline(retriever=retriever, llm=llm, top_k=3, min_score=0.99)

    answer = pipeline.answer("qual a receita de strogonoff de camarão")

    assert answer.answer == NO_ANSWER
    assert answer.grounded is False
    assert answer.citations == []
    assert llm.calls == [], "o LLM não deveria ter sido chamado"


def test_citacao_fora_do_intervalo_e_descartada(retriever):
    # O modelo cita [9] quando só existem 3 trechos: citação inválida some.
    llm = FakeLLM(response="Conforme o documento [1] e também [9].")
    pipeline = RAGPipeline(retriever=retriever, llm=llm, top_k=3, min_score=0.0)

    answer = pipeline.answer("revisores do pull request")

    assert [c.marker for c in answer.citations] == [1]


def test_citacoes_repetidas_sao_deduplicadas(retriever):
    llm = FakeLLM(response="Início [1], meio [2] e fim [1].")
    pipeline = RAGPipeline(retriever=retriever, llm=llm, top_k=3, min_score=0.0)

    answer = pipeline.answer("pull request revisores")

    assert [c.marker for c in answer.citations] == [1, 2]


def test_prompt_recebe_os_trechos_recuperados(pipeline, fake_llm):
    pipeline.answer("post-mortem sem culpados")

    user_message = fake_llm.calls[0]["user"]
    assert "<trechos>" in user_message
    assert "[1]" in user_message
    assert "post-mortem" in user_message.lower()


def test_resposta_registra_latencia_e_uso_de_tokens(pipeline):
    answer = pipeline.answer("plantão turnos")
    assert answer.latency_ms >= 0
    assert answer.usage["model"] == "fake"


def test_recusa_explicita_do_modelo_marca_resposta_como_nao_fundamentada(retriever):
    llm = FakeLLM(response=NO_ANSWER)
    pipeline = RAGPipeline(retriever=retriever, llm=llm, top_k=3, min_score=0.0)

    assert pipeline.answer("assunto qualquer indexado").grounded is False
