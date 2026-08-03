"""Testes da camada de avaliação.

A suíte de avaliação é o instrumento de medida do projeto: se ela estiver
errada, todas as decisões tomadas a partir dela também estarão. Por isso as
métricas têm teste próprio, com valores calculados à mão.
"""

from __future__ import annotations

import pytest

from evaluation.judge import Judge, _parse_json
from evaluation.metrics import aggregate, hit_at_k, recall_at_k, reciprocal_rank
from src.rag.llm import FakeLLM

# --- Métricas de retrieval ---


@pytest.mark.parametrize(
    ("retrieved", "expected", "k", "esperado"),
    [
        (["a.md", "b.md"], ["a.md"], 2, 1.0),
        (["b.md", "c.md"], ["a.md"], 2, 0.0),
        (["b.md", "a.md"], ["a.md"], 1, 0.0),  # acerto ficou fora do corte
        ([], ["a.md"], 3, 0.0),
    ],
)
def test_hit_at_k(retrieved, expected, k, esperado):
    assert hit_at_k(retrieved, expected, k) == esperado


@pytest.mark.parametrize(
    ("retrieved", "expected", "esperado"),
    [
        (["a.md", "b.md"], ["a.md"], 1.0),
        (["b.md", "a.md"], ["a.md"], 0.5),
        (["b.md", "c.md", "a.md"], ["a.md"], pytest.approx(1 / 3)),
        (["x.md"], ["a.md"], 0.0),
    ],
)
def test_reciprocal_rank(retrieved, expected, esperado):
    assert reciprocal_rank(retrieved, expected) == esperado


def test_recall_parcial():
    assert recall_at_k(["a.md", "x.md"], ["a.md", "b.md"], k=2) == 0.5


def test_recall_com_lista_esperada_vazia_e_um():
    assert recall_at_k(["a.md"], [], k=2) == 1.0


def test_aggregate_de_lista_vazia_nao_divide_por_zero():
    assert aggregate([]) == 0.0


# --- Parser do juiz ---


def test_parse_json_simples():
    assert _parse_json('{"score": 5, "reason": "ok"}')["score"] == 5


def test_parse_json_envolvido_em_cercas_de_markdown():
    assert _parse_json('```json\n{"score": 4, "reason": "quase"}\n```')["score"] == 4


def test_parse_json_com_texto_ao_redor():
    assert _parse_json('Claro! {"score": 3, "reason": "meio"} Espero ter ajudado.')["score"] == 3


def test_parse_json_invalido_retorna_score_zero_em_vez_de_explodir():
    resultado = _parse_json("não é json de jeito nenhum")
    assert resultado["score"] == 0
    assert "reason" in resultado


def test_judge_converte_score_para_float():
    judge = Judge(FakeLLM(response='{"score": "5", "reason": "fiel"}'))
    assert judge.faithfulness("contexto", "resposta")["score"] == 5.0


def test_judge_relevance_monta_prompt_com_pergunta_e_esperado():
    llm = FakeLLM(response='{"score": 4, "reason": "boa"}')
    Judge(llm).relevance("Qual o SLA?", "24 horas", "São 24 horas.")
    prompt = llm.calls[0]["user"]
    assert "Qual o SLA?" in prompt
    assert "24 horas" in prompt
