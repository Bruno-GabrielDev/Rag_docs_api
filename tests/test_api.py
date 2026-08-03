"""Testes da camada HTTP.

O pipeline real é substituído por um montado com dublês, via
`dependency_overrides` — os testes de API verificam contrato (status, schema,
validação), não a qualidade da resposta do modelo.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api.main import app, get_pipeline


@pytest.fixture
def client(pipeline):
    app.state.pipeline = pipeline
    app.dependency_overrides[get_pipeline] = lambda: pipeline
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()
    app.state.pipeline = None


@pytest.fixture(autouse=True)
def _skip_lifespan(monkeypatch, pipeline):
    """Impede que o lifespan tente carregar índice e modelo reais."""
    monkeypatch.setattr("src.api.main.build_pipeline", lambda settings: pipeline)


def test_health_reporta_chunks_indexados(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["indexed_chunks"] > 0


def test_ask_retorna_resposta_com_citacoes(client):
    response = client.post("/ask", json={"question": "quantos revisores aprovam um PR?"})
    assert response.status_code == 200
    body = response.json()
    assert body["answer"]
    assert body["citations"]
    assert "chunk_id" in body["citations"][0]


def test_pergunta_curta_demais_e_rejeitada(client):
    assert client.post("/ask", json={"question": "oi"}).status_code == 422


def test_pergunta_ausente_e_rejeitada(client):
    assert client.post("/ask", json={}).status_code == 422


def test_top_k_fora_do_intervalo_e_rejeitado(client):
    response = client.post("/ask", json={"question": "pergunta válida", "top_k": 99})
    assert response.status_code == 422


def test_search_retorna_apenas_o_retrieval(client):
    response = client.post("/search", json={"question": "post-mortem sem culpados"})
    assert response.status_code == 200
    results = response.json()["results"]
    assert results
    assert {"doc_id", "score", "text"} <= results[0].keys()


def test_erro_do_llm_vira_502_e_nao_500(client, pipeline, monkeypatch):
    def boom(question):
        raise RuntimeError("LLM indisponível após 4 tentativas")

    monkeypatch.setattr(pipeline, "answer", boom)
    response = client.post("/ask", json={"question": "pergunta qualquer"})
    assert response.status_code == 502
