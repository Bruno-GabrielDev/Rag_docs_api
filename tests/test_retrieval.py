"""Testes de recuperação: store vetorial, BM25, fusão RRF e retriever híbrido."""

from __future__ import annotations

import numpy as np
import pytest

from src.rag.chunking import chunk_document
from src.rag.retriever import BM25, reciprocal_rank_fusion, tokenize
from src.rag.store import VectorStore

# --- Vector store ---


def test_store_vazio_retorna_lista_vazia(embeddings):
    store = VectorStore(dimension=embeddings.dimension)
    assert store.search(embeddings.embed_query("qualquer coisa"), k=3) == []


def test_search_retorna_resultados_ordenados_por_similaridade(store, embeddings):
    results = store.search(embeddings.embed_query("pull request revisores"), k=3)
    scores = [score for _, score in results]
    assert scores == sorted(scores, reverse=True)


def test_k_maior_que_o_corpus_nao_estoura(store, embeddings):
    results = store.search(embeddings.embed_query("teste"), k=999)
    assert len(results) == len(store)


def test_dimensao_incompativel_e_rejeitada(embeddings):
    store = VectorStore(dimension=8)
    chunks = chunk_document("d.md", "texto qualquer", chunk_size=50, chunk_overlap=0)
    with pytest.raises(ValueError):
        store.add(chunks, np.zeros((len(chunks), 16), dtype=np.float32))


def test_indice_sobrevive_ao_ciclo_de_salvar_e_carregar(store, tmp_path, embeddings):
    store.save(tmp_path)
    reloaded = VectorStore.load(tmp_path)
    assert len(reloaded) == len(store)
    assert reloaded.chunks[0].id == store.chunks[0].id
    np.testing.assert_allclose(reloaded.vectors, store.vectors)


# --- BM25 ---


def test_tokenize_remove_stopwords_e_normaliza_caixa():
    assert tokenize("O Deploy DE produção") == ["deploy", "produção"]


def test_bm25_prioriza_o_documento_que_contem_o_termo():
    corpus = [
        "o deploy em produção acontece toda sexta",
        "receita de bolo de cenoura com cobertura",
        "política de plantão e escalonamento",
    ]
    bm25 = BM25(corpus)
    top_index, score = bm25.search("deploy produção", k=1)[0]
    assert top_index == 0
    assert score > 0


def test_bm25_nao_retorna_documentos_com_score_zero():
    bm25 = BM25(["texto sobre gatos", "texto sobre cachorros"])
    assert bm25.search("astrofísica quântica", k=5) == []


def test_bm25_com_corpus_vazio():
    assert BM25([]).search("qualquer", k=3) == []


# --- RRF ---


def test_rrf_favorece_documento_bem_colocado_nas_duas_listas():
    fused = reciprocal_rank_fusion([[1, 2, 3], [3, 1, 2]], k=60)
    assert fused[0][0] == 1


def test_rrf_mantem_documento_presente_em_uma_unica_lista():
    fused = reciprocal_rank_fusion([[7], [8]], k=60)
    assert {doc_id for doc_id, _ in fused} == {7, 8}


def test_rrf_com_listas_vazias():
    assert reciprocal_rank_fusion([[], []]) == []


# --- Retriever híbrido ---


def test_retriever_encontra_o_documento_certo(retriever):
    results = retriever.retrieve("quantos revisores aprovam um pull request", top_k=2)
    assert results
    assert results[0].chunk.doc_id == "politica.md"


def test_retriever_respeita_o_top_k(retriever):
    assert len(retriever.retrieve("post-mortem plantão", top_k=2)) == 2


def test_retriever_retorna_resultados_ordenados(retriever):
    results = retriever.retrieve("incidentes SEV1 resposta", top_k=3)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)
