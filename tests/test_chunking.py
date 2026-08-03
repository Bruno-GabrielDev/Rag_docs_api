"""Testes da estratégia de chunking.

Casos escolhidos por análise de valor limite: texto menor que o chunk, texto
exatamente no limite, texto sem separador nenhum e overlap inválido.
"""

from __future__ import annotations

import pytest

from src.rag.chunking import chunk_document, normalize


def test_texto_menor_que_o_chunk_gera_um_unico_chunk():
    chunks = chunk_document("doc.md", "Texto curto.", chunk_size=100, chunk_overlap=10)
    assert len(chunks) == 1
    assert chunks[0].text == "Texto curto."


def test_texto_longo_e_dividido_em_varios_chunks():
    text = "\n\n".join(f"Parágrafo número {i} com algum conteúdo textual." for i in range(20))
    chunks = chunk_document("doc.md", text, chunk_size=120, chunk_overlap=0)
    assert len(chunks) > 1
    assert all(len(c.text) <= 120 for c in chunks)


def test_overlap_repete_o_final_do_chunk_anterior():
    text = "\n\n".join(f"Bloco {i} " + "x" * 60 for i in range(4))
    chunks = chunk_document("doc.md", text, chunk_size=100, chunk_overlap=20)
    tail = chunks[0].text[-15:]
    assert tail in chunks[1].text


def test_chunks_recebem_id_unico_e_posicao_sequencial():
    text = "\n\n".join(f"Parágrafo {i}." * 10 for i in range(6))
    chunks = chunk_document("doc.md", text, chunk_size=80, chunk_overlap=10)
    assert [c.position for c in chunks] == list(range(len(chunks)))
    assert len({c.id for c in chunks}) == len(chunks)


def test_texto_sem_separador_algum_e_cortado_sem_estourar_o_limite():
    chunks = chunk_document("doc.md", "a" * 500, chunk_size=100, chunk_overlap=0)
    assert all(len(c.text) <= 100 for c in chunks)
    assert "".join(c.text for c in chunks) == "a" * 500


def test_overlap_maior_que_chunk_size_e_rejeitado():
    with pytest.raises(ValueError):
        chunk_document("doc.md", "texto", chunk_size=50, chunk_overlap=50)


def test_texto_vazio_nao_gera_chunks():
    assert chunk_document("doc.md", "   \n\n  ", chunk_size=100, chunk_overlap=0) == []


def test_normalize_preserva_paragrafos_e_colapsa_espacos():
    assert normalize("a  \t b\n\n\n\nc") == "a b\n\nc"


def test_metadata_e_propagada_para_todos_os_chunks():
    chunks = chunk_document(
        "doc.md", "um\n\ndois\n\ntrês", chunk_size=5, chunk_overlap=0, metadata={"source": "x"}
    )
    assert all(c.metadata == {"source": "x"} for c in chunks)
