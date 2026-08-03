# RAG Docs API

API de perguntas e respostas sobre uma base de documentos, construída com FastAPI e Claude.
Recuperação híbrida (BM25 + vetorial), respostas com citação rastreável, recusa explícita
quando a base não contém a resposta, e uma suíte de avaliação automatizada que mede a
qualidade do sistema a cada mudança.

![CI](https://github.com/Bruno-GabrielDev/rag-docs-api/actions/workflows/ci.yml/badge.svg)

---

## O problema

Um chatbot ligado a documentos internos é fácil de montar e difícil de confiar. Os dois modos
de falha que aparecem em produção são sempre os mesmos:

1. **O sistema inventa.** Quando a pergunta não tem resposta na base, o modelo preenche a
   lacuna com algo plausível e o usuário não tem como saber.
2. **Ninguém sabe se piorou.** Mudou o prompt, mudou o `chunk_size`, trocou o modelo — a
   resposta "parece melhor". Sem medição, cada ajuste é aposta.

Este projeto foi construído em torno dessas duas questões, não em torno da chamada à API.

---

## Arquitetura

```
                      ┌──────────────┐
   pergunta  ────────▶│   FastAPI    │
                      └──────┬───────┘
                             ▼
                   ┌───────────────────┐
                   │  HybridRetriever  │
                   ├─────────┬─────────┤
                   │  BM25   │ Vetorial│   ambos rodam sobre o mesmo corpus
                   │(léxico) │(semânt.)│
                   └────┬────┴────┬────┘
                        └────┬────┘
                             ▼
                      Fusão por RRF  ──▶ top-k chunks
                             ▼
                   ┌───────────────────┐
                   │  Guardrail        │  score < limiar ──▶ "não encontrei"
                   └─────────┬─────────┘  (sem chamar o LLM)
                             ▼
                   ┌───────────────────┐
                   │  Prompt versionado│
                   │      + Claude     │
                   └─────────┬─────────┘
                             ▼
                 resposta + citações validadas
```

O domínio (`chunking`, `retriever`, `pipeline`) não importa nada de framework nem de
fornecedor. Embeddings e LLM entram por `Protocol`, o que permite trocar Voyage por OpenAI
por sentence-transformers via variável de ambiente — e, principalmente, rodar toda a suíte de
testes com dublês, sem rede e sem custo.

---

## Decisões técnicas

**Recuperação híbrida em vez de só vetorial.** Embedding dilui termos raros e exatos: siglas,
códigos, números de artigo, nomes próprios. Uma busca por "SEV2" pode não trazer o documento
que fala de SEV2, porque o vetor de "SEV2" é próximo do de "SEV1". BM25 acerta esse caso em
cheio e erra em paráfrase, que é justamente onde o vetorial acerta. Os dois cobrem os buracos
um do outro.

**RRF em vez de soma ponderada de scores.** Similaridade de cosseno vive entre 0 e 1; BM25 não
tem limite superior. Somar os dois exige calibrar um peso que muda a cada corpus. Reciprocal
Rank Fusion usa só a *posição* em cada ranking, então funciona sem calibração.

**BM25 e o índice vetorial implementados na mão.** Nesta escala (milhares de chunks) a busca
exaustiva com numpy é instantânea, e o código deixa visível o que uma biblioteca faria por
baixo. As interfaces são pequenas o bastante para trocar por pgvector ou Qdrant sem tocar no
pipeline. É uma escolha de projeto de portfólio: em produção com milhões de vetores, o índice
aproximado (HNSW) seria obrigatório.

**Guardrail antes do modelo, não depois.** Se o melhor chunk recuperado fica abaixo do limiar
de similaridade, o pipeline responde "não encontrei" sem chamar o LLM. Isso ataca a alucinação
na origem — o modelo não pode inventar sobre um contexto que nunca recebeu — e ainda economiza
tokens em toda pergunta fora de escopo.

**Citações validadas contra os trechos.** O modelo escreve marcadores `[1]`, `[2]`; o pipeline
os converte em referências com `doc_id` e `chunk_id` reais e **descarta marcadores fora do
intervalo**. Modelo citando `[5]` quando só existem 4 trechos é comum, e uma citação inválida é
pior que citação nenhuma, porque parece verificada.

**Chunking recursivo com overlap.** A quebra tenta primeiro o separador mais semântico
(parágrafo) e só desce para linha, frase e espaço quando o bloco continua grande demais. O
overlap existe para que uma informação que atravessa a fronteira de dois chunks continue
recuperável por pelo menos um deles.

**Prompt em módulo versionado.** `PROMPT_VERSION` entra no relatório de avaliação, então dá
para atribuir uma variação de qualidade a uma mudança específica de texto.

---

## Avaliação

O que diferencia este projeto de um notebook que chama uma API é `evaluation/`. O dataset
dourado (`golden.jsonl`) tem 18 perguntas em três categorias: diretas, parafraseadas
(pergunta que não repete o vocabulário do documento) e **não respondíveis** — estas últimas
existem só para verificar se o sistema recusa em vez de inventar.

As duas etapas são medidas em separado, de propósito. Quando a resposta final piora, a primeira
pergunta é sempre "o retrieval trouxe o trecho certo?"; sem isolar as etapas não dá para saber
onde mexer.

| Camada | Métrica | O que detecta |
| --- | --- | --- |
| Retrieval | `hit@k`, `MRR`, `recall@k` | o trecho certo foi recuperado, e em que posição |
| Geração | `faithfulness` (1–5, LLM-as-judge) | alucinação: a resposta é sustentada pelo contexto |
| Geração | `relevance` (1–5, LLM-as-judge) | a resposta de fato responde à pergunta |
| Guardrail | `abstention_accuracy` | recusou corretamente o que não estava na base |
| Guardrail | `false_abstention_rate` | recusou demais, virando inútil |

As duas últimas linhas formam um par em tensão: é trivial zerar a alucinação recusando tudo.
Medir os dois lados obriga o limiar a ser escolhido, não chutado.

```bash
make eval-fast   # só métricas de retrieval — sem custo de API
make eval        # avaliação completa, com LLM-as-judge
```

O relatório sai em `evaluation/reports/latest.md` e em JSON com carimbo de data, incluindo a
lista das perguntas que falharam no retrieval — que é por onde se começa a depurar.

> **Nota sobre LLM-as-judge:** o método tem viés conhecido (tende a premiar respostas longas e
> a concordar com o próprio estilo). As notas servem para comparar versões entre si ao longo do
> tempo, não como verdade absoluta.

### Resultados

Rode `make eval` e cole aqui a tabela gerada, junto com a configuração usada
(`top_k`, `chunk_size`, provedor de embedding). Comparar duas ou três configurações
lado a lado mostra o processo, não só o resultado.

---

## Como rodar

```bash
# 1. Dependências
python -m venv .venv && source .venv/bin/activate
make install

# 2. Configuração
cp .env.example .env      # e preencha ANTHROPIC_API_KEY

# 3. Indexar os documentos de data/docs (.md, .txt, .pdf)
make ingest

# 4. Subir a API
make run                  # http://localhost:8000/docs
```

Com Docker:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
docker compose up --build
```

### Exemplo

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Quantos revisores são necessários para aprovar um PR?"}'
```

```json
{
  "question": "Quantos revisores são necessários para aprovar um PR?",
  "answer": "São necessários pelo menos dois revisores, sendo um deles obrigatoriamente membro do time responsável pelo módulo alterado [1].",
  "citations": [
    {
      "marker": 1,
      "doc_id": "politica-engenharia.md",
      "chunk_id": "politica-engenharia.md#0#a3f9c1e8b204",
      "snippet": "Todo pull request precisa de aprovação de pelo menos dois revisores..."
    }
  ],
  "grounded": true,
  "latency_ms": 1240,
  "usage": { "input_tokens": 812, "output_tokens": 47, "model": "claude-sonnet-5" }
}
```

Perguntando algo fora da base:

```json
{
  "answer": "Não encontrei essa informação nos documentos fornecidos.",
  "citations": [],
  "grounded": false
}
```

### Endpoints

| Método | Rota | Descrição |
| --- | --- | --- |
| `POST` | `/ask` | Pergunta e resposta com citações |
| `POST` | `/search` | Só o retrieval, sem gerar resposta — para depurar o índice |
| `GET` | `/health` | Status e número de chunks indexados |

---

## Testes

```bash
make test     # 55 testes
make cov      # relatório de cobertura
```

A suíte inteira roda **sem rede, sem chave de API e sem baixar modelo**: os embeddings são
substituídos por um hashing determinístico e o LLM por um dublê que registra as chamadas
recebidas. Teste que depende de resposta de LLM é teste instável, e teste instável logo vira
teste ignorado.

Cobertura por camada: o núcleo de domínio (`chunking`, `pipeline`, `prompts`, `models`) está em
100% e o `retriever` em 97%; os adaptadores de infraestrutura (clientes HTTP, ingestão) ficam
mais baixos por design — testá-los exigiria mockar SDK de terceiro, o que testa o mock e não o
sistema.

Os casos foram escolhidos por análise de valor limite e partição de equivalência: texto menor
que o chunk, texto exatamente no limite, texto sem separador algum, `k` maior que o corpus,
`overlap` inválido, citação fora do intervalo, corpus vazio.

---

## Limitações conhecidas

- **Busca exaustiva.** Latência do retrieval cresce linearmente com o corpus. Acima de ~100 mil
  chunks é preciso índice aproximado (HNSW via Qdrant/pgvector).
- **Índice em memória, reconstruído por inteiro.** Não há atualização incremental por documento.
- **Sem reranking.** Um cross-encoder sobre os candidatos do RRF tipicamente ganha alguns pontos
  de precisão, ao custo de latência.
- **Chunking cego a estrutura.** Cabeçalhos Markdown poderiam virar metadado e permitir filtro
  por seção.
- **Sem cache de perguntas repetidas** nem controle de custo por usuário.
- **PDF via extração de texto simples.** Documentos escaneados ou com tabelas complexas exigem
  OCR e parsing de layout.

## Próximos passos

- [ ] Reranking com cross-encoder sobre os candidatos
- [ ] Streaming de resposta (SSE) na rota `/ask`
- [ ] Ingestão incremental com hash por documento
- [ ] Rodar a avaliação no CI e falhar o build quando as métricas caírem
- [ ] Migrar o índice para pgvector

---

## Estrutura

```
src/rag/          domínio: chunking, embeddings, store, retriever, prompts, llm, pipeline
src/api/          camada HTTP (FastAPI)
evaluation/       dataset dourado, métricas, LLM-as-judge, runner de relatórios
tests/            55 testes, sem rede e sem custo
data/docs/        documentos de exemplo (fictícios)
```

## Stack

Python 3.12 · FastAPI · Pydantic v2 · NumPy · Anthropic SDK (Claude) · sentence-transformers ·
pytest · Ruff · Docker · GitHub Actions

---

Os documentos em `data/docs/` são fictícios, criados apenas como base de testes.
