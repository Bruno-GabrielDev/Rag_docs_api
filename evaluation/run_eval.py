"""Executa a suíte de avaliação sobre o dataset dourado e gera um relatório.

Uso:
    python -m evaluation.run_eval
    python -m evaluation.run_eval --no-judge      # só métricas de retrieval (sem custo)
    python -m evaluation.run_eval --top-k 6       # comparar configurações
"""

from __future__ import annotations

import argparse
import json
import statistics
from datetime import UTC, datetime
from pathlib import Path

from evaluation.judge import Judge
from evaluation.metrics import aggregate, hit_at_k, recall_at_k, reciprocal_rank
from src.rag.config import get_settings
from src.rag.embeddings import build_embedding_provider
from src.rag.llm import AnthropicLLM
from src.rag.pipeline import NO_ANSWER, RAGPipeline
from src.rag.prompts import PROMPT_VERSION, format_context
from src.rag.retriever import HybridRetriever
from src.rag.store import VectorStore

GOLDEN_PATH = Path(__file__).parent / "golden.jsonl"
REPORT_DIR = Path(__file__).parent / "reports"


def load_golden(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_report(results: list[dict], top_k: int, judged: bool) -> dict:
    answerable = [r for r in results if not r["unanswerable"]]
    unanswerable = [r for r in results if r["unanswerable"]]

    report = {
        "timestamp": datetime.now(UTC).isoformat(),
        "prompt_version": PROMPT_VERSION,
        "top_k": top_k,
        "n_questions": len(results),
        "retrieval": {
            f"hit@{top_k}": round(aggregate([r["hit"] for r in answerable]), 3),
            "mrr": round(aggregate([r["mrr"] for r in answerable]), 3),
            f"recall@{top_k}": round(aggregate([r["recall"] for r in answerable]), 3),
        },
        "latency_ms": {
            "p50": round(statistics.median([r["latency_ms"] for r in results]), 1),
            "max": max(r["latency_ms"] for r in results),
        },
        "guardrail": {
            # Das perguntas sem resposta na base, quantas o sistema recusou
            # corretamente em vez de inventar? É a métrica anti-alucinação.
            "abstention_accuracy": round(
                aggregate([float(r["abstained"]) for r in unanswerable]), 3
            )
            if unanswerable
            else None,
            "false_abstention_rate": round(
                aggregate([float(r["abstained"]) for r in answerable]), 3
            ),
        },
    }

    if judged:
        report["generation"] = {
            "faithfulness": round(
                aggregate([r["faithfulness"] for r in answerable if r["faithfulness"]]), 2
            ),
            "relevance": round(
                aggregate([r["relevance"] for r in answerable if r["relevance"]]), 2
            ),
        }
    return report


def to_markdown(report: dict, results: list[dict]) -> str:
    lines = [
        "# Relatório de avaliação — RAG Docs API",
        "",
        f"- Data: {report['timestamp']}",
        f"- Versão do prompt: `{report['prompt_version']}`",
        f"- top_k: {report['top_k']} · perguntas: {report['n_questions']}",
        "",
        "## Retrieval",
        "",
        "| Métrica | Valor |",
        "| --- | --- |",
    ]
    for key, value in report["retrieval"].items():
        lines.append(f"| {key} | {value} |")

    if "generation" in report:
        lines += ["", "## Geração (LLM-as-judge, escala 1-5)", "", "| Métrica | Valor |", "| --- | --- |"]
        for key, value in report["generation"].items():
            lines.append(f"| {key} | {value} |")

    lines += [
        "",
        "## Guardrail",
        "",
        f"- Abstenção correta em perguntas sem resposta: {report['guardrail']['abstention_accuracy']}",
        f"- Abstenção indevida em perguntas respondíveis: {report['guardrail']['false_abstention_rate']}",
        "",
        "## Latência",
        "",
        f"- p50: {report['latency_ms']['p50']} ms · máx: {report['latency_ms']['max']} ms",
        "",
        "## Falhas",
        "",
    ]

    failures = [r for r in results if not r["unanswerable"] and r["hit"] == 0.0]
    if not failures:
        lines.append("Nenhuma falha de retrieval.")
    for failure in failures:
        lines.append(f"- **{failure['question']}** — esperado `{failure['expected_docs']}`, veio `{failure['retrieved_docs']}`")

    return "\n".join(lines) + "\n"


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Avalia o pipeline RAG")
    parser.add_argument("--top-k", type=int, default=settings.top_k)
    parser.add_argument("--no-judge", action="store_true", help="Pula o LLM-as-judge")
    parser.add_argument("--golden", type=Path, default=GOLDEN_PATH)
    args = parser.parse_args()

    dataset = load_golden(args.golden)
    store = VectorStore.load(settings.index_dir)
    embeddings = build_embedding_provider(settings)
    retriever = HybridRetriever(store, embeddings, rrf_k=settings.rrf_k)
    llm = AnthropicLLM(api_key=settings.anthropic_api_key, model=settings.llm_model)
    pipeline = RAGPipeline(
        retriever=retriever,
        llm=llm,
        top_k=args.top_k,
        candidate_k=settings.candidate_k,
        min_score=settings.min_score,
    )
    judge = None if args.no_judge else Judge(llm)

    results: list[dict] = []
    for i, item in enumerate(dataset, start=1):
        question = item["question"]
        expected_docs = item.get("expected_docs", [])
        unanswerable = bool(item.get("unanswerable", False))

        retrieved = retriever.retrieve(
            question, top_k=args.top_k, candidate_k=settings.candidate_k
        )
        retrieved_docs = [r.chunk.doc_id for r in retrieved]
        answer = pipeline.answer(question)

        row = {
            "question": question,
            "expected_docs": expected_docs,
            "retrieved_docs": retrieved_docs,
            "unanswerable": unanswerable,
            "hit": hit_at_k(retrieved_docs, expected_docs, args.top_k),
            "mrr": reciprocal_rank(retrieved_docs, expected_docs),
            "recall": recall_at_k(retrieved_docs, expected_docs, args.top_k),
            "abstained": NO_ANSWER in answer.answer,
            "answer": answer.answer,
            "n_citations": len(answer.citations),
            "latency_ms": answer.latency_ms,
            "faithfulness": None,
            "relevance": None,
        }

        if judge and not unanswerable and not row["abstained"]:
            context = format_context(retrieved)
            row["faithfulness"] = judge.faithfulness(context, answer.answer)["score"]
            row["relevance"] = judge.relevance(
                question, item.get("expected_answer", ""), answer.answer
            )["score"]

        results.append(row)
        print(f"[{i}/{len(dataset)}] {question[:60]}... hit={row['hit']:.0f}")

    report = build_report(results, args.top_k, judged=judge is not None)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    (REPORT_DIR / f"eval-{stamp}.json").write_text(
        json.dumps({"report": report, "results": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (REPORT_DIR / "latest.md").write_text(to_markdown(report, results), encoding="utf-8")

    print("\n" + json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nRelatório salvo em {REPORT_DIR}/latest.md")


if __name__ == "__main__":
    main()
