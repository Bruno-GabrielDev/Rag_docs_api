"""LLM-as-judge para avaliar a geração.

Duas dimensões, avaliadas separadamente porque falham de formas diferentes:

- faithfulness (fidelidade): a resposta é sustentada pelos trechos? É o que
  detecta alucinação. Depende só da resposta + contexto.
- relevance (relevância): a resposta de fato responde à pergunta? Uma resposta
  pode ser 100% fiel ao contexto e ainda assim não responder nada.

O juiz recebe uma escala com âncoras descritas e devolve JSON, para que a nota
seja reprodutível em vez de "vibe". Nota: usar LLM como juiz tem viés conhecido
(tende a premiar respostas longas e concordar com o próprio estilo), então as
notas servem para comparar versões entre si, não como verdade absoluta.
"""

from __future__ import annotations

import json
import re

from src.rag.llm import LLMClient

JUDGE_SYSTEM = """Você é um avaliador rigoroso de sistemas de perguntas e respostas.
Responda SEMPRE e SOMENTE com um objeto JSON válido, sem markdown, sem comentários."""

FAITHFULNESS_TEMPLATE = """Avalie se a RESPOSTA é inteiramente sustentada pelos TRECHOS.

Escala:
1 = contém afirmações que contradizem ou não existem nos trechos (alucinação)
3 = majoritariamente sustentada, mas com algum detalhe não verificável
5 = toda afirmação é rastreável aos trechos

<trechos>
{context}
</trechos>

<resposta>
{answer}
</resposta>

JSON: {{"score": <1-5>, "reason": "<uma frase>"}}"""

RELEVANCE_TEMPLATE = """Avalie se a RESPOSTA responde à PERGUNTA, comparando com a RESPOSTA ESPERADA.

Escala:
1 = não responde ou responde outra coisa
3 = responde parcialmente ou com informação faltando
5 = responde completa e diretamente

<pergunta>
{question}
</pergunta>

<resposta_esperada>
{expected}
</resposta_esperada>

<resposta_obtida>
{answer}
</resposta_obtida>

JSON: {{"score": <1-5>, "reason": "<uma frase>"}}"""


def _parse_json(text: str) -> dict:
    """Extrai JSON mesmo se o modelo envolver em cercas de markdown."""
    cleaned = re.sub(r"```(?:json)?|```", "", text).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
    return {"score": 0, "reason": "resposta do juiz não pôde ser interpretada"}


class Judge:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def _score(self, prompt: str) -> dict:
        response = self.llm.complete(
            system=JUDGE_SYSTEM, user=prompt, max_tokens=256, temperature=0.0
        )
        result = _parse_json(response.text)
        result["score"] = float(result.get("score", 0))
        return result

    def faithfulness(self, context: str, answer: str) -> dict:
        return self._score(FAITHFULNESS_TEMPLATE.format(context=context, answer=answer))

    def relevance(self, question: str, expected: str, answer: str) -> dict:
        return self._score(
            RELEVANCE_TEMPLATE.format(question=question, expected=expected, answer=answer)
        )
