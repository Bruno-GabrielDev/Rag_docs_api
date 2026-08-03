"""Cliente de LLM.

Abstraído atrás de um Protocol por dois motivos: trocar de fornecedor sem
reescrever o pipeline, e — mais importante — permitir que a suíte de testes
rode com um dublê determinístico, sem rede, sem chave e sem custo.
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass, field
from typing import Protocol

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    text: str
    usage: dict = field(default_factory=dict)


class LLMClient(Protocol):
    def complete(self, system: str, user: str, max_tokens: int, temperature: float) -> LLMResponse: ...


class AnthropicLLM:
    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-5",
        max_retries: int = 4,
        base_delay: float = 1.0,
    ) -> None:
        import anthropic

        self._anthropic = anthropic
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model
        self._max_retries = max_retries
        self._base_delay = base_delay

    def complete(
        self, system: str, user: str, max_tokens: int = 1024, temperature: float = 0.0
    ) -> LLMResponse:
        last_error: Exception | None = None

        for attempt in range(self._max_retries):
            try:
                message = self._client.messages.create(
                    model=self._model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    system=system,
                    messages=[{"role": "user", "content": user}],
                )
                text = "".join(
                    block.text for block in message.content if block.type == "text"
                )
                return LLMResponse(
                    text=text,
                    usage={
                        "input_tokens": message.usage.input_tokens,
                        "output_tokens": message.usage.output_tokens,
                        "model": self._model,
                    },
                )
            except (self._anthropic.RateLimitError, self._anthropic.APIStatusError) as exc:
                status = getattr(exc, "status_code", None)
                # 4xx que não seja 429 é erro do nosso lado: repetir não resolve.
                if status is not None and 400 <= status < 500 and status != 429:
                    raise
                last_error = exc
            except self._anthropic.APIConnectionError as exc:
                last_error = exc

            # Backoff exponencial com jitter, para não sincronizar retries.
            delay = self._base_delay * (2**attempt) + random.uniform(0, 0.5)
            logger.warning(
                "Falha na chamada ao LLM (tentativa %d/%d). Repetindo em %.1fs",
                attempt + 1,
                self._max_retries,
                delay,
            )
            time.sleep(delay)

        raise RuntimeError(f"LLM indisponível após {self._max_retries} tentativas") from last_error


class FakeLLM:
    """Dublê determinístico para testes: ecoa o contexto recebido."""

    def __init__(self, response: str = "Resposta de teste [1]") -> None:
        self.response = response
        self.calls: list[dict] = []

    def complete(
        self, system: str, user: str, max_tokens: int = 1024, temperature: float = 0.0
    ) -> LLMResponse:
        self.calls.append({"system": system, "user": user})
        return LLMResponse(
            text=self.response,
            usage={"input_tokens": 0, "output_tokens": 0, "model": "fake"},
        )
