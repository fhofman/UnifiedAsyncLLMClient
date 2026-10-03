from anthropic import (
    AsyncAnthropic,
    APIError as AnthropicAPIError,
    RateLimitError as AnthropicRateLimitError,
    APIConnectionError as AnthropicConnectionError,
)
from BaseLLMClient import BaseLLMClient
from schemas import ChatMessage, ModelResponse, Provider
from typing import List, AsyncGenerator


class AnthropicClient(BaseLLMClient):
    def __init__(self, api_key: str, model: str, temperature: float, max_tokens: int):
        self._client = AsyncAnthropic(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def _build_kwargs(self, messages: List[ChatMessage]) -> dict:
        """Anthropic no acepta el rol 'system' dentro de messages: va en un parámetro aparte.

        El SDK 1.x ya no acepta `temperature` como argumento, así que se manda por extra_body.
        Ojo: Opus 4.7+ y Sonnet 5/5.5 rechazan temperature; Haiku 4.5 y la línea 4.6 lo aceptan.
        """
        kwargs = dict(
            model=self.model,
            max_tokens=self.max_tokens,
            extra_body={"temperature": self.temperature},
            messages=[m.model_dump() for m in messages if m.role != "system"],
        )
        system = "\n\n".join(m.content for m in messages if m.role == "system")
        if system:
            kwargs["system"] = system
        return kwargs

    async def generate(self, messages: List[ChatMessage]) -> ModelResponse:
        try:
            response = await self._client.messages.create(**self._build_kwargs(messages))
            return ModelResponse(
                provider=Provider.ANTHROPIC,
                model=self.model,
                content="".join(b.text for b in response.content if b.type == "text"),
            )
        except AnthropicRateLimitError as e:
            return ModelResponse(provider=Provider.ANTHROPIC, model=self.model, content="",
                                  error=f"Límite de cuota excedido: {e}")
        except AnthropicConnectionError as e:
            return ModelResponse(provider=Provider.ANTHROPIC, model=self.model, content="",
                                  error=f"Error de conexión: {e}")
        except AnthropicAPIError as e:
            return ModelResponse(provider=Provider.ANTHROPIC, model=self.model, content="",
                                  error=f"Error de la API de Anthropic: {e}")
        except Exception as e:
            return ModelResponse(provider=Provider.ANTHROPIC, model=self.model, content="",
                                  error=f"Error desconocido: {e}")

    async def generate_stream(self, messages: List[ChatMessage]) -> AsyncGenerator[str, None]:
        try:
            async with self._client.messages.stream(**self._build_kwargs(messages)) as stream:
                async for texto in stream.text_stream:
                    yield texto
        except (AnthropicRateLimitError, AnthropicConnectionError, AnthropicAPIError) as e:
            yield f"\n[⚠️ Error durante el streaming: {e}]"
        except Exception as e:
            yield f"\n[⚠️ Error desconocido: {e}]"
