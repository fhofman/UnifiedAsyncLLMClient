from BaseLLMClient import BaseLLMClient
from typing import List, AsyncGenerator
from schemas import ChatMessage, ModelResponse, LLMConfig, Provider
from OpenAIClient import OpenAIClient
from AnthropicClient import AnthropicClient
from GeminiClient import GeminiClient


class AsyncLLMManager:
    def __init__(self, config: LLMConfig):
        self.config = config
        self._clients = self.from_config(config)

    @staticmethod
    def from_config(config: LLMConfig) -> BaseLLMClient:
        clients = {
            Provider.OPENAI: OpenAIClient,
            Provider.ANTHROPIC: AnthropicClient,
            Provider.GEMINI: GeminiClient,
        }
        client_cls = clients.get(config.provider)
        if client_cls is None:
            raise ValueError(f"Provider {config.provider} not supported")

        api_key = config.api_key.get_secret_value()
        if not api_key:
            raise ValueError(f"{config.provider.value} api_key is required")

        return client_cls(api_key=api_key, model=config.model, temperature=config.temperature, max_tokens=config.max_tokens)

    async def generate(self, messages: List[ChatMessage]) -> ModelResponse:
        return await self._clients.generate(messages)

    async def generate_stream(self, messages: List[ChatMessage]) -> AsyncGenerator[str, None]:
        async for chunk in self._clients.generate_stream(messages):
            yield chunk
