class AsyncLLMManager:
    def __init__(self, config: LLMConfig):
        self.config = config
        self._clients = BaseLLMClient.from_config(config)

    @classmethod
    def from_config(cls, config: LLMConfig) -> 'BaseLLMClient':
        
        if config.provider == "openai":
            if not config.api_key:
                raise ValueError("OpenAI api_key is required")
            return OpenAIClient(api_key=self.config.api_key, model=config.model, temperature=config.temperature, max_tokens=config.max_tokens)
        elif config.provider == "anthropic":
            if not config.api_key:
                raise ValueError("Anthropic api_key is required")
            return AnthropicClient(api_key=self.config.api_key, model=config.model, temperature=config.temperature, max_tokens=config.max_tokens)
        elif config.provider == "gemini":
            if not config.api_key:
                raise ValueError("Gemini api_key is required")
            return GeminiClient(api_key=self.config.api_key, model=config.model, temperature=config.temperature, max_tokens=config.max_tokens)
        else:
            raise ValueError(f"Provider {config.provider} not supported")
        


    async def generate(self, messages: List[ChatMessage]) -> ModelResponse:
        return await self._clients.generate(messages)
    
    async def generate_stream(self, messages: List[ChatMessage]) -> AsyncGenerator[str, None]:
        async for chunk in self._client.generate_stream(messages):
            yield chunk if chunk else "\n"