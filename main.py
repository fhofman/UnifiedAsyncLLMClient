import asyncio
import keyManagement
from schemas import ChatMessage, LLMConfig
from AsyncLLMManager import AsyncLLMManager
from pydantic import SecretStr
import os

async def main():
    keyManagement.load_keys()
    config = LLMConfig(
        provider="openai",
        api_key=SecretStr(os.environ["OPENAI_API_KEY"]),
        model="gpt-4o-mini",
        temperature=0.7,
        max_tokens=200,
    )
    manager = AsyncLLMManager(config)
    pregunta = [ChatMessage(role="user", content="¿Qué es la entropía? Respondé en 2 líneas.")]
    response = await manager.generate(pregunta)
    print("🟢 OpenAI:", response.content if not response.error else f"❌ {response.error}")


if __name__ == "__main__":
    asyncio.run(main())
