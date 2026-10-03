import asyncio
import keyManagement
from schemas import ChatMessage, LLMConfig
from AsyncLLMManager import AsyncLLMManager
from pydantic import SecretStr
import os

PREGUNTA = [ChatMessage(role="user", content="¿Qué es la entropía? Respondé en 2 líneas.")]


def crear_configs() -> list[LLMConfig]:
    return [
        LLMConfig(
            provider="openai",
            api_key=SecretStr(os.environ["OPENAI_API_KEY"]),
            model="gpt-4o-mini",
            temperature=0.7,
            max_tokens=200,
        ),
        LLMConfig(
            provider="anthropic",
            api_key=SecretStr(os.environ["ANTHROPIC_API_KEY"]),
            model="claude-haiku-4-5",
            temperature=0.7,
            max_tokens=200,
        ),
        LLMConfig(
            provider="gemini",
            api_key=SecretStr(os.environ["GOOGLE_API_KEY"]),
            model="gemini-3.5-flash-lite",
            temperature=0.7,
            max_tokens=200,
        ),
    ]


async def main():
    keyManagement.load_keys()
    managers = [AsyncLLMManager(config) for config in crear_configs()]

    # Respuesta completa: los tres proveedores en paralelo
    print("=== generate() ===")
    responses = await asyncio.gather(*(m.generate(PREGUNTA) for m in managers))
    for response in responses:
        print(f"🟢 {response.provider.value} ({response.model}):",
              response.content if not response.error else f"❌ {response.error}")

    # Streaming: uno después del otro para que no se mezcle la salida
    print("\n=== generate_stream() ===")
    for manager in managers:
        print(f"🟢 {manager.config.provider.value} ({manager.config.model}): ", end="", flush=True)
        async for chunk in manager.generate_stream(PREGUNTA):
            print(chunk, end="", flush=True)
        print()


if __name__ == "__main__":
    asyncio.run(main())
