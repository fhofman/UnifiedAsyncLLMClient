# UnifiedAsyncLLMClient

Una capa asíncrona y unificada para hablar con distintos LLMs (**OpenAI**, **Anthropic** y **Google Gemini**) usando la misma interfaz.

Escribís tu código una sola vez contra `AsyncLLMManager` y cambiás de proveedor modificando solo la configuración.

```python
manager = AsyncLLMManager(config)          # config.provider = "openai" | "anthropic" | "gemini"
response = await manager.generate(messages)
```

## Características

- **Interfaz única** para los tres proveedores: `generate()` y `generate_stream()`.
- **100% async**, construido sobre los clientes asíncronos oficiales de cada SDK.
- **Streaming** de tokens mediante `async for`.
- **Esquemas tipados con Pydantic**: validación de roles, configuración y respuestas.
- **API keys protegidas** con `SecretStr`, así no aparecen en logs ni en `print`s.
- **Manejo de errores uniforme**: los errores no lanzan excepciones, se devuelven en `ModelResponse.error`.
- **Mensajes `system` normalizados**: se usa el rol `system` igual para todos y cada cliente lo traduce al formato de su proveedor (Anthropic y Gemini lo reciben en un parámetro aparte).

## Requisitos

- Python 3.12+
- Una API key del proveedor que quieras usar:
  - OpenAI: <https://platform.openai.com/api-keys>
  - Anthropic: <https://console.anthropic.com/>
  - Gemini (tiene capa gratuita): <https://aistudio.google.com/apikey>

## Instalación

```bash
git clone https://github.com/fhofman/UnifiedAsyncLLMClient.git
```

```bash
cd UnifiedAsyncLLMClient
```

```bash
python -m venv venv && source venv/bin/activate
```

```bash
pip install -r requirements
```

## Configuración de API keys

Las claves se leen de estas variables de entorno:

| Proveedor | Variable            |
|-----------|---------------------|
| OpenAI    | `OPENAI_API_KEY`    |
| Anthropic | `ANTHROPIC_API_KEY` |
| Gemini    | `GOOGLE_API_KEY`    |

Podés exportarlas antes de correr el programa:

```bash
export OPENAI_API_KEY="sk-..."
```

Si falta alguna, `keyManagement.load_keys()` te la pide por consola con `getpass`, sin mostrarla en pantalla.

## Uso rápido

```bash
python main.py
```

### Respuesta completa

```python
import asyncio
import os
from pydantic import SecretStr

from AsyncLLMManager import AsyncLLMManager
from schemas import ChatMessage, LLMConfig


async def main():
    config = LLMConfig(
        provider="anthropic",
        api_key=SecretStr(os.environ["ANTHROPIC_API_KEY"]),
        model="claude-haiku-4-5",
        temperature=0.7,
        max_tokens=300,
    )
    manager = AsyncLLMManager(config)

    messages = [
        ChatMessage(role="system", content="Sos un profesor de física conciso."),
        ChatMessage(role="user", content="¿Qué es la entropía? Respondé en 2 líneas."),
    ]

    response = await manager.generate(messages)
    if response.error:
        print("❌", response.error)
    else:
        print(response.content)


asyncio.run(main())
```

### Streaming

```python
async for chunk in manager.generate_stream(messages):
    print(chunk, end="", flush=True)
```

### Cambiar de proveedor

Solo cambian `provider`, `api_key` y `model`. El resto del código queda igual:

```python
LLMConfig(provider="openai",    api_key=SecretStr(os.environ["OPENAI_API_KEY"]),    model="gpt-4o-mini")
LLMConfig(provider="anthropic", api_key=SecretStr(os.environ["ANTHROPIC_API_KEY"]), model="claude-haiku-4-5")
LLMConfig(provider="gemini",    api_key=SecretStr(os.environ["GOOGLE_API_KEY"]),    model="gemini-2.5-flash")
```

### Consultar varios proveedores en paralelo

Como todo es async, podés consultar varios modelos a la vez con `asyncio.gather`:

```python
managers = [AsyncLLMManager(cfg) for cfg in (cfg_openai, cfg_anthropic, cfg_gemini)]
responses = await asyncio.gather(*(m.generate(messages) for m in managers))

for r in responses:
    print(f"[{r.provider.value}] {r.content or r.error}")
```

## Referencia de la API

### `LLMConfig`

| Campo         | Tipo        | Default | Descripción                                   |
|---------------|-------------|---------|-----------------------------------------------|
| `provider`    | `Provider`  | —       | `"openai"`, `"anthropic"` o `"gemini"`        |
| `api_key`     | `SecretStr` | —       | API key del proveedor                         |
| `model`       | `str`       | —       | Nombre del modelo (ej. `"gpt-4o-mini"`)       |
| `temperature` | `float`     | `0.7`   | Aleatoriedad de la respuesta                  |
| `max_tokens`  | `int`       | `200`   | Máximo de tokens a generar                    |

### `ChatMessage`

| Campo     | Tipo  | Descripción                              |
|-----------|-------|------------------------------------------|
| `role`    | `str` | `"user"`, `"assistant"` o `"system"`     |
| `content` | `str` | Texto del mensaje                        |

Un rol distinto de esos tres lanza un `ValidationError`.

### `ModelResponse`

| Campo      | Tipo            | Descripción                                   |
|------------|-----------------|-----------------------------------------------|
| `provider` | `Provider`      | Proveedor que respondió                       |
| `model`    | `str`           | Modelo usado                                  |
| `content`  | `str`           | Texto generado (vacío si hubo error)          |
| `error`    | `Optional[str]` | Mensaje de error, o `None` si salió todo bien |

### `AsyncLLMManager`

| Método                                  | Devuelve                    |
|-----------------------------------------|-----------------------------|
| `await generate(messages)`              | `ModelResponse`             |
| `generate_stream(messages)`             | `AsyncGenerator[str, None]` |
| `AsyncLLMManager.from_config(config)`   | El cliente concreto (`BaseLLMClient`) |

> **Nota sobre `temperature` en Anthropic:** los modelos Claude más nuevos (Opus 4.7 en adelante, Sonnet 5 y 5.5) rechazan `temperature` con un error 400. Usá un modelo que lo acepte, como `claude-haiku-4-5` o la línea 4.6.

## Manejo de errores

Los clientes **no propagan excepciones** de red ni de la API, así que una falla de un proveedor no te tira abajo el programa:

- En `generate()`, el error se devuelve en `ModelResponse.error` y `content` queda vacío.
- En `generate_stream()`, el error se emite como último chunk con el formato `[⚠️ Error durante el streaming: ...]`.

Se distinguen estos casos: límite de cuota (rate limit), error de conexión, error de la API y error desconocido.

Los errores de **configuración** (proveedor no soportado o `api_key` vacía) sí lanzan `ValueError` al crear el `AsyncLLMManager`, porque son errores del programador y conviene detectarlos temprano.

## Estructura del proyecto

```
.
├── main.py              # Ejemplo de uso
├── AsyncLLMManager.py   # Punto de entrada: elige y delega en el cliente correcto
├── BaseLLMClient.py     # Clase abstracta con la interfaz común
├── OpenAIClient.py      # Implementación para OpenAI
├── AnthropicClient.py   # Implementación para Anthropic
├── GeminiClient.py      # Implementación para Google Gemini
├── schemas.py           # Modelos Pydantic: LLMConfig, ChatMessage, ModelResponse
├── keyManagement.py     # Carga de API keys desde env o consola
└── requirements         # Dependencias
```

## Agregar un nuevo proveedor

1. Creá `MiProveedorClient.py` con una clase que herede de `BaseLLMClient` e implemente `generate()` y `generate_stream()`.
2. Agregá el valor al enum `Provider` en `schemas.py`.
3. Registrá la clase en el diccionario `clients` de `AsyncLLMManager.from_config`.

## Licencia

[MIT](LICENSE) © 2026 Federico Hofman
