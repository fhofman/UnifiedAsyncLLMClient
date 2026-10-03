# schemas.py
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, SecretStr, field_validator

class Provider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"

class LLMConfig(BaseModel):
    provider: Provider
    api_key: SecretStr
    model: str
    temperature: float = 0.7
    max_tokens: int = 200

class ChatMessage(BaseModel):
    role: str = Field(description="'user', 'assistant' o 'system'")
    content: str

    @field_validator("role")
    @classmethod
    def rol_valido(cls, v: str) -> str:
        roles_permitidos = {"user", "assistant", "system"}
        if v not in roles_permitidos:
            raise ValueError(f"role debe ser uno de {roles_permitidos}, recibido: '{v}'")
        return v

class ModelResponse(BaseModel):
    provider: Provider
    model: str
    content: str
    error: Optional[str] = None
    
