from abc import ABC, abstractmethod
from typing import AsyncGenerator, List
from schemas import ChatMessage, ModelResponse

class BaseLLMClient(ABC):

    @abstractmethod
    async def generate(self, messages: List[ChatMessage])-> ModelResponse:
        raise NotImplementedError
    

    @abstractmethod
    async def generate_stream(self, messages: List[ChatMessage]) -> AsyncGenerator[str, None]:
        raise NotImplementedError






    
