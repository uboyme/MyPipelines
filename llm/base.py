from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any


class BaseLLMClient(ABC):
    def __init__(self, model: str):
        self.model = model

    @abstractmethod
    def chat(self, messages: list[dict] | Any, **kwargs) -> Any:
        pass


class LLMFactory(ABC):
    @abstractmethod
    def create_llm_client(self, model: str, platform: str = None, **kwargs) -> "BaseLLMClient":
        pass
