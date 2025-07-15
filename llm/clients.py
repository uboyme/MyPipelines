from llm.base import BaseLLMClient
from typing import Union
from llama_index.llms.dashscope import DashScope
from llama_index.core.llms import ChatMessage, CompletionResponse

class BaiLianLLM(BaseLLMClient):
    def __init__(self, model: str, api_base: str, api_key: str, **kwargs):
        super().__init__(model)
        self.name = model
        self.history = []
        self.client = DashScope(
            model_name=model,
            api_key=api_key,
            **kwargs
        )

    def chat(self, prompt=None, messages=None, **kwargs) -> Union[str, CompletionResponse]:
        print("history:", self.history)
        if messages is not None:
            self.history = messages
        elif prompt is not None:
            self.history.append({"role": "user", "content": prompt})

        response = self.client.chat(self.history, **kwargs)
        reply = response.response if hasattr(response, "response") else str(response)
        self.history.append({"role": "assistant", "content": reply})
        return reply
