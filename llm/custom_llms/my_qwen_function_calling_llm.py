from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core.llms import ChatMessage, ChatResponse, LLMMetadata
from llm.clients.qwen import QwenClient
from typing import List
from pydantic import PrivateAttr


class QwenFunctionCallingLLM(FunctionCallingLLM):
    _model_name: str = PrivateAttr()
    _qwen_client: "QwenClient" = PrivateAttr()  # ✅ 字符串注解，避免加载时报错

    def __init__(self, model_name: str, api_key: str = None, api_base: str = None):
        from llm.clients.qwen import QwenClient  # 🔧 强制本地作用域导入
        super().__init__()
        self._model_name = model_name
        self._qwen_client = QwenClient(model_name=model_name, api_key=api_key, api_base=api_base)


    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            is_function_calling_model=True,
            model_name=self._model_name,
        )

    def chat(self, messages: List[ChatMessage], **kwargs) -> ChatResponse:
        openai_messages = self._qwen_client._convert_to_openai_messages(messages)
        return self._qwen_client.chat(messages=openai_messages, **kwargs)


    def complete(self, prompt: str, **kwargs):
        raise NotImplementedError("complete() not supported")

    def stream_chat(self, messages: List[ChatMessage], **kwargs):
        raise NotImplementedError("stream_chat not supported")

    def stream_complete(self, prompt: str, **kwargs):
        raise NotImplementedError("stream_complete not supported")

    async def achat(self, messages: List[ChatMessage], **kwargs):
        raise NotImplementedError("achat not supported")

    async def acomplete(self, prompt: str, **kwargs):
        raise NotImplementedError("acomplete not supported")

    async def astream_chat(self, messages: List[ChatMessage], **kwargs):
        raise NotImplementedError("astream_chat not supported")

    async def astream_complete(self, prompt: str, **kwargs):
        raise NotImplementedError("astream_complete not supported")

    def _prepare_chat_with_tools(self, tools, messages, **kwargs):
        return {
            "messages": messages,
            "tools": tools,
            **kwargs
        }
