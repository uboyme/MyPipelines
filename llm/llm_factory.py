#from llm.clients.qwen import QwenClient  # ✅ 用于 deepseek 和 qwen
from llm.custom_llms.my_qwen_function_calling_llm import QwenFunctionCallingLLM
from llm.clients.qwen import QwenClient

import os

# 你可以在 .env 或系统变量中配置这些值
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "sk-ada681a2aa6944e59040b800343a3d07")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com/v1")


class BaiLianLLMFactory:
    @staticmethod
    def create_llm_client(model_name: str):
        if model_name == "qwen-plus":
            return QwenClient(
                model_name="qwen-plus"
            )
        else:
            raise ValueError(f"Unsupported BaiLian model: {model_name}")


class LlamaIndexLLMFactory:
    @staticmethod
    def create_llm_client(model_name: str, model_alias: str = None, **kwargs):
        if model_name == "deepseek-chat":
            return QwenFunctionCallingLLM(
                model_name="deepseek-chat",
                api_key=DEEPSEEK_API_KEY,
                api_base=DEEPSEEK_BASE_URL
            )
        else:
            raise ValueError(f"Unsupported DeepSeek model: {model_name}")

