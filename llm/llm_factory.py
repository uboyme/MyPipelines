from llm.base import LLMFactory, BaseLLMClient
from llm.clients import BaiLianLLM
from llama_index.llms.deepseek import DeepSeek

DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DASHSCOPE_API_KEY = "sk-0067532e096c46168dbede4b65a44d6a"

DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
DEEPSEEK_API_KEY = "sk-ada681a2aa6944e59040b800343a3d07"  # ✅ 替换成测试成功的

class BaiLianLLMFactory(LLMFactory):
    def create_llm_client(self, model, platform=None, **kwargs) -> BaseLLMClient:
        return BaiLianLLM(model, api_base=DASHSCOPE_BASE_URL, api_key=DASHSCOPE_API_KEY, **kwargs)

class LlamaIndexLLMFactory(LLMFactory):
    def create_llm_client(self, model, platform=None, **kwargs) -> BaseLLMClient:
        print("[DEBUG] 使用模型名：", model)
        return DeepSeek(model, api_base=DEEPSEEK_BASE_URL, api_key=DEEPSEEK_API_KEY, openai_llm_kwargs=kwargs)

