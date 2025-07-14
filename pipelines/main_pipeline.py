"""
title: My SysML Gen Pipeline
author: wumbuk
date: 2025-07-07
version: 1.0
license: MIT
description: A pipeline that integrates with the Llama Index library to generate SysML models based on user input.
"""
try:
    from llm.clients.qwen import QwenClient
    print("✅ 成功导入 QwenClient")
except Exception as e:
    print("❌ 无法导入 QwenClient:", e)
import llm.clients.qwen as qwen_module
print("✅ QwenClient 实际来源于：", qwen_module.__file__)

import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# from sentence_transformers import SentenceTransformer
from typing import Optional
from typing import List, Union, Generator, Iterator
from llm.llm_factory import BaiLianLLMFactory,LlamaIndexLLMFactory
from sentence_transformers import SentenceTransformer
# from llama_index.embeddings.huggingface import HuggingFaceEmbedding 
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from strategy.llamaIndexStrategy.agentStrategy import LlamaIndexAgentStrategy

# qwen_embedder = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")






Settings.embed_model = HuggingFaceEmbedding(
    model_name="all-mpnet-base-v2",
)

import yaml

def load_keywords_from_yaml(filepath: str) -> List[str]:
    with open(filepath, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    return [kw.lower() for kw in data.get("sysml_keywords", [])]

def convert_messages_to_prompt(messages: List[dict]) -> str:
    prompt = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"].strip()
        if role == "system":
            prompt += f"[System]: {content}\n"
        elif role == "user":
            prompt += f"[User]: {content}\n"
        elif role == "assistant":
            prompt += f"[Assistant]: {content}\n"
        else:
            prompt += f"[{role.capitalize()}]: {content}\n"
    return prompt.strip()


from llama_index.core.llms import ChatMessage

def convert_to_llama_chat_messages(messages: List[dict]) -> List[ChatMessage]:
    role_map = {
        "user": "user",
        "assistant": "assistant",
        "system": "system",
    }

    llama_messages = []
    for msg in messages:
        role = msg["role"]#
        content = msg["content"]
        llama_msg = ChatMessage(role=role, content=content)
        llama_messages.append(llama_msg)
    return llama_messages


# embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
# Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
# llm=BaiLianLLMFactory.create_llm_client("qwen-plus")
llm=LlamaIndexLLMFactory.create_llm_client("deepseek-chat","DeepSeek")
agentStrategy= LlamaIndexAgentStrategy(agent_type="FunctionCallingAgentWorker",llm=llm,use_chat_history=True)

judge_llm=BaiLianLLMFactory.create_llm_client("qwen-plus")


sysml_keywords = load_keywords_from_yaml("config/pipelines_cache/keywords.yaml")







# generated_tool=global_tool_kernel.get_tool("generate_id")
# bdd_model_tool=global_tool_kernel.get_tool("bdd-tool")
# # bdd_model_tool.return_direct = True
# tools = [generated_tool, bdd_model_tool]
# ✅ 自动注册所有 model_tool 工具
from strategy.tools.tool_utils import global_tool_kernel
import strategy.tools.all_tools  # 确保工具类被注册

tools = list(global_tool_kernel.iter_all_tools())
print("✅ 已注册工具:", [tool.metadata.name for tool in tools])


class Pipeline:
    def __init__(self):
        self.documents = None
        self.index = None

    async def on_startup(self):
        import os

        # # Set the OpenAI API key
        # os.environ["OPENAI_API_KEY"] = "your-api-key-here"

        # from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

        # self.documents = SimpleDirectoryReader("./data").load_data()
        # self.index = VectorStoreIndex.from_documents(self.documents)
        # This function is called when the server is started.
        pass

    async def on_shutdown(self):
        # This function is called when the server is stopped.
        pass

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        if user_message.strip().startswith("### Task:"):
            # If the user message starts with "###", it is a system message.
            # You can handle it differently if needed.
            return "This is a system message, not a user query."
        print("messages:", messages)
        converted_prompt = convert_messages_to_prompt(messages[-3:])
        judge_messages = [
            {
                "role": "system",
                "content": "You are a judgment expert. Your only task is to determine whether the user's latest message is related to SysML or MBSE modeling, or any operations involving SysML or MBSE. Only return '1' (related) or '0' (not related). Do not provide any explanations or additional text.。"
            },
            {
                "role": "user",
                "content": converted_prompt
            }
        ]
        llama_messages = convert_to_llama_chat_messages(messages)
        print("llama_messages:", llama_messages)
        key_word_flag = False
        for keyword in sysml_keywords:
            if keyword in user_message.lower():
                key_word_flag = True
                break
        if key_word_flag:
            print("The user message is related to MBSE SysML by keyword.")
            res=agentStrategy.get_agent_response(tools,user_message,messages=llama_messages)
            return str(res)
        else:
            
            judge_llm = BaiLianLLMFactory.create_llm_client("qwen-plus")
            judge_response = judge_llm.chat(messages=judge_messages)
            print("judge_response:", judge_response)
            if judge_response is None:
                return "⚠️ 无法判断当前消息是否需要工具调用"
            if "1" in judge_response:
                print("The user message is related to MBSE SysML.")
                res=agentStrategy.get_agent_response(tools,user_message,messages=llama_messages)
                return str(res)
            else:
                print("The user message is not related to MBSE SysML.")
                
                response = llm.chat(llama_messages)
                text = response.response if hasattr(response, "response") else str(response)
                if text.lower().startswith("assistant:"):
                    text = text[len("assistant:"):].lstrip()

                return text
                

        # This is where you can add your custom RAG pipeline.
        # Typically, you would retrieve relevant information from your knowledge base and synthesize it to generate a response.

        # print(messages)
        # print("start")
        # print(f"User message: {user_message}")
        # print("over")
        # print("messages:", messages)
        # print("body:", body)
        # print("model_id:", model_id)
        # print("over========")

        # query_engine = self.index.as_query_engine(streaming=True)
        # response = query_engine.query(user_message)

        # return response.response_gen
        # res="This is a placeholder response from the Llama Index pipeline."
        # return res
        # print("Using LLM to generate response...")
        # print(f" message: {messages}")
        # return llm.chat(messages)
        # print("type")

