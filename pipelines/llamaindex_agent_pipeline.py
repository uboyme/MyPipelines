"""
title: LlamaIndex Agent Pipeline
author: caojie
description: SysML modeling and QA agent with LlamaIndex and DeepSeek API
dependencies: llama-index, requests, python-dotenv
"""

from typing import List, Union, Generator, Iterator
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.readers.file import PDFReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.agent.openai import OpenAIAgent
from llama_index.llms.openai import OpenAI
from llama_index.core.tools import FunctionTool
from llama_index.core.settings import Settings
from dotenv import load_dotenv
import os, requests

class DeepSeekLLM(OpenAI):
    def _get_model_name(self) -> str:
        return "gpt-3.5-turbo"  # fake name to bypass context size check

    @property
    def metadata(self):
        class Meta:
            context_window = 4096
            num_output = 1024
            is_chat_model = True
            is_function_calling_model = True
            is_code_model = False
        return Meta()

class ToolSet:
    def __init__(self, index):
        self.query_engine = index.as_query_engine()

    def retrieve_knowledge(self, query: str) -> str:
        response = self.query_engine.query(query)
        return str(response)

    def generate_xmi(self, request: str, context: str = "") -> str:
        prompt = f"""You are a SysML assistant. Given a modeling request and context, generate a minimal SysML XMI snippet.

Modeling Request:
{request}

Context:
{context}

Output:
"""

        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_API_BASE", "https://api.deepseek.com/v1")

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        data = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": "You are a SysML modeling expert."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7
        }

        response = requests.post(f"{base_url}/chat/completions", headers=headers, json=data)
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            return f"Error from DeepSeek API:\n{response.text}"

    def get_tools(self):
        return [
            FunctionTool.from_defaults(fn=self.retrieve_knowledge),
            FunctionTool.from_defaults(fn=self.generate_xmi),
        ]


class Pipeline:
    def __init__(self):
        self.agent = None
        self.id = "llamaindex_agent_pipeline"
        self.name = "SysML Agent (DeepSeek + RAG)"
        self.type = "pipe"

    async def on_startup(self):
        load_dotenv()
        embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en")
        Settings.embed_model = embed_model

        llm = DeepSeekLLM(
            model="deepseek-chat",
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_API_BASE", "https://api.deepseek.com/v1"),
            is_chat_model=True,
            strict=False
        )
        Settings.llm = llm

        pdf_path = "C:/develop/LLMDeploy/sysml-knowledge-base/OMGSysML-v1.7-bdd.pdf"
        persist_dir = "./storage"

        if os.path.exists(persist_dir):
            storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
            index = load_index_from_storage(storage_context)
        else:
            documents = PDFReader().load_data(pdf_path)
            index = VectorStoreIndex.from_documents(documents)
            index.storage_context.persist(persist_dir=persist_dir)

        tools = ToolSet(index).get_tools()
        self.agent = OpenAIAgent.from_tools(tools, llm=llm, system_prompt="You are a SysML agent capable of knowledge retrieval and model generation.")

    async def on_shutdown(self):
        pass

    def pipe(self, user_message: str, model_id: str, messages: List[dict], body: dict) -> Union[str, Generator, Iterator]:
        response = self.agent.chat(user_message)
        yield str(response)

Pipeline = Pipeline
