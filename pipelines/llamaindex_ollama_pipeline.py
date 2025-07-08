"""
title: Local LlamaIndex Pipeline with Ollama + BGE Embedding
author: caojie
description: RAG pipeline using LlamaIndex with local embedding and local LLM (DeepSeek-R1 1.5B via Ollama).
requirements: llama-index, sentence-transformers, pymupdf
"""

from typing import List, Union, Generator, Iterator
from schemas import OpenAIChatMessage


class Pipeline:
    def __init__(self):
        self.index = None
        self.query_engine = None
        self.id = "llamaindex_ollama_pipeline"
        self.name = "Local SysML XMI Generator (Ollama + Embedding)"
        self.type = "pipe"

    async def on_startup(self):
        from pathlib import Path
        from llama_index.core.settings import Settings
        from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        from llama_index.readers.file import PDFReader
        embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en")
        Settings.embed_model = embed_model
        Settings.llm = None
        pdf_path = "C:/develop/LLMDeploy/sysml-knowledge-base/OMGSysML-v1.7-bdd.pdf"
        persist_dir = "./storage"

        loader = PDFReader()
        documents = loader.load_data(pdf_path)

 

        if Path(persist_dir).exists():
            storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
            self.index = load_index_from_storage(storage_context)
        else:
            self.index = VectorStoreIndex.from_documents(documents)
            self.index.storage_context.persist(persist_dir=persist_dir)

        self.query_engine = self.index.as_query_engine(streaming=True)

    async def on_shutdown(self):
        pass

    def extract_keywords(self, text: str) -> str:
        import re
        keywords = []
        patterns = ["bdd", "ibd", "activity", "use case", "state machine", "sequence"]
        for p in patterns:
            if re.search(p, text, re.IGNORECASE):
                keywords.append(p)
        return ", ".join(keywords)

    def run_ollama(self, prompt: str) -> str:
        import subprocess

        command = ["ollama", "run", "deepseek-r1:1.5b"]
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding="utf-8"  
        )   
        stdout, stderr = process.communicate(input=prompt)
        if process.returncode != 0:
            return f"Error from Ollama:\n{stderr}"
        return stdout
    


    def pipe(self, user_message: str, model_id: str, messages: List[dict], body: dict) -> Union[str, Generator, Iterator]:
        print(f"[PIPE] User message: {user_message}")
        if not self.query_engine:
            raise RuntimeError("Query engine is not initialized.")

        keywords = self.extract_keywords(user_message)
        rag_query = f"User intends to create SysML diagram(s): {keywords}. Explain and expand.\n\n{user_message}"

        retrieved = self.query_engine.query(rag_query)
        context = "".join([chunk.text for chunk in retrieved.source_nodes])

        prompt = f"""You are a SysML assistant. Given a modeling request and some relevant knowledge context, generate a SysML XMI snippet.

### Modeling Request:
{user_message}

### Keywords: {keywords}

### Context:
{context}

### Output:
Provide a minimal SysML XMI code snippet based on the above.
"""

        result = self.run_ollama(prompt)
        yield result
Pipeline = Pipeline
