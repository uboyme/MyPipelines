
from strategy.tools.kernel import (
    retrive_sysml_standard,
    retrive_sysml_rules,
    retrive_few_shot_examples,
)
from strategy.tools.wrap_whole_xmi_from_content import (
    wrap_whole_xmi_from_content,
    export_xmi_file,  
)

from strategy.llamaIndexStrategy.agentStrategy import LlamaIndexAgentStrategy
from llm.llm_factory import BaiLianLLMFactory,LlamaIndexLLMFactory
from sentence_transformers import SentenceTransformer
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from typing import List, Union, Generator, Iterator
from llama_index.core.llms import ChatMessage
import yaml
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from llama_index.core.tools import FunctionTool

Settings.embed_model = HuggingFaceEmbedding(model_name="all-mpnet-base-v2")

llm = LlamaIndexLLMFactory().create_llm_client(model="deepseek-chat", platform="DeepSeek")
agentStrategy= LlamaIndexAgentStrategy(agent_type="FunctionCallingAgentWorker",llm=llm,use_chat_history=True)
judge_llm=BaiLianLLMFactory().create_llm_client("qwen-plus")

sysml_keywords = []
def load_keywords_from_yaml(filepath: str,keyward) -> List[str]:
    with open(filepath, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    return [kw.lower() for kw in data.get(keyward, [])]

sysml_keywords = load_keywords_from_yaml("config/pipelines_cache/keywords.yaml","sysml_keywords")

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

def convert_to_llama_chat_messages(messages: List[dict]) -> List[ChatMessage]:
    role_map = {"user": "user", "assistant": "assistant", "system": "system"}
    return [ChatMessage(role=role_map.get(m["role"], "user"), content=m["content"]) for m in messages]

tools = [
    FunctionTool.from_defaults(fn=retrive_sysml_standard),
    FunctionTool.from_defaults(fn=retrive_sysml_rules),
    FunctionTool.from_defaults(fn=retrive_few_shot_examples),
    FunctionTool.from_defaults(fn=wrap_whole_xmi_from_content),
    FunctionTool.from_defaults(fn=export_xmi_file),
]

class Pipeline:
    def __init__(self):
        self.documents = None
        self.index = None
        self.last_xmi_content = None

    async def on_startup(self): pass
    async def on_shutdown(self): pass

    def pipe(self, user_message: str, model_id: str, messages: List[dict], body: dict) -> Union[str, Generator, Iterator]:
        if user_message.strip().startswith("### Task:"):
            return "This is a system message, not a user query."

        converted_prompt = convert_messages_to_prompt(messages[-3:])
        sysml_judge_messages = convert_to_llama_chat_messages([
            {"role": "system", "content": "You are a judgment expert. Your only task is to determine whether the user's latest message is related to SysML or MBSE modeling, or any operations involving SysML or MBSE. Only return '1' (related) or '0' (not related). Do not provide any explanations or additional text."},
            {"role": "user", "content": converted_prompt}
        ])
        download_judge_messages = convert_to_llama_chat_messages([
            {"role": "system", "content": "You are a judgment expert. Your only task is to determine whether the user's latest intitive is want to export or download the xmi file. Only return '1' (yes) or '0' (no).  if generate a sysml ,then it's not belong download xmi, just return 0. Do not provide any explanations or additional text."},
            {"role": "user", "content": converted_prompt}
        ])
        llama_messages = convert_to_llama_chat_messages(messages)
        sysml_key_word_flag = any(keyword in user_message.lower() for keyword in sysml_keywords)
        # ✅ 【建议加这一行】
        print("[DEBUG] sysml_key_word_flag:", sysml_key_word_flag)

        judge_result = judge_llm.chat(messages=sysml_judge_messages).strip()
        print("[DEBUG] judge_llm 返回:", judge_result)
        if sysml_key_word_flag or judge_result.strip().endswith("1"):
            print("SysML query detected.")
            if judge_llm.chat(messages=download_judge_messages).strip() == "1":
                if not self.last_xmi_content:
                    return "No last XMI content found, please generate XMI first."
                return export_xmi_file(xmi_content=self.last_xmi_content, file_name="sysml_model.xmi")

            llama_messages.insert(0, ChatMessage(role="system", content=(
                "You are an expert in MBSE and SysML. Use the tools provided to retrieve rules, examples, wrap into XMI, and export files.\n"
                "Never generate plain XMI directly. Always use tools.\n"
                "1. Use the 'retrive_sysml_rules' tool to get the SysML rules.\n"
                "2. Use the 'retrive_few_shot_examples' tool to get few-shot examples. and you should learn focus_element in step 2\n"
                "3. Use the 'wrap_whole_xmi_from_content' tool to wrap the content into XMI.\n"
                "4. Use the 'export_xmi_file' tool to export the file if needed."
            )))
            res = agentStrategy.get_agent_response(tools, user_message, messages=llama_messages)
            return_text = str(res)
            if "<xmi:XMI xmlns:MD_Customization_for_SysML__additional_stereotypes=" in return_text:
                self.last_xmi_content = return_text[6:-3]
            return return_text
        else:
            print("Not related to SysML.")
            response = llm.chat(llama_messages)
            return response.response if hasattr(response, "response") else str(response)
