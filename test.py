from llama_index.llms.deepseek import DeepSeek
from llama_index.core.agent import FunctionCallingAgentWorker
from llama_index.core.program.function_program import function_tool

def test_func(query: str) -> str:
    """Test tool for function calling."""
    print("被触发了！")
    return f"tool result for: {query}"

llm = DeepSeek(model="deepseek-chat", api_key="你的key", api_base="https://api.deepseek.com/v1")
agent_worker = FunctionCallingAgentWorker.from_tools(
    tools=[test_func],
    llm=llm,
    verbose=True,
)
response = agent_worker.as_agent().chat(message="请用工具查找内容", chat_history=None)
print("[DEBUG] agent工具调用结果:", response)
