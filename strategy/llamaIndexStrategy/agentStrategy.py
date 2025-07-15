from llama_index.core.workflow import Context
from llama_index.core.program.function_program import get_function_tool
from llama_index.core.chat_engine.types import (
    AGENT_CHAT_RESPONSE_TYPE,
    AgentChatResponse,
    ChatResponseMode,
    StreamingAgentChatResponse,
)
from llama_index.core.llms import ChatMessage
from llama_index.core.tools import QueryEngineTool
from llama_index.core.query_engine import SubQuestionQueryEngine
from llm.llm_factory import LlamaIndexLLMFactory
from typing import List, Optional, Union
from llama_index.core.agent import ReActAgent, FunctionCallingAgentWorker
class LlamaIndexAgentStrategy():    
    """策略类，用于定义 LLM Agent的策略"""
    def __init__(self, agent_type,llm=None,llamaindex_llm_strategy: LlamaIndexLLMFactory=None,use_chat_history: bool = False):
        
        if llamaindex_llm_strategy is not None:
            self.llm = llamaindex_llm_strategy.llm
        else:
            self.llm = llm
        self.agent_type = agent_type
        self.use_chat_history = use_chat_history  # ✅ 控制是否使用记忆
        self.chat_history: List[ChatMessage] = []
    
    def get_agent_response(self, tools, user_msg, **kargs) -> AgentChatResponse:
        """
        根据 agent_type 获取 agent 的响应
        :param tools: 工具列表
        :param user_msg: 用户消息
        :param kargs: 其他参数
        :return: AgentChatResponse
        """
        res=None
        print("user_msg:", user_msg)
        if self.agent_type == "ReActAgent":
            res= self._get_from_ReActAgent(tools, user_msg, **kargs)
        elif self.agent_type == "FunctionCallingAgentWorker":
            res= self._get_from_FunctionCallingAgentWorker(tools, user_msg, **kargs)
        else:
            raise ValueError(f"Unsupported agent type: {self.agent_type}")
        print(f"Agent response test: {res}")
        return res
        
    def _get_from_ReActAgent(self, tools, user_msg, **kargs)->AgentChatResponse:
        chat_history = self.chat_history if self.use_chat_history else None
        agent = ReActAgent.from_tools(
            tools=tools,
            llm=self.llm,
        )
        response = agent.chat(
            message=user_msg,
            chat_history=chat_history  # ✅ 根据开关决定是否使用历史
        )
        # ✅ 如果使用记忆，就更新历史
        if self.use_chat_history and hasattr(response, "response"):
            self._append_to_history(user_msg, response.response)
        return response
    
    def _get_from_FunctionCallingAgentWorker(self, tools, user_msg, **kargs)->AgentChatResponse:
        chat_history = self.chat_history if self.use_chat_history else None
        
        agent_worker = FunctionCallingAgentWorker.from_tools(
            tools=tools,
            llm=self.llm,
            verbose=kargs.get("verbose", True),
            allow_parallel_tool_calls=kargs.get("allow_parallel_tool_calls", False),
        )
        agent= agent_worker.as_agent()
        response = agent.chat(
            message=user_msg,
            chat_history=chat_history  # ✅ 根据开关决定是否使用历史
        )
        if self.use_chat_history and hasattr(response, "response"):
            self._append_to_history(user_msg, response.response)
        # print(f"Agent response: {response}")
        return response

    def get_agent_response_with_aggregate_tool(
        self,
        tools: list,
        user_msg: str,
        aggregate_tool_name: str = "aggregate_query_tool",
        group_name: str = "agent_tools",
        **kargs
    ) -> AgentChatResponse:
        """
        使用 SubQuestionQueryEngine 构造整体查询工具，并交给 agent 使用。
        source: https://docs.llamaindex.ai/en/stable/examples/agent/Chatbot_SEC/
        """

        # === 1. 将工具转换为 QueryEngineTool（如果尚未转换）
        query_engine_tools = []
        for tool in tools:
            if isinstance(tool, QueryEngineTool):
                query_engine_tools.append(tool)
            elif hasattr(tool, "as_query_engine"):
                query_engine = tool.as_query_engine()
                query_engine_tool = QueryEngineTool.from_defaults(
                    query_engine=query_engine,
                    name=tool.metadata.name,
                    description=tool.metadata.description,
                )
                query_engine_tools.append(query_engine_tool)
            else:
                raise ValueError(f"Tool {tool} is not a valid query engine tool")

        # === 2. 构建整体查询引擎
        query_engine = SubQuestionQueryEngine.from_defaults(
            query_engine_tools=query_engine_tools,
            llm=self.llm,
        )

        # === 3. 将整体查询引擎包装成工具
        aggregate_tool = QueryEngineTool.from_defaults(
            query_engine=query_engine,
            name=aggregate_tool_name,
            description="一个聚合工具，会将复杂问题拆分后由多个子引擎处理后聚合结果。",
        )

        # === 4. 注册工具
        global_tool_kernel.register_tool(aggregate_tool, group=group_name, override=True)
        for tool in query_engine_tools:
            global_tool_kernel.register_tool(tool, group=group_name, override=True)

        # === 5. 构造完整工具列表并执行
        full_tools = [aggregate_tool] + query_engine_tools
        return self.get_agent_response(full_tools, user_msg, **kargs)
    
    def _append_to_history(self, user_input: str, agent_output: str):
        self.chat_history.append(ChatMessage(role="user", content=user_input))
        self.chat_history.append(ChatMessage(role="assistant", content=agent_output))

    def clear_chat_history(self):
        self.chat_history.clear()