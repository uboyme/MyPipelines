from typing import Optional,Callable,Type
from pydantic import BaseModel
from llama_index.core.tools import FunctionTool

# ✅ 全局工具注册器
class GlobalToolKernel:
    def __init__(self):
        self._tools = {}

    def register_tool(self, tool, group: str = "default", override: bool = False):
        name = tool.metadata.name
        if group not in self._tools:
            self._tools[group] = {}
        if name in self._tools[group] and not override:
            raise ValueError(f"Tool '{name}' already registered in group '{group}'.")
        self._tools[group][name] = tool

    def get_tool(self, name: str, group: str = "default"):
        return self._tools.get(group, {}).get(name)

    def iter_all_tools(self):
        for group in self._tools.values():
            for tool in group.values():
                yield tool

    def list_all_tools(self):
        return [tool for group in self._tools.values() for tool in group.values()]


global_tool_kernel = GlobalToolKernel()
def model_tool(
    name: str=None,
    group: str = "models",
    return_direct: bool = False,
    description: Optional[str] = None
) -> Callable[[Type[BaseModel]], Type[BaseModel]]:
    """将 Pydantic 模型转换为工具"""
    def decorator(model_cls: Type[BaseModel]) -> Type[BaseModel]:
        # 假设 get_function_tool 是你的自定义函数
        tool = FunctionTool.from_defaults(fn=model_cls.call) 
        tool.metadata.name = name or model_cls.__name__
        if description:
            tool.metadata.description = description
        elif model_cls.__doc__:
            tool.metadata.description = model_cls.__doc__ 

        if return_direct:
            tool.metadata.return_direct = return_direct
        
        global_tool_kernel.register_tool(tool, group=group)
        return model_cls
    return decorator