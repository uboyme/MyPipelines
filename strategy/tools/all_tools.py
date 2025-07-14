from pathlib import Path
import re
from pydantic import BaseModel
from strategy.tools.tool_utils import model_tool

# === 原始核心函数逻辑（从 kernel.py 迁移而来） ===
def retrive_sysml_standard(query: str) -> str:
    base_path = Path("dataset/rag_content/standard")
    matched = []
    for file in base_path.glob("*.md"):
        content = file.read_text(encoding="utf-8")
        if query.lower() in content.lower():
            matched.append(f"### {file.stem}\n\n{content}\n")
    return "\n".join(matched[:3]) if matched else "未检索到相关 SysML 标准内容。"

def retrive_sysml_rules(query: str) -> str:
    base_path = Path("dataset/rag_content/sysml-rules")
    matched = []
    for file in base_path.glob("*.md"):
        content = file.read_text(encoding="utf-8")
        if query.lower() in content.lower():
            matched.append(f"### {file.stem}\n\n{content}\n")
    return "\n".join(matched[:3]) if matched else "未检索到相关建模规则。"

def retrive_few_shot_examples(query: str) -> str:
    base_path = Path("dataset/rag_content/few_shot_examples")
    matched = []
    for file in base_path.glob("*.md"):
        content = file.read_text(encoding="utf-8")
        if query.lower() in content.lower():
            matched.append(f"### {file.stem}\n\n{content}\n")
    return "\n".join(matched[:3]) if matched else "未检索到相关示例。"

# === 封装工具 ===
@model_tool(name="export_xmi_file", group="xmi")
class ExportXMIModel(BaseModel):
    """导出XMI文件"""
    file_name: str
    content: str

    def call(self) -> str:
        print(f"Exporting SysML model with name: {self.file_name}")

        download_dir = Path("C:/develop/LLMDeploy/pipelines/uploaded_xmi")
        download_dir.mkdir(parents=True, exist_ok=True)

        safe_name = "".join(c for c in self.file_name if c.isalnum() or c in ('_', '-', '.')).rstrip('.')
        if not safe_name.endswith(".xml"):
            safe_name += ".xml"

        file_path = download_dir / safe_name

        pattern = r"```(?:xml|xmi|uml)?\s*([\s\S]*?)\s*```"
        match = re.search(pattern, self.content)
        fileter_content = match.group(1).strip() if match else self.content.strip()

        if "<xmi:XMI" not in fileter_content or "<uml:Model" not in fileter_content:
            print("Content already contains XMI root element, no need to wrap.")
            return "Content does not contain valid XMI root element. Please ensure the content is a whold xmi file."

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(fileter_content)

        url = f"http://localhost:8087/downloads/{safe_name}"
        return f"[点击下载 {safe_name}]({url})"


@model_tool(name="retrive_sysml_standard", group="retrive")
class SysMLStandardQuery(BaseModel):
    """基于查询语句，从SysML标准中检索内容"""
    query: str

    def call(self) -> str:
        return retrive_sysml_standard(self.query)


@model_tool(name="retrive_sysml_rules", group="retrive")
class SysMLRulesQuery(BaseModel):
    """检索与生成SysML模型相关的规则内容"""
    query: str

    def call(self) -> str:
        return retrive_sysml_rules(self.query)


@model_tool(name="retrive_few_shot_examples", group="retrive")
class FewShotExamplesQuery(BaseModel):
    """检索与任务相关的few-shot 示例"""
    query: str

    def call(self) -> str:
        return retrive_few_shot_examples(self.query)
