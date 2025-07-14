from openai import OpenAI
from llama_index.core.llms import ChatResponse, ChatMessage
from llama_index.core.llms import MessageRole, TextBlock


class QwenClient:
    def __init__(self, model_name="qwen-plus", api_key=None, api_base=None):
        self.name = model_name
        self.api_key = api_key or "sk-0067532e096c46168dbede4b65a44d6a"
        self.api_base = api_base or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.api_base
        )

    def _convert_to_openai_messages(self, messages):
        result = []
        for msg in messages:
            if isinstance(msg, ChatMessage):
                text = ""
                for block in msg.blocks:
                    if isinstance(block, TextBlock):
                        text += block.text
                result.append({
                    "role": msg.role.value,
                    "content": text.strip()
                })
            elif isinstance(msg, dict):
                result.append(msg)
        return result

    def chat(self, prompt=None, messages=None, **kwargs):
        try:
            if messages is not None:
                payload = self._convert_to_openai_messages(messages)
            elif prompt is not None:
                payload = [
                    {"role": "system", "content": "You are a helpful assistant"},
                    {"role": "user", "content": prompt}
                ]
            else:
                raise ValueError("必须提供 prompt 或 messages")

            print("🧪 Final payload to OpenAI:", payload)

            response = self.client.chat.completions.create(
                model=self.name,
                messages=payload,
                stream=False
            )

            reply = response.choices[0].message.content
            return ChatResponse(message=ChatMessage(role="assistant", content=reply))

        except Exception as e:
            print(f"LLM调用出错: {e}")
            return ChatResponse(message=ChatMessage(role="assistant", content="对话出错，请稍后重试"))