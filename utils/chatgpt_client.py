import os
from openai import OpenAI

# 可以在环境变量中配置 OPENAI_API_KEY，否则在 app.py 中传入
_api_key = os.getenv("OPENAI_API_KEY", None)


def get_client(api_key: str = None) -> OpenAI:
    """
    返回一个 OpenAI 客户端实例。优先使用传入的 api_key，否则尝试环境变量。
    """
    key = api_key or _api_key
    if not key:
        raise ValueError("必须提供 OpenAI API Key")
    return OpenAI(api_key=key)


def chat_completion(client: OpenAI, model: str, messages: list, temperature: float = 0.7, max_tokens: int = 2048):
    """
    统一封装对 OpenAI Chat Completion 的调用。

    参数：
    - client: OpenAI 客户端实例
    - model: 模型名称，例如 "chatgpt-4o-latest"、"gpt-4"、"gpt-3.5-turbo"
    - messages: Chat API 的消息列表，格式同 OpenAI SDK 要求
    - temperature: 生成随机性参数
    - max_tokens: 最大 token 数

    返回：
    - 完整的文本响应
    """
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens
    )
    return response.choices[0].message.content
