"""
utils/llm_client.py
统一 LLM 调用接口。支持 OpenAI 兼容 API（含本地模型）。
"""

from typing import List, Dict, Optional
import os

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


class LLMClient:
    def __init__(
        self,
        model: str,
        api_key: str = None,
        base_url: str = None,
    ):
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.base_url = base_url  # e.g. "http://localhost:8000/v1" for local

        if OpenAI is None:
            raise ImportError("请先安装 openai 包：pip install openai")

        self._client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

    def chat(
        self,
        messages: List[Dict],
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 512,
    ) -> str:
        """
        发送 messages 列表，返回 assistant 的 content 字符串。
        messages 格式: [{"role": "system"/"user"/"assistant", "content": "..."}]
        """
        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()
