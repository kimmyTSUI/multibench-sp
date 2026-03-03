"""
utils/llm_client.py
统一 LLM 调用接口。支持 OpenAI 兼容 API（含本地模型）。
"""

from typing import List, Dict
import os

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


def _clean_secret(value: str) -> str:
    """标准化 token / key 字符串，避免引号/空白导致请求头编码异常。"""
    if value is None:
        return ""
    v = str(value).strip()
    if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
        v = v[1:-1].strip()
    return v


class LLMClient:
    def __init__(
        self,
        model: str,
        api_key: str = None,
        base_url: str = None,
    ):
        self.model = model

        resolved_key = (
            api_key
            or os.environ.get("OPENAI_API_KEY", "")
            or os.environ.get("OPENROUTER_API_KEY", "")
            or os.environ.get("HF_TOKEN", "")
            or os.environ.get("HUGGINGFACEHUB_API_TOKEN", "")
        )
        self.api_key = _clean_secret(resolved_key)
        self.base_url = base_url.strip() if isinstance(base_url, str) else base_url  # e.g. "http://localhost:8000/v1"

        if self.api_key:
            try:
                self.api_key.encode("ascii")
            except UnicodeEncodeError as e:
                raise ValueError(
                    "API key contains non-ASCII characters. Please re-copy your token without Chinese quotes/whitespace."
                ) from e

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
