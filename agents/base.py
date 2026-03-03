"""
agents/base.py
BaseAgent：所有 Player Agent 的抽象基类。
"""

from typing import Optional, Callable
from envs.sp_state import SPState


class BaseAgent:
    def __init__(
        self,
        name: str,
        client,                           # LLMClient 实例
        system_prompt_builder: Callable,  # make_player_system_prompt(role, mode)
        question_prompt_builder: Callable,# make_question_prompt(role, state, mode)
        final_prompt_builder: Callable,   # make_final_answer_prompt(state)
        role: str = "fact",
        mode: str = "zero",
    ):
        """
        name                   : agent 标识符，如 'A' / 'B' / 'C'
        client                 : LLMClient
        system_prompt_builder  : prompts.sp_prompts.make_player_system_prompt
        question_prompt_builder: prompts.sp_prompts.make_question_prompt
        final_prompt_builder   : prompts.sp_prompts.make_final_answer_prompt
        role                   : 'fact' | 'value' | 'knowledge'
        mode                   : 'zero' | 'few'
        """
        self.name = name
        self.client = client
        self.system_prompt_builder = system_prompt_builder
        self.question_prompt_builder = question_prompt_builder
        self.final_prompt_builder = final_prompt_builder
        self.role = role
        self.mode = mode

        # 缓存 system prompt（每局不变）
        self._system_prompt: Optional[str] = None

    @property
    def system_prompt(self) -> str:
        if self._system_prompt is None:
            self._system_prompt = self.system_prompt_builder(
                role=self.role, mode=self.mode
            )
        return self._system_prompt

    # ── 核心接口 ──────────────────────────────────────────────────────────────

    def act(self, state: SPState) -> str:
        """
        在当前 state 下生成下一个 Yes/No 问题。
        返回问题字符串。
        """
        raise NotImplementedError

    def finalize(self, state: SPState) -> str:
        """
        游戏结束时，根据完整 history 生成对谜底的最终解释。
        返回自由文本。
        """
        raise NotImplementedError

    # ── 工具方法 ──────────────────────────────────────────────────────────────

    def _build_messages(self, user_content: str):
        """构造标准 [system, user] 消息列表。"""
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user",   "content": user_content},
        ]
