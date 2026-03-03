"""
agents/player_fact.py
Player A：专注于「事实维度」提问（What / Who / When / Where）。
差异仅来自 role='fact' 传给 prompt_builder；核心逻辑在 BaseAgent。
"""

from .base import BaseAgent
from envs.sp_state import SPState


class PlayerFact(BaseAgent):
    """Player A：Fact-oriented questioner."""

    def __init__(self, client, system_prompt_builder, question_prompt_builder,
                 final_prompt_builder, mode: str = "zero"):
        super().__init__(
            name="A",
            client=client,
            system_prompt_builder=system_prompt_builder,
            question_prompt_builder=question_prompt_builder,
            final_prompt_builder=final_prompt_builder,
            role="fact",
            mode=mode,
        )

    def act(self, state: SPState) -> str:
        user_content = self.question_prompt_builder(
            role=self.role, state=state, mode=self.mode
        )
        messages = self._build_messages(user_content)
        return self.client.chat(messages, temperature=0.7, max_tokens=128)

    def finalize(self, state: SPState) -> str:
        user_content = self.final_prompt_builder(state=state)
        messages = self._build_messages(user_content)
        return self.client.chat(messages, temperature=0.3, max_tokens=512)
