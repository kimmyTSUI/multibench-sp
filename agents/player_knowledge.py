"""
agents/player_knowledge.py
Player C：专注于「背景知识/常识维度」提问（common sense / domain knowledge）。
"""

from .base import BaseAgent
from envs.sp_state import SPState


class PlayerKnowledge(BaseAgent):
    """Player C：Background knowledge-oriented questioner."""

    def __init__(self, client, system_prompt_builder, question_prompt_builder,
                 final_prompt_builder, mode: str = "zero"):
        super().__init__(
            name="C",
            client=client,
            system_prompt_builder=system_prompt_builder,
            question_prompt_builder=question_prompt_builder,
            final_prompt_builder=final_prompt_builder,
            role="knowledge",
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
