"""
envs/sp_env.py
SP 游戏环境：负责规则与反馈。
- reset(sample) 初始化状态
- step(state, agent, question) 调用 Judge 给出 Yes/No/Unknown 并写入 history
"""

from typing import Tuple, Callable
from .sp_state import SPState, QATurn
from data.sp_dataset import SPSample


class SPEnv:
    def __init__(
        self,
        judge_client,                    # LLMClient 实例，用于回答 Yes/No/Unknown
        answer_prompt_builder: Callable, # 函数：(surface, bottom, question) -> str
        max_round: int = 25,
    ):
        """
        judge_client:          用于裁判的 LLMClient
        answer_prompt_builder: 导入自 prompts/sp_prompts.py 的 make_referee_prompt
        max_round:             最大轮数（可由 HostAgent 提前终止）
        """
        self.judge_client = judge_client
        self.answer_prompt_builder = answer_prompt_builder
        self.max_round = max_round

    # ── 初始化 ────────────────────────────────────────────────────────────────

    def reset(self, sample: SPSample) -> SPState:
        """载入一个 SPSample，返回初始化状态。"""
        return SPState(
            sample_id=sample.index,
            surface=sample.surface,
            bottom=sample.bottom,
            key_questions=sample.key_questions,
            max_round=self.max_round,
        )

    # ── 单步 ──────────────────────────────────────────────────────────────────

    def step(
        self,
        state: SPState,
        agent_name: str,
        question: str,
    ) -> Tuple[SPState, str]:
        """
        执行一步交互：
        1. 用 judge_client 回答 question（Yes / No / Unknown）
        2. 写入 history
        3. 返回 (new_state, answer)
        """
        # 构造裁判 prompt 并调用 LLM
        prompt = self.answer_prompt_builder(
            surface=state.surface,
            bottom=state.bottom,
            question=question,
        )
        messages = [{"role": "user", "content": prompt}]
        raw_answer = self.judge_client.chat(messages, temperature=0.0, max_tokens=16)

        # 规范化输出为 Yes / No / Unknown
        answer = self._normalize_answer(raw_answer)

        # 写入 history
        turn = QATurn(agent=agent_name, question=question, answer=answer)
        state.history.append(turn)

        return state, answer

    # ── 内部工具 ──────────────────────────────────────────────────────────────

    @staticmethod
    def _normalize_answer(raw: str) -> str:
        text = raw.strip().lower()
        if text.startswith("yes"):
            return "Yes"
        if text.startswith("no"):
            return "No"
        return "Unknown"
