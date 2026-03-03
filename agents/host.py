"""
agents/host.py
HostAgent（Player D）：调度轮次、汇总线索、判断终止。
"""

from typing import List, Optional, Callable
from envs.sp_state import SPState


class HostAgent:
    def __init__(
        self,
        client,
        summarize_prompt_builder: Callable,           # make_summarize_prompt(state)
        stop_prompt_builder: Optional[Callable] = None,  # make_stop_prompt(state) 可选
        final_prompt_builder: Optional[Callable] = None, # make_host_final_prompt(state, player_answers)
    ):
        """
        client                  : LLMClient（Host 使用的 LLM）
        summarize_prompt_builder: prompts.sp_prompts.make_summarize_prompt
        stop_prompt_builder     : prompts.sp_prompts.make_stop_prompt（可选；None 时退化为固定轮数）
        final_prompt_builder    : prompts.sp_prompts.make_host_final_prompt（可选）
        """
        self.client = client
        self.summarize_prompt_builder = summarize_prompt_builder
        self.stop_prompt_builder = stop_prompt_builder
        self.final_prompt_builder = final_prompt_builder

    # ── 核心接口 ──────────────────────────────────────────────────────────────

    def summarize(self, state: SPState) -> str:
        """
        把当前 history 总结成 host_summary，供下一轮共享给 A/B/C。
        直接更新并返回 state.host_summary 字符串。
        """
        prompt = self.summarize_prompt_builder(state=state)
        messages = [{"role": "user", "content": prompt}]
        summary = self.client.chat(messages, temperature=0.3, max_tokens=256)
        state.host_summary = summary
        return summary

    def choose_turn_order(
        self, state: SPState, players: List[str]
    ) -> List[str]:
        """
        返回本轮 player 的出场顺序列表。
        默认固定顺序 A -> B -> C；如需动态顺序可在子类中覆盖。
        """
        return players  # ['A', 'B', 'C']

    def should_stop(self, state: SPState) -> bool:
        """
        终止判断：
        1. 若提供了 stop_prompt_builder，调用 LLM 判断是否输出 'STOP'；
        2. 否则退化为固定轮数判断。
        """
        if state.round >= state.max_round:
            return True

        if self.stop_prompt_builder is not None:
            try:
                prompt = self.stop_prompt_builder(state=state)
                messages = [{"role": "user", "content": prompt}]
                resp = self.client.chat(messages, temperature=0.0, max_tokens=8)
                return resp.strip().upper().startswith("STOP")
            except NotImplementedError:
                pass  # stop_prompt_builder 未实现时退化

        return False

    def aggregate_final(
        self, state: SPState, player_answers: dict
    ) -> str:
        """
        汇总 A/B/C 的最终解释，输出 Host 版本。
        若 final_prompt_builder 未实现，直接拼接 player_answers。
        """
        if self.final_prompt_builder is not None:
            try:
                prompt = self.final_prompt_builder(
                    state=state, player_answers=player_answers
                )
                messages = [{"role": "user", "content": prompt}]
                return self.client.chat(messages, temperature=0.3, max_tokens=512)
            except NotImplementedError:
                pass

        # 降级：简单拼接
        parts = [f"[{k}]: {v}" for k, v in player_answers.items()]
        return "\n".join(parts)
