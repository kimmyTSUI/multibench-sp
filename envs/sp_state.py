"""
envs/sp_state.py
SP 游戏的核心状态数据结构（单一事实来源）。
"""

from dataclasses import dataclass, field
from typing import List, Optional, Any, Dict


@dataclass
class QATurn:
    agent: str           # 提问者：'A' / 'B' / 'C'
    question: str        # 该 agent 提出的问题
    answer: str          # 环境/Judge 给出的回答：Yes / No / Unknown
    axis: Optional[str] = None   # 可选：由 Host 或 Judge 赋值的轴标签


@dataclass
class SPState:
    sample_id: int
    surface: str                        # 谜面（Players 可见）
    bottom: str                         # 谜底（仅 Env/Judge 可见）
    key_questions: List[str]            # AR-Bench key_question 列表

    round: int = 0
    max_round: int = 25

    history: List[QATurn] = field(default_factory=list)
    host_summary: str = ""

    done: bool = False
    done_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """序列化为可 JSON 化的字典（供 EpisodeLog 使用）。"""
        return {
            "sample_id": self.sample_id,
            "surface": self.surface,
            "bottom": self.bottom,
            "key_questions": self.key_questions,
            "round": self.round,
            "max_round": self.max_round,
            "history": [
                {
                    "agent": t.agent,
                    "question": t.question,
                    "answer": t.answer,
                    "axis": t.axis,
                }
                for t in self.history
            ],
            "host_summary": self.host_summary,
            "done": self.done,
            "done_reason": self.done_reason,
        }
