"""
prompts/sp_prompts.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Prompt Registry — 请在此文件中实现所有 prompt 函数。
其他模块均通过本文件统一获取 prompt，无需修改其他代码。

函数签名已固定（供 agents/ envs/ scripts/ 调用），
只需填写 return 语句中的字符串内容即可。
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from typing import Any


# ─────────────────────────────────────────────────────────────────────────────
# 1. Player System Prompts（角色定义）
#    role : 'fact' | 'value' | 'knowledge'
#    mode : 'zero' | 'few'
# ─────────────────────────────────────────────────────────────────────────────

def make_player_system_prompt(role: str, mode: str = "zero") -> str:
    """
    为 A/B/C 三类 player 生成 system prompt。
    role  : 'fact'      -> Player A，专注于事实性提问
            'value'     -> Player B，专注于价值/情感维度提问
            'knowledge' -> Player C，专注于背景知识/常识性提问
    mode  : 'zero' 不含示例；'few' 注入 few-shot 示例（见下方 FEW_SHOT_EXAMPLES）
    """
    # ── TODO：在此填写你的 system prompt ──────────────────────────────────────
    raise NotImplementedError(
        "请在 prompts/sp_prompts.py 中实现 make_player_system_prompt()"
    )


# ─────────────────────────────────────────────────────────────────────────────
# 2. Question Generation Prompt（生成下一问）
# ─────────────────────────────────────────────────────────────────────────────

def make_question_prompt(role: str, state: Any, mode: str = "zero") -> str:
    """
    给定当前游戏状态，返回用于生成下一个 Yes/No 问题的 user prompt。

    可用字段：
        state.surface        谜面文字
        state.host_summary   Host 上一轮的汇总（可为空字符串）
        state.history        QATurn 列表，每项有 .agent / .question / .answer / .axis
        state.round          当前轮数
    """
    # ── TODO：在此填写你的 question prompt ────────────────────────────────────
    raise NotImplementedError(
        "请在 prompts/sp_prompts.py 中实现 make_question_prompt()"
    )


# ─────────────────────────────────────────────────────────────────────────────
# 3. Final Answer Prompt（游戏结束时的最终解释）
# ─────────────────────────────────────────────────────────────────────────────

def make_final_answer_prompt(state: Any) -> str:
    """
    游戏结束后，让 player 根据 history 生成对谜底（bottom）的最终解释。

    可用字段同 make_question_prompt。
    """
    # ── TODO：在此填写你的 final answer prompt ────────────────────────────────
    raise NotImplementedError(
        "请在 prompts/sp_prompts.py 中实现 make_final_answer_prompt()"
    )


# ─────────────────────────────────────────────────────────────────────────────
# 4. Referee Prompt（裁判：回答 Yes / No / Unknown）
# ─────────────────────────────────────────────────────────────────────────────

def make_referee_prompt(surface: str, bottom: str, question: str) -> str:
    """
    裁判 prompt：已知谜面和谜底，判断 player 的问题应回答 Yes / No / Unknown。
    输出必须且只能是三者之一，供 SPEnv._normalize_answer() 解析。
    """
    # ── TODO：在此填写你的 referee prompt ────────────────────────────────────
    raise NotImplementedError(
        "请在 prompts/sp_prompts.py 中实现 make_referee_prompt()"
    )


# ─────────────────────────────────────────────────────────────────────────────
# 5. Host Prompts（汇总 & 停止判断）
# ─────────────────────────────────────────────────────────────────────────────

def make_summarize_prompt(state: Any) -> str:
    """
    Host summarize prompt：将本轮所有 Q&A 汇总成一段简洁的线索摘要，
    供下一轮各 Player 参考。
    """
    # ── TODO：在此填写你的 summarize prompt ──────────────────────────────────
    raise NotImplementedError(
        "请在 prompts/sp_prompts.py 中实现 make_summarize_prompt()"
    )


def make_stop_prompt(state: Any) -> str:
    """
    （可选）Host 智能停止判断 prompt：
    根据当前 history 判断是否已足够还原谜底，输出 'STOP' 或 'CONTINUE'。
    未实现时 HostAgent 将退回到固定轮数停止。
    """
    # ── TODO（可选）：在此填写你的 stop prompt ────────────────────────────────
    raise NotImplementedError(
        "请在 prompts/sp_prompts.py 中实现 make_stop_prompt()（可选）"
    )


def make_host_final_prompt(state: Any, player_answers: dict) -> str:
    """
    （可选）Host 最终汇总 prompt：整合 A/B/C 的最终解释，输出 Host 版本的谜底重构。
    """
    # ── TODO（可选）：在此填写你的 host final prompt ──────────────────────────
    raise NotImplementedError(
        "请在 prompts/sp_prompts.py 中实现 make_host_final_prompt()（可选）"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Few-Shot 示例库（mode='few' 时由上方函数注入，在此统一管理）
# ─────────────────────────────────────────────────────────────────────────────

FEW_SHOT_EXAMPLES: dict = {
    # 示例格式：
    # "fact": [
    #     {
    #         "surface": "...",
    #         "qa_pairs": [
    #             {"question": "...", "answer": "Yes"},
    #             ...
    #         ],
    #         "bottom": "...",
    #     }
    # ],
    "fact": [],       # TODO：填写 Player A (fact) 的 few-shot 示例
    "value": [],      # TODO：填写 Player B (value) 的 few-shot 示例
    "knowledge": [],  # TODO：填写 Player C (knowledge) 的 few-shot 示例
}
