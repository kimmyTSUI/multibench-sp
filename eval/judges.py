"""
eval/judges.py
评估阶段的 LLM Judge 工具：
- key_question_judge_prompt : 判断某 key_question 是否被 history 覆盖
- final_answer_judge_prompt : 评价 final_answer 与 bottom 的语义相似度
"""

from typing import Optional


def key_question_judge_prompt(key_question: str, history_text: str) -> str:
    """
    判断 history_text 中的信息是否足以回答 key_question。
    输出：'Yes' 或 'No'
    """
    return (
        f"Based on the following Q&A history from a mystery story game:\n\n"
        f"{history_text}\n\n"
        f"Can you determine the answer to this key question: \"{key_question}\"?\n"
        f"Reply ONLY with 'Yes' or 'No'."
    )


def semantic_match_judge_prompt(prediction: str, reference: str) -> str:
    """
    评价 prediction 与 reference（谜底）的语义一致性。
    输出：'Yes' 或 'No'
    """
    return (
        f"Reference (true story bottom):\n{reference}\n\n"
        f"Predicted explanation:\n{prediction}\n\n"
        f"Does the predicted explanation capture the core meaning of the reference? "
        f"Reply ONLY with 'Yes' or 'No'."
    )
