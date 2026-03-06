"""
prompts/sp_prompts.py
Centralized prompt registry for MultiBench SP.
"""

from typing import Any


def _history_to_text(state: Any, limit: int = 18) -> str:
    turns = state.history[-limit:] if state.history else []
    if not turns:
        return "(empty)"
    lines = []
    for i, t in enumerate(turns, 1):
        lines.append(f"{i}. [{t.agent}] Q: {t.question} | A: {t.answer}")
    return "\n".join(lines)


# 1) Player System Prompts ---------------------------------------------------

def make_player_system_prompt(role: str, mode: str = "zero") -> str:
    role_desc = {
        "fact": (
            "You are Player A (Fact axis). Ask concrete, objective yes/no questions "
            "about who/what/when/where and observable events."
        ),
        "value": (
            "You are Player B (Value/Emotion axis). Ask yes/no questions about "
            "intentions, motives, emotional state, interpersonal relations, and value conflict."
        ),
        "knowledge": (
            "You are Player C (Background Knowledge axis). Ask yes/no questions that leverage "
            "common sense, domain knowledge, social rules, or hidden context assumptions."
        ),
    }.get(role, "You are a careful puzzle player.")

    base = (
        f"{role_desc}\n\n"
        "Rules:\n"
        "1) Output exactly ONE question per turn.\n"
        "2) The question must be answerable by Yes / No / Yes and No / No Relation.\n"
        "3) Keep it short and precise (<= 25 words).\n"
        "4) Avoid repeating already-asked questions.\n"
        "5) Do not reveal chain-of-thought; output only the question text."
    )

    if mode == "few" and FEW_SHOT_EXAMPLES.get(role):
        ex = FEW_SHOT_EXAMPLES[role][0]
        qa_lines = "\n".join([f"Q: {x['question']} -> A: {x['answer']}" for x in ex.get("qa_pairs", [])])
        return (
            base
            + "\n\nMini example:\n"
            + f"Surface: {ex.get('surface', '')}\n"
            + qa_lines
            + "\nGood next question style: concise, non-redundant, axis-specific."
        )
    return base


# 2) Next-question Prompt ----------------------------------------------------

def make_question_prompt(role: str, state: Any, mode: str = "zero") -> str:
    role_goal = {
        "fact": "Focus on facts/events/entities/time/place.",
        "value": "Focus on motives/emotions/value conflicts.",
        "knowledge": "Focus on common-sense/domain-knowledge assumptions.",
    }.get(role, "Ask a useful yes/no style question.")

    host_summary = state.host_summary.strip() if state.host_summary else "(none yet)"
    history_text = _history_to_text(state)

    return (
        "You are in a situational-puzzle game. Generate the next question.\n\n"
        f"Role goal: {role_goal}\n"
        f"Round: {state.round}\n\n"
        f"Surface:\n{state.surface}\n\n"
        f"Host summary:\n{host_summary}\n\n"
        f"Recent Q&A history:\n{history_text}\n\n"
        "Output requirements:\n"
        "- Output exactly one question sentence.\n"
        "- The question must be answerable by: Yes / No / Yes and No / No Relation.\n"
        "- Do not repeat an existing question.\n"
        "- No extra commentary."
    )


# 3) Final-answer Prompt -----------------------------------------------------

def make_final_answer_prompt(state: Any) -> str:
    history_text = _history_to_text(state, limit=9999)
    host_summary = state.host_summary.strip() if state.host_summary else "(none)"
    return (
        "The game is over. Reconstruct the most plausible hidden story (Bottom).\n\n"
        f"Surface:\n{state.surface}\n\n"
        f"Host summary:\n{host_summary}\n\n"
        f"Full Q&A history:\n{history_text}\n\n"
        "Output format:\n"
        "1) Final explanation (4-8 sentences).\n"
        "2) Key evidence bullets (3-6 bullets, each references one Q&A clue).\n"
        "Be concrete and internally consistent."
    )


# 4) Referee Prompt ----------------------------------------------------------

def make_referee_prompt(surface: str, bottom: str, question: str) -> str:
    return (
        "You are the referee of a situational-puzzle game.\n"
        "Players can see Surface; you can see both Surface and Bottom.\n"
        "Judge the player question and respond with exactly ONE label:\n"
        "- Yes\n"
        "- No\n"
        "- Yes and No\n"
        "- No Relation\n\n"
        "Decision rules:\n"
        "- Yes: the question is supported/affirmed by the Bottom.\n"
        "- No: the question is contradicted by the Bottom.\n"
        "- Yes and No: partly true / conditional / mixed across entities or time.\n"
        "- No Relation: irrelevant, not inferable, or cannot be determined from Bottom.\n\n"
        "Return only the label, no explanation.\n\n"
        f"Surface:\n{surface}\n\n"
        f"Bottom:\n{bottom}\n\n"
        f"Question:\n{question}"
    )


# 5) Host Prompts ------------------------------------------------------------

def make_summarize_prompt(state: Any) -> str:
    history_text = _history_to_text(state)
    return (
        "You are Host D. Summarize useful clues for the next round.\n"
        "Keep it concise and operational for Player A/B/C.\n\n"
        f"Surface:\n{state.surface}\n\n"
        f"Recent Q&A:\n{history_text}\n\n"
        "Output format:\n"
        "- Confirmed clues\n"
        "- Rejected clues\n"
        "- Ambiguous/No-Relation clues\n"
        "- Suggested next exploration directions for A/B/C"
    )


def make_stop_prompt(state: Any) -> str:
    history_text = _history_to_text(state)
    return (
        "You are Host D. Decide whether information is sufficient to stop questioning.\n"
        "Output only STOP or CONTINUE.\n\n"
        f"Surface:\n{state.surface}\n\n"
        f"Host summary:\n{state.host_summary or '(none)'}\n\n"
        f"Recent Q&A:\n{history_text}\n\n"
        "Use STOP only if a coherent and specific bottom can already be reconstructed "
        "with high confidence; otherwise output CONTINUE."
    )


def make_host_final_prompt(state: Any, player_answers: dict) -> str:
    history_text = _history_to_text(state, limit=9999)
    answers = "\n".join([f"[{k}] {v}" for k, v in player_answers.items()])
    return (
        "You are Host D. Merge the three players' final answers into one best reconstruction.\n"
        "Output a single integrated explanation (4-8 sentences) and then 3-6 evidence bullets.\n\n"
        f"Surface:\n{state.surface}\n\n"
        f"History:\n{history_text}\n\n"
        f"Player finals:\n{answers}"
    )


# Few-shot store -------------------------------------------------------------
FEW_SHOT_EXAMPLES: dict = {
    "fact": [],
    "value": [],
    "knowledge": [],
}
