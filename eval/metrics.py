"""
eval/metrics.py
Single-agent evaluation metrics:
- final_accuracy
- key_coverage
"""

import re
from typing import Dict, Any, List

from utils.io import EpisodeLog


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _normalize_key_questions(raw_kqs) -> List[str]:
    if not raw_kqs:
        return []
    out: List[str] = []

    def _walk(x):
        if isinstance(x, str):
            t = x.strip()
            if t and t not in out:
                out.append(t)
        elif isinstance(x, list):
            for it in x:
                _walk(it)

    _walk(raw_kqs)
    return out


def compute_final_accuracy(log: EpisodeLog) -> Dict[str, float]:
    """
    计算最终准确度（exact match）：
    若 final answer 与 bottom 标准化后完全一致记 1，否则 0。
    """
    if not log.final_answers:
        return {"final_accuracy": 0.0}

    # 单 agent 版本：取第一个 final answer
    pred = next(iter(log.final_answers.values()))
    acc = 1.0 if _normalize_text(pred) == _normalize_text(log.bottom) else 0.0
    return {"final_accuracy": acc}


def compute_key_coverage(log: EpisodeLog) -> Dict[str, Any]:
    raw_kqs = log.meta.get("key_questions", []) if log.meta else []
    kqs = _normalize_key_questions(raw_kqs)
    if not kqs:
        return {"covered": [], "coverage_rate": 0.0}

    history_text = " ".join(f"{t['question']} {t['answer']}" for t in log.history)
    covered = [
        any(word in history_text for word in kq.split() if len(word) > 1)
        for kq in kqs
    ]
    return {
        "covered": covered,
        "coverage_rate": sum(covered) / len(covered),
    }


def compute_all_metrics(log: EpisodeLog) -> Dict[str, Any]:
    metrics = {}
    metrics.update(compute_final_accuracy(log))
    metrics["key_coverage"] = compute_key_coverage(log)
    return metrics
