"""
eval/metrics.py
评估指标集中管理：F1、key question 覆盖、Axis contribution、
误导-纠正轨迹、收敛曲线。所有指标以 EpisodeLog 为输入。
"""

import re
from typing import Dict, Any, List, Optional
from collections import Counter

from utils.io import EpisodeLog


# ─────────────────────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────────────────────

def _tokenize_zh(text: str) -> List[str]:
    """简单中文字级分词（按字拆分，去标点）。"""
    return list(re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9]", "", text))


def _f1_score(pred: str, ref: str) -> Dict[str, float]:
    """计算字级 F1（F1_char）和词级 F1（F1_word，空格分词）。"""
    pred_chars = _tokenize_zh(pred)
    ref_chars  = _tokenize_zh(ref)
    pred_words = pred.split()
    ref_words  = ref.split()

    def token_f1(pred_toks, ref_toks):
        if not pred_toks or not ref_toks:
            return 0.0
        common = Counter(pred_toks) & Counter(ref_toks)
        n_common = sum(common.values())
        if n_common == 0:
            return 0.0
        p = n_common / len(pred_toks)
        r = n_common / len(ref_toks)
        return 2 * p * r / (p + r)

    return {
        "f1_char": token_f1(pred_chars, ref_chars),
        "f1_word": token_f1(pred_words, ref_words),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 1. F1（字 / 词）
# ─────────────────────────────────────────────────────────────────────────────

def compute_f1(log: EpisodeLog) -> Dict[str, float]:
    """
    对 A/B/C/Host 的 final_answers 分别与 log.bottom（谜底）计算 F1。
    返回：{'A_f1_char': ..., 'A_f1_word': ..., 'B_...', ...}
    """
    result = {}
    for agent, answer in log.final_answers.items():
        scores = _f1_score(pred=answer, ref=log.bottom)
        result[f"{agent}_f1_char"] = scores["f1_char"]
        result[f"{agent}_f1_word"] = scores["f1_word"]
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 2. Key Question 覆盖（Process Score）
# ─────────────────────────────────────────────────────────────────────────────

def compute_key_coverage(
    log: EpisodeLog,
    judge_client=None,
    judge_prompt_builder=None,
) -> Dict[str, Any]:
    """
    按论文 process score 思路：判断在每轮 state 下能否回答 key_questions。
    
    若提供 judge_client 和 judge_prompt_builder，则调用 LLM 判断覆盖；
    否则退化为关键词匹配（基线）。

    返回：
        {
            'covered': [bool, ...],      # 每个 key_question 是否被覆盖
            'coverage_rate': float,      # 覆盖比例
            'coverage_by_round': [...],  # 每轮累计覆盖率
        }
    """
    kqs = log.meta.get("key_questions", []) if log.meta else []
    if not kqs:
        return {"covered": [], "coverage_rate": 0.0, "coverage_by_round": []}

    history_text = " ".join(
        f"{t['question']} {t['answer']}" for t in log.history
    )

    if judge_client is not None and judge_prompt_builder is not None:
        # LLM 判断（推荐）
        covered = []
        for kq in kqs:
            prompt = judge_prompt_builder(
                key_question=kq,
                history_text=history_text,
            )
            resp = judge_client.chat(
                [{"role": "user", "content": prompt}],
                temperature=0.0, max_tokens=8,
            )
            covered.append(resp.strip().lower().startswith("yes"))
    else:
        # 关键词匹配（降级）
        covered = [
            any(word in history_text for word in kq.split() if len(word) > 1)
            for kq in kqs
        ]

    # 逐轮覆盖率（简化：按问题顺序，每轮后累计）
    coverage_by_round = []
    covered_so_far = set()
    for turn in log.history:
        turn_text = turn["question"] + " " + turn["answer"]
        for i, kq in enumerate(kqs):
            if i not in covered_so_far:
                if any(w in turn_text for w in kq.split() if len(w) > 1):
                    covered_so_far.add(i)
        coverage_by_round.append(len(covered_so_far) / len(kqs))

    return {
        "covered": covered,
        "coverage_rate": sum(covered) / len(covered),
        "coverage_by_round": coverage_by_round,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. Axis Contribution（各 Agent 的提问轴分布与边际贡献）
# ─────────────────────────────────────────────────────────────────────────────

def compute_axis_contribution(log: EpisodeLog) -> Dict[str, Any]:
    """
    统计每个 agent 的提问数及 axis 分布，以及 Yes 率（粗略反映贡献质量）。
    返回：{'A': {'count': n, 'yes_rate': f, 'axis_dist': {...}}, ...}
    """
    agents = ["A", "B", "C"]
    result = {}
    for agent in agents:
        turns = [t for t in log.history if t["agent"] == agent]
        count = len(turns)
        yes_rate = sum(1 for t in turns if t["answer"] == "Yes") / count if count else 0.0
        axis_dist: Dict[str, int] = {}
        for t in turns:
            ax = t.get("axis") or "unknown"
            axis_dist[ax] = axis_dist.get(ax, 0) + 1
        result[agent] = {
            "count": count,
            "yes_rate": yes_rate,
            "axis_dist": axis_dist,
        }
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 4. 误导-纠正轨迹（No/Unknown → 纠正 Yes 的路径）
# ─────────────────────────────────────────────────────────────────────────────

def compute_misleading_correction(log: EpisodeLog) -> Dict[str, Any]:
    """
    分析 No/Unknown 后的纠正行为：
    - no_unknown_count : No/Unknown 总数
    - correction_count : No/Unknown 之后最终出现 Yes 的次数
    - unknown_to_yes   : Unknown -> Yes 转化数
    """
    turns = log.history
    no_unknown_count = sum(1 for t in turns if t["answer"] in {"No", "Unknown"})
    
    corrections = 0
    unknown_to_yes = 0
    for i, t in enumerate(turns):
        if t["answer"] in {"No", "Unknown"}:
            # 看后续是否有 Yes（同 agent 或任意 agent）
            later = turns[i + 1:]
            if any(lt["answer"] == "Yes" for lt in later):
                corrections += 1
            if t["answer"] == "Unknown":
                if any(lt["answer"] == "Yes" for lt in later[:5]):
                    unknown_to_yes += 1

    return {
        "no_unknown_count": no_unknown_count,
        "correction_count": corrections,
        "unknown_to_yes": unknown_to_yes,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5. 收敛曲线（逐轮覆盖率 & 增量）
# ─────────────────────────────────────────────────────────────────────────────

def compute_convergence_curve(log: EpisodeLog) -> Dict[str, Any]:
    """
    利用 compute_key_coverage 结果，返回逐轮覆盖率及其增量。
    coverage_by_round 需先通过 compute_key_coverage 获取。
    """
    cov_result = compute_key_coverage(log)
    curve = cov_result.get("coverage_by_round", [])
    delta = [curve[0]] + [curve[i] - curve[i - 1] for i in range(1, len(curve))]
    return {
        "coverage_t": curve,
        "delta_coverage_t": delta,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 汇总：对单条 EpisodeLog 计算全部指标
# ─────────────────────────────────────────────────────────────────────────────

def compute_all_metrics(log: EpisodeLog, judge_client=None) -> Dict[str, Any]:
    metrics = {}
    metrics.update(compute_f1(log))
    metrics["axis_contribution"] = compute_axis_contribution(log)
    metrics["misleading_correction"] = compute_misleading_correction(log)
    metrics["convergence"] = compute_convergence_curve(log)
    try:
        metrics["key_coverage"] = compute_key_coverage(log, judge_client)
    except Exception as e:
        metrics["key_coverage"] = {"error": str(e)}
    return metrics
