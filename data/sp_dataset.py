"""
data/sp_dataset.py
数据加载模块：将 test.json 统一解析为 SPSample 列表。
"""

import json
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from pathlib import Path


@dataclass
class SPSample:
    index: int
    surface: str                       # 谜面（puzzle surface）
    bottom: str                        # 谜底（真相）
    key_questions: List[str]           # AR-Bench 中的 key_question 列表
    meta: Dict[str, Any] = field(default_factory=dict)  # supernatural / someone_dies 等


# ── 递归提取 key_questions ────────────────────────────────────────────────────

def _extract_key_questions(node: Dict) -> List[str]:
    """从 story_tree 递归提取所有 key_question 字段。"""
    kqs = []
    if "key_question" in node:
        kqs.append(node["key_question"])
    for child in node.get("children", []):
        kqs.extend(_extract_key_questions(child))
    return kqs


# ── 加载函数 ──────────────────────────────────────────────────────────────────

def load_sp_dataset(path: str) -> List[SPSample]:
    """
    读取 test.json，返回 SPSample 列表。

    JSON 字段对应关系：
      surface        -> SPSample.surface
      bottom         -> SPSample.bottom
      index          -> SPSample.index
      story_tree     -> 递归提取 key_questions
      supernatural / someone_dies -> meta
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    samples = []
    for i, item in enumerate(raw):
        # 从 story_tree 中递归提取 key_questions
        story_tree = item.get("story_tree", {})
        kqs = _extract_key_questions(story_tree)

        # 如果顶层也有 key_question 字段则一并加入
        top_kq = item.get("key_question")
        if isinstance(top_kq, list):
            # 兼容数据集中 key_question 为列表的情况
            for q in reversed(top_kq):
                if isinstance(q, str) and q not in kqs:
                    kqs.insert(0, q)
        elif isinstance(top_kq, str):
            if top_kq not in kqs:
                kqs.insert(0, top_kq)

        meta = {
            k: v
            for k, v in item.items()
            if k not in {"surface", "bottom", "index", "story_tree", "key_question", "core_sentence"}
        }

        sample = SPSample(
            index=item.get("index", i),
            surface=item["surface"],
            bottom=item["bottom"],
            key_questions=kqs,
            meta=meta,
        )
        samples.append(sample)

    return samples


def load_sp_dataset_slice(path: str, start: int = 0, end: int = None) -> List[SPSample]:
    """加载数据集的子集，方便小规模测试。"""
    samples = load_sp_dataset(path)
    return samples[start:end]
