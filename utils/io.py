"""
utils/io.py
EpisodeLog 数据结构 + JSONL 读写工具。
"""

import json
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional
from pathlib import Path


@dataclass
class EpisodeLog:
    sample_id: int
    surface: str          # 谜面
    bottom: str           # 谜底（真相）
    history: List[Dict[str, Any]]          # 序列化后的 QATurn 列表
    final_answers: Dict[str, str]          # {'A': ..., 'B': ..., 'C': ..., 'Host': ...}
    tokens: Dict[str, int] = field(default_factory=dict)   # 各 agent token 消耗（可选）
    meta: Dict[str, Any] = field(default_factory=dict)     # 附加元信息


# ── I/O helpers ──────────────────────────────────────────────────────────────

def save_episode(log: EpisodeLog, path: str) -> None:
    """追加写入一条 EpisodeLog 到 JSONL 文件。"""
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(log), ensure_ascii=False) + "\n")


def load_episodes(path: str) -> List[EpisodeLog]:
    """从 JSONL 文件读取所有 EpisodeLog。"""
    logs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                d = json.loads(line)
                logs.append(EpisodeLog(**d))
    return logs


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)
