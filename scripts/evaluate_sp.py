"""
scripts/evaluate_sp.py
单 agent 评估脚本：
  读取 EpisodeLog JSONL -> 计算指标 -> 输出汇总（JSON + 打印）

用法示例：
  python scripts/evaluate_sp.py \
      --input logs/run_single.jsonl \
      --output logs/metrics_single.json
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
from statistics import mean

from utils.io import load_episodes
from eval.metrics import compute_all_metrics


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="EpisodeLog JSONL 路径")
    p.add_argument("--output", default=None, help="指标输出 JSON 路径（可选）")
    return p.parse_args()


def aggregate(all_metrics):
    agg = {}
    acc_vals = [m.get("final_accuracy", 0.0) for m in all_metrics]
    agg["final_accuracy"] = mean(acc_vals) if acc_vals else 0.0

    cov_vals = [
        m.get("key_coverage", {}).get("coverage_rate", 0.0)
        for m in all_metrics
        if "key_coverage" in m
    ]
    agg["key_coverage_rate"] = mean(cov_vals) if cov_vals else 0.0
    return agg


def main():
    args = parse_args()
    logs = load_episodes(args.input)
    print(f"[evaluate_sp] 共 {len(logs)} 条 EpisodeLog")

    all_metrics = []
    for log in logs:
        m = compute_all_metrics(log)
        m["sample_id"] = log.sample_id
        all_metrics.append(m)

    agg = aggregate(all_metrics)

    print("\n=== 汇总指标 ===")
    for k, v in agg.items():
        print(f"  {k}: {v:.4f}")

    output = {
        "aggregated": agg,
        "per_episode": all_metrics,
    }

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print(f"\n[evaluate_sp] 已写入 {args.output}")


if __name__ == "__main__":
    main()
