"""
scripts/evaluate_sp.py
评估脚本：
  读取 EpisodeLog JSONL -> 计算指标 -> 输出汇总（JSON + 打印）

用法示例：
  python scripts/evaluate_sp.py \
      --input logs/run_001.jsonl \
      --output logs/metrics_001.json
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
    p.add_argument("--input",  required=True, help="EpisodeLog JSONL 路径")
    p.add_argument("--output", default=None,  help="指标输出 JSON 路径（可选）")
    return p.parse_args()


def aggregate(all_metrics):
    """对所有 episode 的指标做宏平均。"""
    agg = {}
    keys = ["A_f1_char", "A_f1_word", "B_f1_char", "B_f1_word",
            "C_f1_char", "C_f1_word"]
    for k in keys:
        vals = [m[k] for m in all_metrics if k in m]
        agg[k] = mean(vals) if vals else 0.0

    # key coverage 宏平均
    cov_vals = [m["key_coverage"].get("coverage_rate", 0.0)
                for m in all_metrics
                if "key_coverage" in m and "error" not in m["key_coverage"]]
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
