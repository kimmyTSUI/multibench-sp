"""
scripts/run_sp.py
运行主入口：
  读取 dataset -> 逐个 sample 跑 game loop -> 写 EpisodeLog JSONL

用法示例：
  python scripts/run_sp.py \
      --data data/test.json \
      --output logs/run_001.jsonl \
      --model gpt-4o-mini \
      --mode zero \
      --max_round 25 \
      --start 0 --end 10
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import traceback

from utils.llm_client import LLMClient
from utils.io import EpisodeLog, save_episode, ensure_dir
from data.sp_dataset import load_sp_dataset_slice
from envs.sp_env import SPEnv
from agents.player_fact import PlayerFact
from agents.player_value import PlayerValue
from agents.player_knowledge import PlayerKnowledge
from agents.host import HostAgent

# ── Prompt builders（从 prompts/sp_prompts.py 导入）──────────────────────────
from prompts.sp_prompts import (
    make_player_system_prompt,
    make_question_prompt,
    make_final_answer_prompt,
    make_referee_prompt,
    make_summarize_prompt,
    make_stop_prompt,
    make_host_final_prompt,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data",      default="data/test.json",      help="数据集路径")
    p.add_argument("--output",    default="logs/run.jsonl",       help="输出 JSONL 路径")
    p.add_argument("--model",     default="gpt-4o-mini",          help="默认 LLM 模型名称（未单独指定时生效）")
    p.add_argument("--player_model", default=None,                  help="Player A/B/C 使用的模型（默认同 --model）")
    p.add_argument("--host_model",   default=None,                  help="Host D 使用的模型（默认同 --model）")
    p.add_argument("--judge_model",  default=None,                  help="裁判 Env/Judge 使用的模型（默认同 --model）")
    p.add_argument("--api_key",   default=None,                   help="API Key（默认读 OPENAI_API_KEY/HF_TOKEN 环境变量）")
    p.add_argument("--hf_token", default=None,                   help="HuggingFace token（等价于 --api_key）")
    p.add_argument("--base_url",  default=None,                   help="OpenAI 兼容 base_url（本地模型）")
    p.add_argument("--mode",      default="zero", choices=["zero", "few"], help="prompt 模式")
    p.add_argument("--max_round", type=int, default=25,           help="最大轮数")
    p.add_argument("--start",     type=int, default=0,            help="数据集起始索引")
    p.add_argument("--end",       type=int, default=None,         help="数据集结束索引（不含）")
    return p.parse_args()


def build_components(args):
    """根据命令行参数构建所有组件。"""
    resolved_api_key = args.api_key or args.hf_token
    player_model = args.player_model or args.model
    host_model = args.host_model or args.model
    judge_model = args.judge_model or args.model

    player_client = LLMClient(
        model=player_model,
        api_key=resolved_api_key,
        base_url=args.base_url,
    )
    host_client = LLMClient(
        model=host_model,
        api_key=resolved_api_key,
        base_url=args.base_url,
    )
    judge_client = LLMClient(
        model=judge_model,
        api_key=resolved_api_key,
        base_url=args.base_url,
    )

    # 环境（裁判）
    env = SPEnv(
        judge_client=judge_client,
        answer_prompt_builder=make_referee_prompt,
        max_round=args.max_round,
    )

    # Players
    player_kwargs = dict(
        client=player_client,
        system_prompt_builder=make_player_system_prompt,
        question_prompt_builder=make_question_prompt,
        final_prompt_builder=make_final_answer_prompt,
        mode=args.mode,
    )
    players = {
        "A": PlayerFact(**player_kwargs),
        "B": PlayerValue(**player_kwargs),
        "C": PlayerKnowledge(**player_kwargs),
    }

    # Host
    host = HostAgent(
        client=host_client,
        summarize_prompt_builder=make_summarize_prompt,
        stop_prompt_builder=make_stop_prompt,
        final_prompt_builder=make_host_final_prompt,
    )

    return env, players, host, {"player": player_model, "host": host_model, "judge": judge_model}


def run_episode(env, players, host, sample):
    """跑一局游戏，返回 EpisodeLog。"""
    state = env.reset(sample)
    player_order = ["A", "B", "C"]

    while not state.done:
        order = host.choose_turn_order(state, player_order)
        for p in order:
            q = players[p].act(state)
            state, _ = env.step(state, p, q)

        host.summarize(state)
        state.round += 1

        if host.should_stop(state):
            state.done = True
            state.done_reason = "max_round_or_host_stop"

    # 收集最终答案
    final_answers = {}
    for p in player_order:
        final_answers[p] = players[p].finalize(state)
    final_answers["Host"] = host.aggregate_final(state, final_answers)

    # 构造 EpisodeLog
    log = EpisodeLog(
        sample_id=state.sample_id,
        surface=state.surface,
        bottom=state.bottom,
        history=state.to_dict()["history"],
        final_answers=final_answers,
        meta={
            "round": state.round,
            "done_reason": state.done_reason,
            "key_questions": state.key_questions,
            **sample.meta,
        },
    )
    return log


def main():
    args = parse_args()
    ensure_dir(os.path.dirname(args.output) or ".")
    samples = load_sp_dataset_slice(args.data, start=args.start, end=args.end)
    env, players, host, model_map = build_components(args)

    print(f"[run_sp] 共 {len(samples)} 个样本，输出至 {args.output}")
    print("[run_sp] 模型配置:", model_map)
    for i, sample in enumerate(samples):
        try:
            log = run_episode(env, players, host, sample)
            save_episode(log, args.output)
            print(f"  [{i+1}/{len(samples)}] sample_id={sample.index} done")
        except Exception as e:
            print(f"  [{i+1}/{len(samples)}] sample_id={sample.index} ERROR: {e}")
            traceback.print_exc()

    print("[run_sp] 完成。")


if __name__ == "__main__":
    main()
