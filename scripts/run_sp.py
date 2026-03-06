"""
scripts/run_sp.py
Single-agent SP runner:
  load dataset -> run one selected agent (A/B/C) per sample -> save EpisodeLog JSONL
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

from prompts.sp_prompts import (
    make_player_system_prompt,
    make_question_prompt,
    make_final_answer_prompt,
    make_referee_prompt,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/test.json", help="数据集路径")
    p.add_argument("--output", default="logs/run_single.jsonl", help="输出 JSONL 路径")
    p.add_argument("--agent", default="A", choices=["A", "B", "C"], help="单 agent 角色：A/B/C")

    p.add_argument("--model", default="gpt-4o-mini", help="默认 LLM 模型名称（未单独指定时生效）")
    p.add_argument("--agent_model", default=None, help="单 agent 使用模型（默认同 --model）")
    p.add_argument("--judge_model", default=None, help="裁判 Env/Judge 使用模型（默认同 --model）")

    p.add_argument("--api_key", default=None, help="API Key（默认读 OPENAI_API_KEY/OPENROUTER_API_KEY/HF_TOKEN）")
    p.add_argument("--hf_token", default=None, help="HuggingFace token（等价于 --api_key）")
    p.add_argument("--base_url", default=None, help="OpenAI 兼容 base_url")

    p.add_argument("--mode", default="zero", choices=["zero", "few"], help="prompt 模式")
    p.add_argument("--max_round", type=int, default=15, help="最大轮数（单 agent 回合数）")
    p.add_argument("--start", type=int, default=0, help="数据集起始索引")
    p.add_argument("--end", type=int, default=None, help="数据集结束索引（不含）")
    return p.parse_args()


def build_components(args):
    resolved_api_key = args.api_key or args.hf_token
    agent_model = args.agent_model or args.model
    judge_model = args.judge_model or args.model

    agent_client = LLMClient(model=agent_model, api_key=resolved_api_key, base_url=args.base_url)
    judge_client = LLMClient(model=judge_model, api_key=resolved_api_key, base_url=args.base_url)

    env = SPEnv(judge_client=judge_client, answer_prompt_builder=make_referee_prompt, max_round=args.max_round)

    player_kwargs = dict(
        client=agent_client,
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

    return env, players, {"agent": agent_model, "judge": judge_model}


def run_episode(env, player, agent_name: str, sample, args):
    state = env.reset(sample)

    while state.round < state.max_round:
        q = player.act(state)
        state, _ = env.step(state, agent_name, q)
        state.round += 1

    state.done = True
    state.done_reason = "max_round_reached"

    final_answers = {agent_name: player.finalize(state)}

    log = EpisodeLog(
        sample_id=state.sample_id,
        surface=state.surface,
        bottom=state.bottom,
        history=state.to_dict()["history"],
        final_answers=final_answers,
        meta={
            "agent": agent_name,
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
    env, players, model_map = build_components(args)

    player = players[args.agent]

    print(f"[run_sp] 单 agent={args.agent}，共 {len(samples)} 个样本，输出至 {args.output}")
    print("[run_sp] 模型配置:", model_map)
    for i, sample in enumerate(samples):
        try:
            log = run_episode(env, player, args.agent, sample, args)
            save_episode(log, args.output)
            print(f"  [{i+1}/{len(samples)}] sample_id={sample.index} done")
        except Exception as e:
            print(f"  [{i+1}/{len(samples)}] sample_id={sample.index} ERROR: {e}")
            traceback.print_exc()

    print("[run_sp] 完成。")


if __name__ == "__main__":
    main()
