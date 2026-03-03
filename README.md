# MultiBench SP — 4-Agent Interaction Framework

多角色协作「情境推理（Situational Puzzle）」评估框架。  
**Host D + Player A/B/C** 以 interaction loop 方式组织，支持可扩展的 Prompt 自定义与多指标评估。

---

## 目录结构

```
multibench/
├── agents/
│   ├── base.py              # BaseAgent 抽象基类
│   ├── player_fact.py       # Player A（事实维度）
│   ├── player_value.py      # Player B（价值/情感维度）
│   ├── player_knowledge.py  # Player C（背景知识维度）
│   └── host.py              # Host D（调度 & 汇总）
├── envs/
│   ├── sp_state.py          # SPState / QATurn 数据结构
│   └── sp_env.py            # 游戏环境（裁判 + step/reset）
├── prompts/
│   └── sp_prompts.py        # ⭐ Prompt Registry（请在此文件填写 prompt）
├── data/
│   └── sp_dataset.py        # 数据加载（test.json -> SPSample）
├── eval/
│   ├── judges.py            # 评估 Judge prompt 工具
│   └── metrics.py           # F1 / 覆盖率 / Axis / 收敛曲线
├── scripts/
│   ├── run_sp.py            # 运行游戏 -> 写 EpisodeLog JSONL
│   └── evaluate_sp.py       # 读 JSONL -> 输出指标汇总
└── utils/
    ├── llm_client.py        # LLM 统一调用接口（OpenAI 兼容）
    └── io.py                # EpisodeLog 数据结构 + JSONL 读写
```

---

## 快速开始

### 1. 安装依赖

```bash
pip install openai
```

### 2. 填写 Prompt（必须）

打开 `prompts/sp_prompts.py`，实现以下函数：

| 函数 | 说明 |
|---|---|
| `make_player_system_prompt(role, mode)` | A/B/C 的角色 system prompt |
| `make_question_prompt(role, state, mode)` | 生成下一个 Yes/No 问题的 user prompt |
| `make_final_answer_prompt(state)` | 游戏结束时的最终解释 prompt |
| `make_referee_prompt(surface, bottom, question)` | 裁判回答 Yes/No/Unknown 的 prompt |
| `make_summarize_prompt(state)` | Host 汇总线索的 prompt |
| `make_stop_prompt(state)` | （可选）Host 智能停止判断 |
| `make_host_final_prompt(state, player_answers)` | （可选）Host 最终整合 |

### 3. 运行游戏

```bash
export OPENAI_API_KEY=sk-...

python scripts/run_sp.py \
    --data data/test.json \
    --output logs/run_001.jsonl \
    --model gpt-4o-mini \
    --mode zero \
    --max_round 25 \
    --start 0 --end 5   # 先跑 5 条测试
```

### 4. 评估

```bash
python scripts/evaluate_sp.py \
    --input logs/run_001.jsonl \
    --output logs/metrics_001.json
```

---

## 数据格式（test.json）

每条样本字段：

| 字段 | 说明 |
|---|---|
| `surface` | 谜面（Players 可见） |
| `bottom` | 谜底真相（仅 Env/Judge 可见） |
| `key_questions` / `story_tree` | AR-Bench key questions |
| `supernatural` / `someone_dies` | 元信息 |

---

## 游戏循环（伪代码）

```python
state = env.reset(sample)
while not state.done:
    for p in host.choose_turn_order(state, ['A','B','C']):
        q = players[p].act(state)          # 生成问题
        state, a = env.step(state, p, q)   # 裁判回答
    host.summarize(state)                  # 更新 host_summary
    state.round += 1
    if host.should_stop(state):
        state.done = True
final_answers = {p: players[p].finalize(state) for p in ['A','B','C']}
final_answers['Host'] = host.aggregate_final(state, final_answers)
save_episode(log, output_path)
```

---

## 评估指标

- **F1_char / F1_word**：final_answer 与谜底的字/词级 F1
- **Key Coverage**：key questions 覆盖率（process score）
- **Axis Contribution**：各 Agent 提问分布与 Yes 率
- **Misleading Correction**：No/Unknown 后的纠正轨迹
- **Convergence Curve**：逐轮覆盖率曲线
