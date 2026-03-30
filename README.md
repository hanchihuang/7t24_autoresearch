# 7t24 Autoresearch

一个面向 `GRPO + GSM8K` 的自动研究仓库。

这个仓库不是单纯的论文笔记，也不是一次性的实验归档。它的目标是把一条数学推理实验线做成可以 `7×24` 接力运行的“自动研究永动机”。

## 永动机原理

这套系统的核心不是“持续训练”，而是“持续做小实验并自动决策”。

它按下面这条闭环工作：

```text
提出一个局部假设
  -> 修改代码 / 配置
  -> 跑一次小 scout 验证
  -> 如果明显变好，再花更大预算做 confirm200
  -> 根据结果决定 keep / discard / pivot
  -> 记录 lessons 和 results
  -> 再进入下一轮
```

它之所以能长期运行，不是因为它会无脑死循环，而是因为它把“研究动作”拆成了稳定的最小单元：

1. `Hypothesis`
   每次只尝试一个局部方向，比如改 reward、改 synthetic mix、改 rerank、改 verifier 接法。

2. `Scout`
   先用小预算验证方向值不值得继续，避免每次都直接烧 200-sample confirm。

3. `Confirm`
   只有 scout 明显超过 retained neighborhood，才进入更重的全量确认。

4. `Decision`
   每轮实验结束后，不是只看分数，而是明确写成：
   - `keep`
   - `discard`
   - `refine`
   - `pivot`
   - `search`

5. `Persistence`
   所有结果都写入状态文件和日志，下次重启可以继续接力，而不是重新开始。

6. `Watchdog + Supervisor`
   watchdog 负责守护运行时，supervisor 负责判断这轮是否还能继续跑；如果进入 `soft_blocked / needs_human`，就停下来等人工切换 hypothesis family，而不是浪费算力。

所以这套“永动机”的本质是：

```text
自动做实验
自动记录证据
自动保留有效改动
自动丢弃失败尝试
自动在低价值阶段停机
人工只在需要切换研究方向时介入
```

## 这套系统由什么组成

它包含两层内容：

- `codex-autoresearch` 运行时引擎：负责把“提出假设 -> 修改代码 -> 跑验证 -> keep / discard / pivot”这套循环自动化。
- `grpo_gsm8k_snapshot/`：这条 `GRPO on GSM8K` 研究线的可接力快照，带主脚本、watchdog、launch/state/runtime 文件、lessons、results 和关键 run summary。

## 当前状态

| 项目 | 数值 |
| --- | --- |
| Baseline retained metric | `0.445` |
| Current best retained metric | `0.48` |
| Best keep iteration | `75` |
| Current recorded iteration | `88` |
| Runtime status | `needs_human` |
| 当前建议方向 | `reward-side / slice-aware shaping` |

## 这条主线现在做到哪了

这条线当前已经明确分成三层结果，不能混写：

1. 公平训练主对照：`0.175 -> 0.36`
2. 本地 test-time 强化主线：`0.36 -> 0.45 -> 0.505`
3. autoresearch 当前 retained 主线：`0.445 -> 0.475 -> 0.48`

其中这套“永动机”仓库当前真正保留的最好自动研究主线是：

- `iter21`: `0.475 (95/200)`
- `iter75`: `0.48 (96/200)`

而 `iter75` 之后最新两次主要 scout 都失败了：

- `iter85 percentage_focus -> 0.20`
- `iter87 difficulty_prompt_replay -> 0.20`

所以当前停在一个很明确的结论上：

- 继续窄化 synthetic template 没有信号
- 做本地轻量 replay 近似也没有信号
- 下一跳应该转向 `reward-side / slice-aware shaping`

## 仓库结构

```text
.
├── agents/                         # codex-autoresearch agent config
├── docs/                           # upstream docs
├── references/                     # upstream protocols / invariants
├── scripts/                        # autoresearch runtime scripts
├── tests/                          # upstream tests
├── grpo_gsm8k_snapshot/
│   ├── llama3_1_(8b)_grpo.py       # 主 GRPO / eval 脚本
│   ├── grpo_autoresearch_watchdog.py
│   ├── autoresearch-launch.json
│   ├── autoresearch-state.json
│   ├── autoresearch-runtime.json
│   ├── autoresearch-lessons.md
│   ├── research-results.tsv
│   ├── grpo_autoresearch_progress_iter75_88.md
│   └── gsm8k_improved/             # 关键 run summary 快照
└── README.md
```

## 最值得先看的文件

- [`grpo_gsm8k_snapshot/grpo_autoresearch_progress_iter75_88.md`](grpo_gsm8k_snapshot/grpo_autoresearch_progress_iter75_88.md)
  这里是 `iter75 -> iter88` 的中文进展表，最适合快速理解为什么当前停在 `0.48`。

- [`grpo_gsm8k_snapshot/autoresearch-state.json`](grpo_gsm8k_snapshot/autoresearch-state.json)
  这里是最新状态快照，能直接看到 `best_metric`、`best_iteration`、`current_status` 和 supervisor 的推荐动作。

- [`grpo_gsm8k_snapshot/research-results.tsv`](grpo_gsm8k_snapshot/research-results.tsv)
  这里是每一轮实验的权威日志，包含 `keep / discard / pivot / search / refine`。

- [`grpo_gsm8k_snapshot/autoresearch-lessons.md`](grpo_gsm8k_snapshot/autoresearch-lessons.md)
  这里沉淀了每次成功与失败后的 lesson，适合做下一轮 hypothesis 选择。

- [`grpo_gsm8k_snapshot/llama3_1_(8b)_grpo.py`](grpo_gsm8k_snapshot/llama3_1_(8b)_grpo.py)
  这是当前工作主脚本，新的算法实验最终都会落到这里。

## 如何继续接力这套永动机

### 1. 先理解当前停机原因

当前不是程序崩了，而是 supervisor 明确判定为：

- `status = needs_human`
- `terminal_reason = soft_blocked`

这表示：

- 运行机制本身还在
- 当前 family 的实验空间已经被扫得差不多了
- 继续无脑自动重启大概率只会浪费算力

### 2. 当前最值得继续的方向

按当前状态文件和 lessons，下一跳应优先考虑：

- `reward-side shaping`
- `slice-aware shaping`
- `percentage-specific intermediate-step rewards`
- `verifier-informed penalties`

不建议继续做：

- 更窄的 synthetic-template sweep
- 本地轻量 replay 近似
- 只在 low-volume synthetic count 上做来回微调

### 3. 运行入口

这套仓库的引擎基于 `codex-autoresearch` 的 runtime controller。

核心文件：

- `scripts/autoresearch_runtime_ctl.py`
- `grpo_gsm8k_snapshot/autoresearch-launch.json`
- `grpo_gsm8k_snapshot/grpo_autoresearch_watchdog.py`

如果你要继续把这条线跑起来，推荐顺序是：

1. 先读 `grpo_gsm8k_snapshot/autoresearch-state.json`
2. 再读 `grpo_gsm8k_snapshot/autoresearch-lessons.md`
3. 确认新的 hypothesis family
4. 修改 `grpo_gsm8k_snapshot/llama3_1_(8b)_grpo.py`
5. 用小 scout 先过 30-sample gate
6. 只有明显超过 retained neighborhood，才花 200-sample confirm 预算

## 当前进展摘要

### 成功节点

- `iter21`: verifier-aware confirm200 到 `0.475`
- `iter75`: `mask_truncated_completions` confirm200 到 `0.48`

### 失败节点

- `iter77`: schema-archetypes 没超过 retained-control
- `iter78`: synth16 只有弱 hint，不足以 confirm
- `iter81`: step-alignment reward 没真正生效
- `iter85`: percentage-only synthetic warmup 退化到 `0.20`
- `iter87`: difficulty prompt replay 退化到 `0.20`

### 当前判断

训练和推理上的低成本局部修补已经接近跑干净了。下一步如果还想继续往上推，不是“再搜一点”，而是要把 value signal / reward shaping 做得更直接、更针对失败切片。

## 上游来源

本仓库的自动研究引擎基于本地 `codex-autoresearch` skill checkout 导出，导出时对应的上游提交是：

- upstream skill commit: `0bb284a`

## 备注

这个仓库刻意没有把大体积 checkpoint 和完整训练产物全部打包进来，只保留了：

- 运行引擎
- 主实验脚本
- 最新状态快照
- 关键结果日志
- 关键 run summary

这样仓库保持轻量，同时仍然足够支持继续接力。
