# 7t24 Autoresearch

一个通用的 `7×24` 自动实验系统仓库。

这个仓库不是单纯的论文笔记，也不是一次性的实验归档。它的目标是把“提出实验假设 -> 跑验证 -> 保留有效改动 -> 放弃失败方向”这件事做成可以 `7×24` 接力运行的自动研究永动机。

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

更完整一点，可以把它看成下面这张通用实验闭环图：

```text
                 +----------------------+
                 |   Goal / Metric      |
                 |   Verify / Guard     |
                 +----------+-----------+
                            |
                            v
                 +----------------------+
                 |  Read state/results  |
                 |  Read lessons/logs   |
                 +----------+-----------+
                            |
                            v
                 +----------------------+
                 |  Pick one hypothesis |
                 |  inside one family   |
                 +----------+-----------+
                            |
                            v
                 +----------------------+
                 |  Make atomic change  |
                 |  run small scout     |
                 +----------+-----------+
                            |
          +-----------------+-----------------+
          |                                   |
          v                                   v
 +----------------------+            +----------------------+
 | scout clearly wins   |            | scout weak / fails   |
 | run larger confirm   |            | discard or refine    |
 +----------+-----------+            +----------+-----------+
            |                                   |
            v                                   v
 +----------------------+            +----------------------+
 | confirm wins -> keep |            | repeated no-signal   |
 | write lesson         |            | -> pivot family      |
 +----------+-----------+            +----------+-----------+
            |                                   |
            +-----------------+-----------------+
                              |
                              v
                   +----------------------+
                   | persist state/logs   |
                   | watchdog/supervisor  |
                   | continue or stop     |
                   +----------------------+
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

7. `Automatic Direction Shift Advice`
   当 recent history 显示当前 family 已经反复无信号时，supervisor 不再只给出 `needs_human`，还会自动产出 `suggested_directions`，提示下一组更值得切换的 hypothesis families。

所以这套“永动机”的本质是：

```text
自动做实验
自动记录证据
自动保留有效改动
自动丢弃失败尝试
自动在低价值阶段停机
人工只在需要切换研究方向时介入
```

## 其他用户的一键启动代码

最简单的方式是直接用仓库里的 shell 包装器：

```bash
git clone https://github.com/hanchihuang/7t24_autoresearch.git
cd 7t24_autoresearch

./scripts/new_task.sh \
  /abs/path/to/your_project \
  "Improve benchmark pass rate beyond 0.72" \
  benchmark_pass_rate \
  higher \
  "python3 run_eval.py --samples 30" \
  "python3 -m py_compile main.py" \
  benchmark_run_20260330 \
  "/abs/path/to/your_project/src,/abs/path/to/your_project/tests" \
  "Continue iterating on this benchmark task with scout-first protocol."
```

它会在你的项目目录下生成：

```text
/abs/path/to/your_project/autoresearch-launch.json
```

然后直接启动：

```bash
python3 scripts/autoresearch_runtime_ctl.py start --repo /abs/path/to/your_project
```

如果你更喜欢显式 Python 命令，也可以用：

```bash
python3 scripts/init_generic_task.py \
  --project-dir /abs/path/to/project \
  --output /abs/path/to/project/autoresearch-launch.json \
  --goal "Improve benchmark pass rate beyond 0.72" \
  --metric benchmark_pass_rate \
  --direction higher \
  --verify "python3 run_eval.py --samples 30" \
  --guard "python3 -m py_compile main.py" \
  --run-tag benchmark_run_20260330 \
  --scope "/abs/path/to/project/src,/abs/path/to/project/tests" \
  --prompt "Continue iterating on the benchmark task with scout-first protocol."
```

查看系统当前是否建议切换大方向：

```bash
python3 scripts/autoresearch_direction_advisor.py \
  --repo /abs/path/to/your_project
```

### 每个参数如何设置

`new_task.sh` 的参数顺序是：

```text
<project_dir> <goal> <metric> <direction> <verify> <guard> <run_tag> [scope] [prompt]
```

逐个解释如下：

- `project_dir`
  目标项目的绝对路径。
  应该填你真正要做实验的仓库根目录，例如：
  ` /home/user/my_project `

- `goal`
  用自然语言写清楚这轮实验总目标。
  应该可衡量，别写成“提升效果”这种空话。
  好例子：
  `Improve benchmark pass rate beyond 0.72`
  `Reduce p95 latency below 180ms`

- `metric`
  结果日志里的指标名字。
  建议用稳定的英文 snake_case，例如：
  `benchmark_pass_rate`
  `p95_latency_ms`
  `test_exact_match`

- `direction`
  指标优化方向，只能填：
  - `higher`
  - `lower`
  如果是准确率、通过率、召回率，就用 `higher`。
  如果是延迟、成本、错误数、内存，就用 `lower`。

- `verify`
  真正用于比较实验效果的机械验证命令。
  这是最关键的参数之一。
  建议这里放“小 scout 预算”的验证命令，而不是最贵的全量跑法。
  好例子：
  `python3 run_eval.py --samples 30`
  `pytest tests/bench -q`
  `python3 eval_rag.py --subset dev30`

- `guard`
  廉价但必要的防护命令，用来阻止明显坏改动进入下一轮。
  常见是：
  `python3 -m py_compile main.py`
  `pytest tests/smoke -q`
  `npm run lint`
  `cargo check`
  guard 应该比 verify 便宜，而且优先检查“代码没坏”。

- `run_tag`
  这轮实验的唯一标识。
  建议带日期和任务名，例如：
  `benchmark_run_20260330`
  `rag_rerank_20260330`
  `latency_tune_20260330`

- `scope`
  可选参数。
  填允许系统重点修改或关注的路径范围，多个路径用逗号分隔。
  不填时默认就是整个 `project_dir`。
  好例子：
  `"/abs/path/to/project/src,/abs/path/to/project/tests"`
  `"/abs/path/to/project/app,/abs/path/to/project/eval"`

- `prompt`
  可选参数。
  这是给 Codex 的原始任务说明，会写进 launch 文件。
  不填时默认使用 `goal`。
  建议写成带约束的操作指令，例如：
  `Continue iterating with scout-first protocol; only run full confirm after a clear gain.`

### 参数设置建议

- `goal` 决定方向，写得越具体越好。
- `metric` 决定结果日志是否清晰，尽量不要中途改名。
- `verify` 决定系统是不是在优化你真正关心的东西。
- `guard` 决定它会不会把代码改坏。
- `scope` 决定系统搜索空间，不要一上来放得太大。
- `prompt` 决定 Codex 的研究风格，可以加入：
  - 先 scout 再 confirm
  - 不要扩大搜索范围
  - 优先 reward-side
  - 禁止改某些目录

### 三组直接可用的例子

训练任务：

```bash
./scripts/new_task.sh \
  /abs/path/to/train_repo \
  "Improve dev exact match beyond 0.41" \
  dev_exact_match \
  higher \
  "python3 train_eval.py --eval-samples 30" \
  "python3 -m py_compile train.py" \
  train_run_20260330 \
  "/abs/path/to/train_repo/src,/abs/path/to/train_repo/eval" \
  "Use scout-first protocol and only spend full confirm budget after a clear gain."
```

RAG 任务：

```bash
./scripts/new_task.sh \
  /abs/path/to/rag_repo \
  "Improve answer accuracy beyond 0.68" \
  answer_accuracy \
  higher \
  "python3 eval_rag.py --split dev --limit 50" \
  "pytest tests/smoke -q" \
  rag_run_20260330 \
  "/abs/path/to/rag_repo/retrieval,/abs/path/to/rag_repo/prompts" \
  "Optimize retrieval and rerank first; avoid broad architecture changes."
```

性能任务：

```bash
./scripts/new_task.sh \
  /abs/path/to/system_repo \
  "Reduce p95 latency below 180ms" \
  p95_latency_ms \
  lower \
  "python3 bench.py --scenario api --iters 20" \
  "pytest tests/smoke -q" \
  latency_run_20260330 \
  "/abs/path/to/system_repo/service,/abs/path/to/system_repo/bench" \
  "Prioritize low-risk latency improvements and keep regression checks green."
```

## 这套系统适合哪些任务

这套系统并不只适合 `GRPO + GSM8K`。只要任务满足“能定义指标，能机械验证，能分轮次试错”，都可以接进来。

典型任务包括：

- 模型训练实验
  比如 GRPO、SFT、DPO、reward shaping、curriculum、数据配比、超参搜索。

- 推理时实验
  比如 rerank、aggregation、self-consistency、verifier 接入、temperature / top-p / candidate count 调整。

- 检索与 RAG 实验
  比如召回策略、chunking、重排、prompt 模板、索引参数、缓存策略。

- 数据与样本工程
  比如 hard negative 生成、样本过滤、切片对齐、合成数据注入、teacher-bank 选择。

- 系统指标优化
  比如延迟、吞吐、内存、成本、成功率、coverage、回归率、错误率。

- 代码与测试闭环
  比如测试通过率、lint 数量、类型错误数、benchmark 分数、bug 修复回归验证。

更抽象地说，这套系统适用于：

```text
有目标指标
有可重复验证
有候选改动空间
允许多轮试错
```

## 这套系统由什么组成

仓库当前包含两层内容：

- `codex-autoresearch` 运行时引擎：负责把“提出假设 -> 修改代码 -> 跑验证 -> keep / discard / pivot”这套循环自动化。
- `grpo_gsm8k_snapshot/`：一个具体的示例任务快照，也就是这条 `GRPO on GSM8K` 研究线，里面保留了主脚本、watchdog、launch/state/runtime 文件、lessons、results 和关键 run summary。

## 当前状态

| 项目 | 数值 |
| --- | --- |
| Baseline retained metric | `0.445` |
| Current best retained metric | `0.48` |
| Best keep iteration | `75` |
| Current recorded iteration | `88` |
| Runtime status | `needs_human` |
| 当前建议方向 | `reward-side / slice-aware shaping` |

上面这张表是仓库里附带的 `GRPO/GSM8K` 示例任务当前状态，不是系统本身的限制。系统本身是通用的，换一套 metric、verify 和 hypothesis family，就可以迁移到别的实验任务。

## 系统什么时候会改“大方向”

系统不是某次实验失败就立刻乱跳方向。只有当当前方法族已经反复验证、但仍然没有稳定信号时，才会进入 `pivot`，也就是切换改进的大方向。

典型触发条件有：

1. 连续多个实验都 `discard`
   说明不是单个参数点没调好，而是这一类方法整体没有产出。

2. 只有弱 hint，但无法复现
   某个 scout 看起来有一点提升，但邻域复查或 confirm 后消失。

3. 核心机制没有真正触发
   例如 reward path、replay path、verifier path 在日志里根本没有生效。

4. 小样本收益不能迁移到大样本确认
   30-sample scout 亮眼，但 200-sample confirm 站不住。

5. supervisor 判断继续跑已不划算
   此时系统会进入：
   - `pivot`
   - `needs_human`
   - `soft_blocked`

所以更准确地说：

```text
系统会在“当前方法族已经被试得差不多，而且没有稳定增益”时改大方向，
而不是因为某一次实验失败就乱切 family。
```

现在这一条判断不只是写在文档里，也已经接进了 supervisor：

- 连续 discard 过多
- pivot 累积过多
- recent labels 显示某个 family 已经被扫空

这些条件出现时，系统会自动在 supervisor state 里产出：

- `direction_shift.should_shift_direction`
- `direction_shift.exhausted_families`
- `direction_shift.suggested_directions`
- `direction_shift.rationales`

## 示例任务：这条 GRPO 主线现在做到哪了

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

## 如何迁移到其他实验任务

如果你要把这套系统迁移到别的任务，最少需要替换四件事：

1. `Goal`
   你想优化什么，例如 exact match、latency、pass rate、cost、coverage。

2. `Metric`
   必须有明确方向：
   - higher is better
   - lower is better

3. `Verify`
   需要有一条能机械执行的验证命令。

4. `Hypothesis families`
   需要预先划分当前允许搜索的实验家族，例如：
   - 数据侧
   - 模型侧
   - 推理侧
   - 奖励侧
   - 检索侧

迁移时最重要的不是复制某个 `GRPO` 技巧，而是保留这个研究协议：

```text
baseline
-> scout
-> confirm
-> keep / discard / pivot
-> lessons
-> next family
```

更细的任务模式可以看：

- [`docs/TASK_PATTERNS.md`](docs/TASK_PATTERNS.md)
- [`docs/NEXT_STAGE_ROUTE_MAP.md`](docs/NEXT_STAGE_ROUTE_MAP.md)

## 如何初始化一个新任务

现在仓库里已经补了通用任务模板和初始化脚本：

- [`templates/autoresearch-task.template.json`](templates/autoresearch-task.template.json)
- [`scripts/init_generic_task.py`](scripts/init_generic_task.py)
- [`scripts/new_task.sh`](scripts/new_task.sh)

最小用法示例：

```bash
python3 scripts/init_generic_task.py \
  --project-dir /abs/path/to/project \
  --output /abs/path/to/project/autoresearch-launch.json \
  --goal "Improve benchmark pass rate beyond 0.72" \
  --metric benchmark_pass_rate \
  --direction higher \
  --verify "python3 run_eval.py --samples 30" \
  --guard "python3 -m py_compile main.py" \
  --run-tag benchmark_run_20260330 \
  --scope "/abs/path/to/project/src,/abs/path/to/project/tests" \
  --prompt "Continue iterating on the benchmark task with scout-first protocol."
```

生成好 launch 文件后，就可以把它接到 runtime controller 上继续跑。

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
