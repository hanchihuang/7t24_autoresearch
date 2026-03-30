# GRPO GSM8K Snapshot

这是这条 `GRPO on GSM8K` 自动研究线的工作快照。

这里保留的是“能继续接力”的最小必要集合，不是全量大文件归档。

## 包含什么

- `llama3_1_(8b)_grpo.py`
  当前主 GRPO / eval 脚本。

- `grpo_autoresearch_watchdog.py`
  本地 watchdog，会轮询 runtime 状态，并在合适时机自动重启或记录异常。

- `autoresearch-launch.json`
  启动 manifest，定义目标、metric、verify、scope、run tag 等。

- `autoresearch-state.json`
  最新状态快照。

- `autoresearch-runtime.json`
  最新 runtime 快照。

- `autoresearch-lessons.md`
  历史 lesson / pivot 记录。

- `research-results.tsv`
  每轮实验的权威日志。

- `grpo_autoresearch_progress_iter75_88.md`
  `iter75 -> iter88` 的中文进展总结。

## 关键 run summary

只保留了几个最关键的 summary：

- `gsm8k_improved/autoresearch_confirm200_masktrunc_iter75c/`
  当前 best retained：`0.48 (96/200)`

- `gsm8k_improved/autoresearch_scout_iter85_percentage_focus/`
  窄化 percentage synthetic 模板后退化到：`0.20`

- `gsm8k_improved/autoresearch_scout_iter87_difficulty_prompt_replay/`
  本地轻量 difficulty prompt replay 后退化到：`0.20`

## 当前状态

| 项目 | 数值 |
| --- | --- |
| Best retained metric | `0.48` |
| Best iteration | `75` |
| Current recorded iteration | `88` |
| Current state | `pivot` |
| Supervisor recommendation | `needs_human` |

## 现在最重要的结论

### 已经成立的

- `iter75` 把 retained 主线从 `0.475` 推到 `0.48`
- `mask_truncated_completions` 是当前最有价值的 continuation recipe

### 已经基本排除的

- 更窄的 synthetic-template sweep
- 低量 synthetic count 来回微调
- 本地轻量 replay 近似

### 下一跳

- `reward-side shaping`
- `slice-aware shaping`
- `percentage / rate_or_ratio` 失败切片定向奖励
- `verifier-informed penalty / reward`

## 如果你要继续跑

建议按这个顺序接手：

1. 先看 `autoresearch-state.json`
2. 再看 `grpo_autoresearch_progress_iter75_88.md`
3. 再看 `autoresearch-lessons.md`
4. 最后改 `llama3_1_(8b)_grpo.py`

操作原则保持不变：

- 先做 30-sample scout
- 只有明显超过 retained neighborhood，才做 200-sample confirm
- 不要把 `0.48 retained` 和局部 scout hint 混成同一个口径

## 导出来源

这个快照来自本地工作区导出，底层自动研究引擎来自本地 `codex-autoresearch` skill checkout：

- upstream skill commit: `0bb284a`
