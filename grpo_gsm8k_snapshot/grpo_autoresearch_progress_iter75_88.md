# GRPO Autoresearch 进展表（iter75 -> iter88）

## 当前总状态

- `baseline_metric`: `0.445`
- `best_metric`: `0.48`
- `best_iteration`: `75`
- `current_iteration`: `88`
- `current_status`: `pivot`
- `recommended_action`: `needs_human`
- `terminal_reason`: `soft_blocked`

## 关键进展表

| Iter | 状态 | 指标 | 核心动作 | 结果解读 |
| --- | --- | --- | --- | --- |
| 75 | `keep` | `0.48 (96/200)` | 在 `iter74` scout 成功的基础上，对 `mask_truncated_completions + num_generations6 + transform_aug + synthetic_sft_only` 做 `confirm200` 全量确认 | 当前最佳 retained 主线。超过旧 retained `0.475`，并且 XML 稳定性明显恢复，成为新的 best continuation recipe。 |
| 76 | `refine` | `0.48` | 对 `iter75c` 保留下来的 adapter 做 30-sample retained control recheck | 校准新 retained 的 scout 门槛。结论是未来小实验至少要明显超过 `~0.27` 邻域，才值得花 200-sample confirm 预算。 |
| 77 | `discard` | `0.2667 (8/30)` | 做 failure-aligned hard synthetic archetypes，扩展 targeted GSM8K synthetic schema | 只打平 retained-control neighborhood，没有形成明确提升，不值得 confirm200。 |
| 78 | `discard` | `0.30 (9/30)` | 把 targeted hard synthetic warmup volume 降到 `synth16` | 比 retained-control 只高一个样本，方向上有 hint，但提升不够稳，仍不足以进入 confirm200。 |
| 79 | `discard` | `0.2667 (8/30)` | 把低量 synthetic 从 `synth16` 提回 `synth24` | XML 变好，但 exact match 回落到 retained-control 邻域，说明更高 synthetic 覆盖没有带来净增益。 |
| 80 | `pivot` | `0.48` | 放弃 post-keep 的低量 synthetic warmup sweep | 主线从“继续调低量 synthetic”切到“让 targeted hard synthetic 对 GRPO 可见，并加入 step-alignment reward”。 |
| 81 | `discard` | `0.30 (9/30)` | 让少量 targeted hard synthetic 参与 GRPO，同时启用 step-alignment reward | 结果只复现了 `iter78` 的单样本 hint，更关键的是 shaping reward 全程未真正生效，因此不能 confirm200。 |
| 82 | `discard` | `0.2667 (8/30)` | 保持 `synth16`，但把 anchor replay 从 16 恢复到 32 | 直接抹掉了 `iter78` 的轻微提升，说明之前的 hint 依赖于同步降低 anchor replay，而不只是 synthetic volume。 |
| 83 | `pivot` | `0.48` | 放弃 `synth16` neighborhood sweep | 结论是低量 synthetic count 微调这条 family 已经没有更高信号，主线转去寻找其他稳定化机制。 |
| 84 | `search` | `0.48` | 搜 replay-buffer / prompt-replay / dynamic replay 文献与 TRL 能力 | 得出下一候选 family：优先尝试 prompt replay / medium-difficulty replay，而不是继续缩窄 synthetic 模板。 |
| 85 | `discard` | `0.20 (6/30)` | 做 `percentage_focus`，把 hard synthetic warmup 收窄到单一 percentage family | 明显退化。percentage 虽是薄弱 slice，但单模板强化导致 broader GSM8K reasoning structure 受损。 |
| 86 | `search` | `0.48` | 结合 Prompt Replay 论文与当前 TRL 版本能力继续做可行性判断 | 结论是当前 `trl==0.22.2` 不直接支持 GRPO replay buffer，因此只能试本地轻量 replay 近似。 |
| 87 | `discard` | `0.20 (6/30)` | 做 `difficulty_prompt_replay`，按中等难度 prompt 做轻量 replay 近似 | 仍然明显退化。虽然找到 `pass_rate≈0.5` 的 prompt，但没有转化成 exact-match 提升，实验代码已回滚。 |
| 88 | `pivot` | `0.48` | 放弃“更窄 synthetic 模板”和“本地轻 replay 近似”这条 family | 当前自动链停在人工复盘节点。下一跳建议转向 `reward-side / slice-aware shaping`，例如 percentage-specific intermediate-step rewards 或 verifier-informed penalties。 |

## 阶段总结

### 1. 最佳 retained 主线已经从 `0.475` 升到 `0.48`

- 旧 retained：`0.475 (95/200)`
- 新 retained：`0.48 (96/200)`
- 真正带来提升的动作是 `iter75` 的 `mask_truncated_completions` 确认，而不是后续的 replay 或窄模板实验。

### 2. `iter75` 之后的主线探索基本都失败了

- `schema_archetypes` 没超过 retained-control
- `low_synth_volume` 只有弱 hint，无法稳定复现
- `grpo_visible_synth + step_alignment_reward` 的 shaping 路径没有真正触发
- `percentage_focus` 和 `difficulty_prompt_replay` 都退化到 `0.20`

### 3. 当前最合理的下一步

- 不再继续窄化 synthetic template
- 不再继续做本地轻量 replay 近似
- 直接转向 `reward-side / slice-aware shaping`
- 重点针对 `percentage`、`rate_or_ratio` 这些失败切片设计更直接的 reward / penalty

## 相关文件

- 状态文件：`/home/user/图片/autoresearch-state.json`
- 运行时文件：`/home/user/图片/autoresearch-runtime.json`
- lessons：`/home/user/图片/autoresearch-lessons.md`
- 结果日志：`/home/user/图片/research-results.tsv`
