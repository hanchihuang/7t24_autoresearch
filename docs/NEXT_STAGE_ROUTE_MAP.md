# Next Stage Route Map

这份文档针对一个典型问题：当实验从 `0.475` 只推到 `0.48`，后面大量 scout 都没有形成稳定增益时，系统下一阶段应该怎么提速。

## 先判断：为什么会慢

如果当前表现是下面这种模式：

- 小 scout 偶尔有弱 hint
- confirm 很少真正成立
- 最新 keep 提升幅度很小
- 后续多轮实验主要在同一 family 内部横跳
- supervisor 已接近 `soft_blocked / needs_human`

那通常说明：

1. 当前 family 的高价值空间已经接近耗尽
2. 你优化的是局部修补，而不是新增益源
3. 继续扩大同类搜索，只会越来越慢

## 什么时候该切大方向

系统最应该切大方向的时机，不是“有一次没涨”，而是：

- 连续多轮 `discard`
- 只有弱 hint，复查后消失
- reward / replay / verifier 路径没有真正生效
- 小样本收益不能迁移到 confirm
- 当前 family 已经出现“越搜越碎”的迹象

## 下一阶段优先级

### A 类：可能带来新台阶的方向

- `reward_side_shaping`
- `slice_aware_shaping`
- `verifier_guided_reward_or_penalty`
- `hard_slice_curriculum`
- `stronger_teacher_or_distillation`

这些方向的共同点是：

- 不是继续在原 family 内部微调
- 而是引入新的增益源

### B 类：稳健增强

- conservative verifier integration
- failure-slice monitoring
- better scout gating with intermediate metrics
- retained-mainline hardening

这些方向未必直接带来大跃迁，但能减少错误 scout，提升实验效率。

### C 类：应减少预算投入

- 低量 synthetic count 来回微调
- 更窄 template sweep
- 本地轻量 replay 近似
- 没有实际激活的 reward 变体反复重试

这类方向的共同特点是：

- 便宜，但高概率只产生碎片化 hint
- 很难形成确认级提升

## 推荐的提速策略

### 1. 缩减低价值 scout

别再把大量预算放在同一 family 的细碎变体上。

更好的做法是：

- 只保留真正新机制
- 合并相近小实验
- 优先跑能改变信号源的 family

### 2. 给 scout 增加中间判据

不要只看最终 exact match。

还应该同时监控：

- failure slice 是否改善
- verifier score 是否改善
- formatting / extraction 是否改善
- rate_or_ratio / percentage 等薄弱切片是否改善

### 3. 用 family 预算而不是单实验预算

例如：

- A 类方向，每类允许更多试错
- B 类方向，适度探索
- C 类方向，快速止损

### 4. 自动产出下一跳建议

这也是当前系统已经加上的能力：

- 当 recent history 显示某个 family 已耗尽
- 或连续 discard / pivot 过多
- supervisor 会给出 `suggested_directions`

## 一句话总结

当实验已经从 `0.475` 到 `0.48` 但再往上变得很慢时，问题通常不是“系统不够勤奋”，而是“当前 family 的增益已经被榨得差不多了”。真正的提速，不是继续扫更多同类实验，而是切到新的信号源。
