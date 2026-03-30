# Task Patterns

这份文档说明如何把 `7t24_autoresearch` 从当前的 `GRPO + GSM8K` 示例任务迁移到其他实验任务。

## 适用前提

一个任务适合接入本系统，通常需要满足下面四个条件：

1. 有明确目标
   例如提升准确率、降低延迟、减少错误率、提高 pass rate。

2. 有机械验证
   也就是可以写成 shell 命令或脚本，不依赖人工主观判断。

3. 有可枚举的改动空间
   例如超参、模板、数据选择、reward、rerank、检索、阈值、策略切换。

4. 允许多轮试错
   不是一次性拍板，而是适合 `keep / discard / pivot` 这种渐进优化。

## 通用接入模板

无论做什么任务，都先把这几项定义清楚：

```text
Goal:
Metric:
Direction:
Verify:
Guard:
Scout budget:
Confirm budget:
Hypothesis families:
```

## 常见任务类型

### 1. 训练类实验

适用例子：

- GRPO / PPO / DPO / SFT
- reward shaping
- curriculum
- synthetic data mixing
- teacher distillation

推荐拆法：

- family A: 数据侧
- family B: reward 侧
- family C: update dynamics
- family D: verifier / value signal

常见 metric：

- exact match
- pass@k
- held-out accuracy
- pairwise accuracy

### 2. 推理时实验

适用例子：

- rerank
- aggregation
- self-consistency
- verifier-guided decoding
- temperature / top-p / candidate count

推荐拆法：

- candidate generation
- scoring
- grouping
- final selection

常见 metric：

- test exact match
- pass@k
- rerank win rate
- latency / cost tradeoff

### 3. RAG / 检索实验

适用例子：

- recall strategy
- chunking
- query rewrite
- rerank
- cache policy
- prompt template

常见 metric：

- recall@k
- answer accuracy
- grounding rate
- latency
- token cost

### 4. 数据清洗与样本工程

适用例子：

- hard negative mining
- sample filtering
- dedup
- slice balancing
- synthetic generation

常见 metric：

- downstream task score
- noisy sample ratio
- failure slice hit rate
- train/eval leakage checks

### 5. 系统优化与工程任务

适用例子：

- benchmark 提速
- 降低内存
- 降低失败率
- 降低测试回归
- 修复 flaky tests

常见 metric：

- p50 / p95 latency
- throughput
- memory footprint
- failure rate
- passed test count

## 什么时候应该 pivot

系统不应该因为一次失败就改大方向。更合理的是满足下面条件再切 family：

1. 连续多轮 `discard`
2. 只有局部 hint，但 confirm 站不住
3. 核心机制未实际生效
4. 小样本收益无法迁移
5. supervisor 认为继续跑只会浪费预算

## 一条通用运行协议

```text
建立 baseline
-> 设计一个小假设
-> 跑 scout
-> scout 有明显提升才跑 confirm
-> confirm 成功则 keep
-> confirm 失败或 scout 明显差则 discard
-> 当前 family 连续无信号则 pivot
-> 把结论写入 lessons
-> 进入下一轮
```

## 不要做的事

- 不要没有 baseline 就开始扫参数
- 不要每轮同时改很多变量
- 不要 scout 一亮眼就当成最终结论
- 不要把不同口径结果混在一起汇报
- 不要在已经 `soft_blocked` 的 family 上无脑续跑
