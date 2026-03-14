# TTT.py vs micro.py 主要区别

## 概述

| 方面           | micro.py (标准 GPT) | ttt.py (TTT)                    |
| -------------- | ------------------- | ------------------------------- |
| **训练阶段**   | 1 个阶段：预训练    | 2 个阶段：预训练 + TTT推理      |
| **推理时参数** | 完全冻结            | 所有参数可更新                  |
| **推理开销**   | O(1) 前向传播       | O(n × ttt_steps) 前向+反向+更新 |
| **优化参数量** | 0                   | 4192 (全部)                     |

## 核心代码差异

### micro.py: 标准推理

```python
for sample_idx in range(20):
    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    token_id = BOS
    sample = []
    for pos_id in range(block_size):
        logits = gpt(token_id, pos_id, keys, values)  # 仅前向传播
        probs = softmax([l / temperature for l in logits])
        token_id = random.choices(range(vocab_size), weights=[p.data for p in probs])[0]
        if token_id == BOS:
            break
        sample.append(uchars[token_id])
    print(f"sample {sample_idx+1:2d}: {''.join(sample)}")
```

### ttt.py: TTT 推理

```python
# 1. 新增：TTT 配置
ttt_learning_rate = 0.001  # 更小的学习率
ttt_steps = 5              # 每个样本的训练步数

# 2. 新增：独立的 TTT 优化器状态
ttt_m = [0.0] * len(params)
ttt_v = [0.0] * len(params)

# 3. 新增：参数快照/恢复（全部参数）
def save_params():
    return [p.data for p in params]

def restore_params(snapshot):
    for p, val in zip(params, snapshot):
        p.data = val

# 4. 新增：TTT 训练步骤
def ttt_train_step(tokens):
    # 前向传播
    # 计算损失
    loss.backward()
    # 使用 Adam 更新所有参数
    for i, p in enumerate(params):
        ttt_m[i] = ttt_beta1 * ttt_m[i] + (1 - ttt_beta1) * p.grad
        ttt_v[i] = ttt_beta2 * ttt_v[i] + (1 - ttt_beta2) * p.grad ** 2
        p.data -= ttt_learning_rate * ttt_m[i] / (ttt_v[i] ** 0.5 + ttt_eps)
        p.grad = 0

# 5. 新增：TTT 生成流程
def ttt_generate_with_context(context_tokens, num_generate=1):
    param_snapshot = save_params()       # 保存所有参数

    # TTT: 使用上下文训练
    for ttt_step in range(ttt_steps):
        ttt_loss = ttt_train_step(context_tokens)

    # 使用 TTT 调整后的参数生成
    generated_samples = []
    # ... 生成逻辑

    restore_params(param_snapshot)       # 恢复参数
    return generated_samples
```

## 流程对比

### micro.py 流程

```
┌─────────────┐     ┌─────────────┐
│  预训练      │ ──→ │  推理(冻结)  │
│  所有参数    │     │  仅前向传播  │
└─────────────┘     └─────────────┘
```

### ttt.py 流程

```
┌─────────────┐     ┌──────────────────────────────────────┐
│  预训练      │ ──→ │  TTT 推理                             │
│  所有参数    │     │  ┌────────────────────────────────┐ │
└─────────────┘     │  │ 保存全部参数快照               │ │
                    │  │ ↓                              │ │
                    │  │ TTT训练(5步, 更新所有参数)      │ │
                    │  │ ↓                              │ │
                    │  │ 生成样本                       │ │
                    │  │ ↓                              │ │
                    │  │ 恢复参数                       │ │
                    │  └────────────────────────────────┘ │
                    └──────────────────────────────────────┘
```

## TTT 设计原理

### 1. 更新所有参数

最大化适应能力，让模型能根据当前输入全面调整：

```
参数层         TTT 状态    更新策略
───────────────────────────────────
wte (嵌入)     可训练      Adam 更新
wpe (位置)     可训练      Adam 更新
attn_*         可训练      Adam 更新
mlp_*          可训练      Adam 更新
lm_head        可训练      Adam 更新
```

### 2. 更小学习率

使用 0.001 学习率（vs 预训练的 0.01），避免破坏预训练知识：

```python
# 预训练学习率
learning_rate = 0.01

# TTT 学习率（10x 更小）
ttt_learning_rate = 0.001
```

### 3. 多步训练

每个样本进行 5 步训练，充分适应当前输入：

```python
for ttt_step in range(ttt_steps):  # ttt_steps = 5
    ttt_loss = ttt_train_step(context_tokens)
```

### 4. 参数快照恢复

让每个样本独立适应，避免样本间干扰：

```python
param_snapshot = save_params()       # 记录当前状态
for _ in range(ttt_steps):
    ttt_train_step(context_tokens)   # 针对当前输入训练
generate_sample()                    # 使用调整后参数生成
restore_params(param_snapshot)       # 恢复到原始状态
```

## 运行结果对比

### micro.py 输出

```
num docs: 32033
vocab size: 27
num params: 4192
step 1000 / 1000 | loss 2.1234
--- inference (new, hallucinated names) ---
sample  1: anari
sample  2: molina
...
```

### ttt.py 输出

```
num docs: 32033
vocab size: 27
num params: 4192

=== Phase 1: Pre-training ===
step 1000 / 1000 | loss 2.6497

=== Phase 2: Test Time Training (TTT) Inference ===
TTT config: lr=0.001, steps=5 per sample

sample  1 (TTT from BOS): ara
sample  2 (TTT from BOS): ara
sample  3 (TTT from BOS): eyll
...

=== TTT Complete ===
Total params: 4192
TTT added minimal overhead: 5 gradient steps per inference
```

## TTT vs TTO vs micro 对比

| 特性     | micro.py | ttt.py              | tto.py              |
| -------- | -------- | ------------------- | ------------------- |
| 训练方式 | 预训练   | 预训练 + 推理时训练 | 预训练 + 推理时优化 |
| 优化参数 | 0        | **4192 (全部)**     | 432 (仅 lm_head)    |
| 优化器   | -        | Adam                | SGD + momentum      |
| 计算开销 | 低       | **最高**            | 中等                |
| 适应能力 | 无       | **最强**            | 中等                |
| 效率     | 最高     | 最低                | 9.7x 于 TTT         |

## 三种方法的权衡

```
适应能力
    ↑
    │                    ★ TTT (最强)
    │                   /
    │                  /
    │              ★ TTO (中等)
    │             /
    │            /
    │       ★ micro (无)
    │
    └──────────────────────────→ 计算效率
         低                    高
```

| 场景           | 推荐方法 | 原因                    |
| -------------- | -------- | ----------------------- |
| 通用生成任务   | micro.py | 无需适应，效率最高      |
| 需要强适应能力 | ttt.py   | 全参数更新，适应最强    |
| 平衡效率与适应 | tto.py   | 9.7x 效率，仍有适应能力 |

## 总结

TTT 的核心思想是：**在推理阶段继续训练模型**。

- **优点**：最大化适应能力，模型能根据输入动态调整
- **缺点**：计算开销大（每个样本 5 步梯度更新）
- **适用**：需要模型高度适应特定输入的场景
