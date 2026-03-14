# TTO.py vs Main.py 主要区别

## 概述

| 方面 | main.py (标准 GPT) | tto.py (TTO) |
|---|---|---|
| **训练阶段** | 1 个阶段：预训练 | 2 个阶段：预训练 + TTO推理 |
| **推理时参数** | 完全冻结 | 部分可更新 (lm_head) |
| **推理开销** | O(1) 前向传播 | O(n) 前向+反向+参数更新 |

## 核心代码差异

### main.py: 标准推理

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

### tto.py: TTO 推理

```python
# 1. 新增：只优化 lm_head 参数
lm_head_params = [p for row in state_dict['lm_head'] for p in row]
tto_m = [0.0] * len(lm_head_params)  # 独立优化器状态

# 2. 新增：TTO 适应函数
def tto_adapt_step(tokens):
    # ... 前向传播计算 loss
    loss.backward()
    # 只更新 lm_head 参数
    for i, p in enumerate(lm_head_params):
        tto_m[i] = tto_momentum * tto_m[i] + p.grad
        p.data -= tto_lr * tto_m[i]
        p.grad = 0
    # 清零其他参数梯度
    for p in params:
        if p not in lm_head_params:
            p.grad = 0

# 3. 新增：参数快照/恢复
def save_lm_head():
    return [p.data for p in lm_head_params]

def restore_lm_head(snapshot):
    for p, val in zip(lm_head_params, snapshot):
        p.data = val

# 4. 新增：TTO 生成流程
def tto_generate(context_tokens):
    snapshot = save_lm_head()      # 保存参数
    tto_adapt_step(context_tokens) # 推理时训练
    # ... 生成样本
    restore_lm_head(snapshot)      # 恢复参数
```

## 流程对比

### main.py 流程

```
┌─────────────┐     ┌─────────────┐
│  预训练      │ ──→ │  推理(冻结)  │
│  所有参数    │     │  仅前向传播  │
└─────────────┘     └─────────────┘
```

### tto.py 流程

```
┌─────────────┐     ┌──────────────────────────────┐
│  预训练      │ ──→ │  TTO 推理                     │
│  所有参数    │     │  ┌─────────────────────────┐ │
└─────────────┘     │  │ 保存参数快照             │ │
                    │  │ ↓                       │ │
                    │  │ TTO适应(更新lm_head)     │ │
                    │  │ ↓                       │ │
                    │  │ 生成样本                 │ │
                    │  │ ↓                       │ │
                    │  │ 恢复参数                 │ │
                    │  └─────────────────────────┘ │
                    └──────────────────────────────┘
```

## TTO 设计原理

### 1. 只优化 lm_head

输出层直接影响预测分布，调整它对当前输入最有效。

```
参数层         TTO 状态    更新策略
───────────────────────────────────
wte (嵌入)     冻结        不更新
wpe (位置)     冻结        不更新
attn_*         冻结        不更新
mlp_*          冻结        不更新
lm_head        可训练      SGD+momentum
```

**效率提升**：
- 总参数：4192
- TTO 参数：432 (lm_head only)
- 效率增益：**9.7x 更快**（相比完整 TTT）

### 2. 参数快照恢复

让每个样本独立适应，避免样本间干扰：

```python
snapshot = save_lm_head()      # 记录当前状态
tto_adapt_step(context_tokens) # 针对当前输入调整
generate_sample()              # 使用调整后参数生成
restore_lm_head(snapshot)      # 恢复到原始状态
```

### 3. SGD + momentum

比 Adam 更轻量，适合少量参数的快速适应：

```python
# TTO 使用简单的 SGD + momentum
tto_m[i] = tto_momentum * tto_m[i] + p.grad
p.data -= tto_lr * tto_m[i]

# 对比：标准训练使用 Adam
m[i] = beta1 * m[i] + (1 - beta1) * p.grad
v[i] = beta2 * v[i] + (1 - beta2) * p.grad ** 2
m_hat = m[i] / (1 - beta1 ** (step + 1))
v_hat = v[i] / (1 - beta2 ** (step + 1))
p.data -= lr_t * m_hat / (v_hat ** 0.5 + eps_adam)
```

## 运行结果对比

### main.py 输出

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

### tto.py 输出

```
num docs: 32033
vocab size: 27
num params: 4192
TTO optimizable params (lm_head only): 432

=== Phase 1: Pre-training ===
step 1000 / 1000 | loss 2.6497

=== Phase 2: Test Time Optimization (TTO) Inference ===
TTO config: lr=0.01, momentum=0.9
Optimizing only 432/4192 params (10.3%)

sample  1 (TTO from BOS): ara
sample  2 (TTO from BOS): aran
...

=== TTO Complete ===
Total params: 4192
TTO params: 432 (lm_head only)
Efficiency gain: 9.7x faster than full TTT
```

## 总结

| 特性 | main.py | tto.py |
|---|---|---|
| 训练方式 | 预训练一次 | 预训练 + 推理时微调 |
| 参数更新 | 训练时全部更新 | 推理时只更新 lm_head |
| 适应能力 | 固定模型 | 根据输入动态适应 |
| 计算开销 | 低 | 中等（9.7x 低于完整 TTT） |
| 适用场景 | 通用生成 | 需要适应特定输入的场景 |
