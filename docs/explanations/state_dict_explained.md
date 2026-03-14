# state_dict 完整解析

## 1. 什么是 state_dict？

**state_dict** 是一个 Python 字典，存储神经网络的所有可学习参数。

```python
state_dict = {
    'wte': [[...], [...], ...],      # 权重矩阵
    'wpe': [[...], [...], ...],      # 权重矩阵
    'layer0.attn_wq': [[...], ...],  # 权重矩阵
    ...
}
```

## 2. 完整结构

```
state_dict (字典)
│
├─ 嵌入层
│   ├─ 'wte'                    → 27×16 矩阵 (432 参数)
│   │   └─ Token 嵌入：将 token ID 转换为向量
│   │
│   └─ 'wpe'                    → 16×16 矩阵 (256 参数)
│       └─ 位置嵌入：编码位置信息
│
├─ Transformer 层 (layer0)
│   ├─ 自注意力模块
│   │   ├─ 'layer0.attn_wq'     → 16×16 矩阵 (256 参数)
│   │   │   └─ Query 权重：生成查询向量
│   │   │
│   │   ├─ 'layer0.attn_wk'     → 16×16 矩阵 (256 参数)
│   │   │   └─ Key 权重：生成键向量
│   │   │
│   │   ├─ 'layer0.attn_wv'     → 16×16 矩阵 (256 参数)
│   │   │   └─ Value 权重：生成值向量
│   │   │
│   │   └─ 'layer0.attn_wo'     → 16×16 矩阵 (256 参数)
│   │       └─ 输出投影：合并多头输出
│   │
│   └─ 前馈网络模块 (MLP)
│       ├─ 'layer0.mlp_fc1'     → 64×16 矩阵 (1024 参数)
│       │   └─ 扩展层：16维 → 64维
│       │
│       └─ 'layer0.mlp_fc2'     → 16×64 矩阵 (1024 参数)
│           └─ 投影层：64维 → 16维
│
└─ 输出层
    └─ 'lm_head'                → 27×16 矩阵 (432 参数)
        └─ 语言模型头：预测下一个 token

总计: 4192 个参数
```

## 3. 每个矩阵的作用

### 3.1 Token 嵌入矩阵 (wte)

```python
# 形状：27 × 16
wte = state_dict['wte']

# 作用：将 token ID 转换为向量
token_id = 12  # 'm' 的 token ID
tok_emb = wte[token_id]  # 取出第12行
# 结果：[0.13, 0.054, 0.242, ..., -0.02] (16维向量)

# 类比：
# wte 就像一本"字典"，每个 token 有一个唯一的向量表示
```

**实际数值**：

```
token 0 (BOS):  [0.130, 0.054, 0.242, -0.061, ...]
token 1 ('a'):  [0.125, 0.239, 0.082, 0.218, ...]
token 2 ('b'):  [-0.089, -0.063, -0.320, -0.072, ...]
...
token 12 ('m'): [?, ?, ?, ...]  # 第12行
...
token 26 ('z'): [?, ?, ?, ...]  # 第26行
```

### 3.2 位置嵌入矩阵 (wpe)

```python
# 形状：16 × 16
wpe = state_dict['wpe']

# 作用：编码位置信息
pos_id = 2  # 第2个位置
pos_emb = wpe[pos_id]  # 取出第2行
# 结果：[-0.058, -0.176, -0.726, ..., 0.04] (16维向量)

# 类比：
# wpe 给每个位置一个"位置编码"，让模型知道"这是第几个字"
```

**实际数值**：

```
位置 0: [-0.058, -0.176, -0.726, -0.013, ...]
位置 1: [-0.089, -0.063, -0.320, -0.072, ...]
位置 2: [-0.058, -0.176, -0.726, -0.013, ...]
...
位置 15: [?, ?, ?, ...]
```

### 3.3 Query 权重矩阵 (attn_wq)

```python
# 形状：16 × 16
attn_wq = state_dict['layer0.attn_wq']

# 作用：生成 Query 向量
x = [0.12, -0.19, 0.27, ...]  # 输入向量 (16维)
q = linear(x, attn_wq)        # 矩阵乘法
# 结果：[-0.018, -0.04, -0.01, ...] (16维 Query 向量)

# 实际数值示例：
attn_wq = [
    [-0.134, 0.15, 0.069, -0.043, ...],  # 第1行
    [-0.017, -0.231, -0.139, -0.301, ...], # 第2行
    ...
]
```

### 3.4 Key 权重矩阵 (attn_wk)

```python
# 形状：16 × 16
attn_wk = state_dict['layer0.attn_wk']

# 作用：生成 Key 向量
k = linear(x, attn_wk)
# 结果：[0.015, 0.051, 0.011, ...] (16维 Key 向量)
```

### 3.5 Value 权重矩阵 (attn_wv)

```python
# 形状：16 × 16
attn_wv = state_dict['layer0.attn_wv']

# 作用：生成 Value 向量
v = linear(x, attn_wv)
# 结果：[-0.023, 0.045, 0.027, ...] (16维 Value 向量)
```

### 3.6 输出投影矩阵 (attn_wo)

```python
# 形状：16 × 16
attn_wo = state_dict['layer0.attn_wo']

# 作用：合并多头注意力的输出
x_attn = [...]  # 多头输出 (16维)
output = linear(x_attn, attn_wo)
# 结果：投影后的输出 (16维)
```

### 3.7 MLP 扩展层 (mlp_fc1)

```python
# 形状：64 × 16
mlp_fc1 = state_dict['layer0.mlp_fc1']

# 作用：扩展维度 (16 → 64)
x = [...]  # 输入 (16维)
h = linear(x, mlp_fc1)
# 结果：[...] (64维，扩展了4倍)

# 为什么扩展？
# 增加模型容量，学习更复杂的特征
```

### 3.8 MLP 投影层 (mlp_fc2)

```python
# 形状：16 × 64
mlp_fc2 = state_dict['layer0.mlp_fc2']

# 作用：恢复维度 (64 → 16)
h = [...]  # 扩展后的向量 (64维)
output = linear(h, mlp_fc2)
# 结果：[...] (16维，恢复原始维度)
```

### 3.9 语言模型头 (lm_head)

```python
# 形状：27 × 16
lm_head = state_dict['lm_head']

# 作用：预测下一个 token
x = [...]  # 最后一层的输出 (16维)
logits = linear(x, lm_head)
# 结果：[...] (27维，每个 token 的分数)

# 转换为概率
probs = softmax(logits)
# 结果：[0.01, 0.02, 0.95, ...] (27个概率)
# 预测：概率最高的 token
```

## 4. 数据流动全过程

```
输入："emma"
    ↓
【嵌入层】
    ├─ Token 嵌入: wte[12] → [0.13, 0.054, ...]  (token 'm')
    └─ 位置嵌入: wpe[2] → [-0.058, -0.176, ...]  (位置 2)
    └─ 相加: [0.072, -0.122, ...] (16维)
    ↓
【Transformer 层】
    ├─ 自注意力
    │   ├─ Query: linear(x, attn_wq) → [q0, q1, ..., q15]
    │   ├─ Key:   linear(x, attn_wk) → [k0, k1, ..., k15]
    │   ├─ Value: linear(x, attn_wv) → [v0, v1, ..., v15]
    │   ├─ 注意力计算
    │   └─ 输出投影: linear(x_attn, attn_wo)
    │
    └─ 前馈网络
        ├─ 扩展: linear(x, mlp_fc1) → 64维
        ├─ 激活: ReLU
        └─ 投影: linear(h, mlp_fc2) → 16维
    ↓
【输出层】
    └─ 语言模型头: linear(x, lm_head) → 27维 logits
    └─ Softmax: [0.01, 0.02, 0.95, ...]
    └─ 预测: token 12 ('m') → 概率 95%
```

## 5. 代码对应关系

### 5.1 创建参数

```python
# main.py:106-115
state_dict = {
    'wte': matrix(vocab_size, n_embd),  # 创建 27×16 矩阵
    'wpe': matrix(block_size, n_embd),  # 创建 16×16 矩阵
    'lm_head': matrix(vocab_size, n_embd)
}

for i in range(n_layer):
    state_dict[f'layer{i}.attn_wq'] = matrix(n_embd, n_embd)  # 16×16
    state_dict[f'layer{i}.attn_wk'] = matrix(n_embd, n_embd)  # 16×16
    state_dict[f'layer{i}.attn_wv'] = matrix(n_embd, n_embd)  # 16×16
    state_dict[f'layer{i}.attn_wo'] = matrix(n_embd, n_embd)  # 16×16
    state_dict[f'layer{i}.mlp_fc1'] = matrix(4 * n_embd, n_embd)  # 64×16
    state_dict[f'layer{i}.mlp_fc2'] = matrix(n_embd, 4 * n_embd)  # 16×64
```

### 5.2 使用参数

```python
# main.py:140-142 - 嵌入层
tok_emb = state_dict['wte'][token_id]  # 取出 token 嵌入
pos_emb = state_dict['wpe'][pos_id]    # 取出位置嵌入
x = [t + p for t, p in zip(tok_emb, pos_emb)]

# main.py:149-151 - 注意力层
q = linear(x, state_dict[f'layer{li}.attn_wq'])
k = linear(x, state_dict[f'layer{li}.attn_wk'])
v = linear(x, state_dict[f'layer{li}.attn_wv'])

# main.py:167 - 输出投影
x = linear(x_attn, state_dict[f'layer{li}.attn_wo'])

# main.py:172-174 - MLP 层
x = linear(x, state_dict[f'layer{li}.mlp_fc1'])
x = [xi.relu() for xi in x]
x = linear(x, state_dict[f'layer{li}.mlp_fc2'])

# main.py:177 - 输出层
logits = linear(x, state_dict['lm_head'])
```

## 6. 参数更新过程

### 6.1 展平参数

```python
# main.py:116
params = [p for mat in state_dict.values() for row in mat for p in row]

# 拆解：
params = []
for mat in state_dict.values():      # 遍历每个矩阵
    for row in mat:                  # 遍历每行
        for p in row:                # 遍历每个元素
            params.append(p)         # 添加到参数列表

# 结果：4192 个 Value 对象的列表
```

### 6.2 反向传播

```python
# main.py:210
loss.backward()

# 效果：计算每个参数的梯度
# 例如：
# state_dict['layer0.attn_wq'][0][0].grad = 0.023
# state_dict['layer0.attn_wq'][0][1].grad = -0.015
# ...
```

### 6.3 参数更新

```python
# main.py:214-220
for i, p in enumerate(params):
    # Adam 优化器更新
    m[i] = beta1 * m[i] + (1 - beta1) * p.grad
    v[i] = beta2 * v[i] + (1 - beta2) * p.grad ** 2
    m_hat = m[i] / (1 - beta1 ** (step + 1))
    v_hat = v[i] / (1 - beta2 ** (step + 1))
    p.data -= lr_t * m_hat / (v_hat ** 0.5 + eps_adam)
    p.grad = 0
```

## 7. 可视化矩阵维度

```
嵌入层:
wte      [27×16]  ═════════════════════════════════════
wpe      [16×16]  ════════════════════════════════════

注意力层:
attn_wq  [16×16]  ════════════════════════════════════
attn_wk  [16×16]  ════════════════════════════════════
attn_wv  [16×16]  ════════════════════════════════════
attn_wo  [16×16]  ════════════════════════════════════

前馈网络:
mlp_fc1  [64×16]  ══════════════════════════════════════════════════════════════════════════════════
mlp_fc2  [16×64]  ══════════════════════════════════════════════════════════════════════════════════

输出层:
lm_head  [27×16]  ═════════════════════════════════════
```

## 8. 关键要点

| 要点 | 说明 |
|------|------|
| **存储位置** | state_dict 是一个字典 |
| **键的命名** | 'layer{层数}.{参数类型}' |
| **矩阵形状** | 由模型配置决定 (n_embd, vocab_size 等) |
| **参数类型** | 所有元素都是 Value 对象 |
| **初始化** | 高斯分布 N(0, 0.08) |
| **更新方式** | 反向传播 + Adam 优化器 |
| **总数** | 4192 个参数 |

## 9. 常见问题

### Q1: 为什么叫 state_dict？

**答**：state（状态）+ dict（字典），存储模型的"状态"（即所有参数）。

### Q2: 为什么用 f-string 格式？

```python
key = f'layer{li}.attn_wq'

# 支持多层 Transformer
# li=0 → 'layer0.attn_wq'
# li=1 → 'layer1.attn_wq'
# li=2 → 'layer2.attn_wq'
```

### Q3: 为什么有些矩阵是 16×16，有些是 64×16？

- **16×16**：保持维度不变（Query、Key、Value、输出投影）
- **64×16**：扩展维度（MLP 扩展层）
- **16×64**：恢复维度（MLP 投影层）

### Q4: 如何保存和加载 state_dict？

```python
# 保存
import pickle
with open('model.pkl', 'wb') as f:
    pickle.dump({k: [[v.data for v in row] for row in mat] 
                 for k, mat in state_dict.items()}, f)

# 加载
with open('model.pkl', 'rb') as f:
    loaded = pickle.load(f)
    state_dict = {k: [[Value(v) for v in row] for row in mat] 
                  for k, mat in loaded.items()}
```

## 10. 总结

**state_dict 是模型的"大脑"，存储所有可学习的知识。**

- **创建**：随机初始化（高斯分布）
- **使用**：前向传播时参与计算
- **更新**：反向传播计算梯度，优化器更新参数
- **保存**：持久化训练好的模型

**一句话总结**：state_dict 是一个字典，存储所有权重矩阵，是神经网络的核心数据结构。
