# gpt() 函数详解：从输入到输出

## ⭐ gpt() 函数是做什么的？

**一句话**：`gpt()` 函数接收当前的 token，返回对所有可能的下一个 token 的预测得分。

```python
logits = gpt(token_id, pos_id, keys, values)
# 输入：当前 token 的 ID
# 输出：27 个得分（每个 token 一个）
```

---

## 🔍 逐步解析 gpt() 函数

让我逐行解释这个函数（main.py:136-175）：

### 输入参数

```python
def gpt(token_id, pos_id, keys, values):
    """GPT模型前向传播：核心函数"""
```

**参数说明**：
- `token_id`: 当前 token 的 ID（0-26）
- `pos_id`: 当前位置的 ID（0-15）
- `keys`: 缓存的历史 keys（用于注意力）
- `values`: 缓存的历史 values（用于注意力）

---

### 第 1 步：获取 Embedding

```python
# Token Embedding：将 token ID 转换为向量
tok_emb = state_dict['wte'][token_id]  # 16 维向量

# Position Embedding：编码位置信息
pos_emb = state_dict['wpe'][pos_id]    # 16 维向量

# 相加：结合 token 和位置信息
x = [t + p for t, p in zip(tok_emb, pos_emb)]  # 16 维向量
```

**例子**：
```python
# 假设输入是 "ann"，当前处理第 2 个 'n'
token_id = 13  # 'n'
pos_id = 3     # 第 4 个位置（0-based）

tok_emb = state_dict['wte'][13]  # 'n' 的嵌入向量
pos_emb = state_dict['wpe'][3]   # 位置 3 的嵌入向量

x = tok_emb + pos_emb  # 16 维向量
# 这个向量包含了：
# - "当前字母是 'n'"
# - "在位置 3（第 4 个字符）"
```

---

### 第 2 步：预归一化

```python
x = rmsnorm(x)  # RMS 归一化
```

**作用**：稳定训练，防止数值过大或过小

---

### 第 3 步：Transformer 层处理

```python
for li in range(n_layer):  # 遍历每一层（本项目只有 1 层）
    # 1) 多头注意力机制
    x_residual = x  # 保存残差
    x = rmsnorm(x)
    # ... 注意力计算 ...
    x = [a + b for a, b in zip(x, x_residual)]  # 残差连接

    # 2) 前馈神经网络
    x_residual = x
    x = rmsnorm(x)
    # ... MLP 计算 ...
    x = [a + b for a, b in zip(x, x_residual)]  # 残差连接
```

**这一步做了什么？**
- 提取上下文信息（注意力）
- 进行推理处理（MLP）
- 输出：`x` (16 维向量) - 包含了所有学到的信息

---

### 第 4 步：生成预测（LM Head）

```python
# 关键！将 16 维隐藏状态映射到 27 个 token 的得分
logits = linear(x, state_dict['lm_head'])
#        ↑           ↑
#    16 维输入    16 × 27 的矩阵
#               = 27 维输出

return logits  # 27 个得分
```

**这就是预测！**

---

## 🎯 logits 是什么？

**logits = 原始预测得分（未归一化）**

```python
logits = [2.3, 0.1, -1.5, ..., 0.8]
#         0    1    2         26
#         'a'  'b'  'c'       BOS
```

**每个值的含义**：
- `logits[0] = 2.3`: 模型认为下一个是 'a' 的得分是 2.3
- `logits[1] = 0.1`: 模型认为下一个是 'b' 的得分是 0.1
- `logits[13] = -1.5`: 模型认为下一个是 'n' 的得分是 -1.5

**得分越高 = 越可能是下一个 token**

---

## 📊 完整流程可视化

```
输入：token_id = 0 ('a'), pos_id = 1
      ↓
┌─────────────────────────────────────┐
│  1. Embedding                       │
│     tok_emb = wte[0]  → [16 维]     │
│     pos_emb = wpe[1]  → [16 维]     │
│     x = tok_emb + pos_emb → [16]    │
└─────────────────────────────────────┘
      ↓
┌─────────────────────────────────────┐
│  2. RMSNorm                         │
│     x = rmsnorm(x)  → [16 维]       │
└─────────────────────────────────────┘
      ↓
┌─────────────────────────────────────┐
│  3. Transformer 层                 │
│     ┌───────────────────────────┐   │
│     │ 注意力机制                │   │
│     │ 看看前面的 token          │   │
│     └───────────────────────────┘   │
│              ↓                      │
│     ┌───────────────────────────┐   │
│     │ MLP (推理)                │   │
│     │ 处理提取的信息            │   │
│     └───────────────────────────┘   │
│              ↓                      │
│     x (16 维)                      │
└─────────────────────────────────────┘
      ↓
┌─────────────────────────────────────┐
│  4. LM Head (预测)                  │
│     logits = x @ W_lm_head          │
│            [16] @ [16×27]            │
│            = [27 维]                │
└─────────────────────────────────────┘
      ↓
输出：logits = [2.3, 0.1, -1.5, ..., 0.8]
# 27 个得分，每个 token 一个
```

---

## 💡 Transformer 里有预测吗？

**问题**：Transformer 里有预测的内容吗？

**答案**：有的！但不是"直接预测"，而是"计算表示"。

### Transformer 做了什么？

```
Transformer 层（注意力 + MLP）：
- 输入：16 维向量（当前 token + 位置）
- 输出：16 维向量（精炼后的表示）

这个 16 维向量包含了：
- "当前看到的是 'n'"
- "前面有 'a' 和 'n'"
- "这是名字的中间部分"
- "下一个字母很可能是 'a' 或 'n'"
```

**注意**：Transformer **不直接输出预测**，它只是**计算出一个好的表示**！

### 谁在做预测？

**LM Head 才是真正做预测的！**

```python
# Transformer 的输出
x (16 维向量) ← 包含了所有理解的信息

# LM Head 做预测
logits = x @ W_lm_head
#       [16 维] @ [16 × 27]
#       = [27 维] ← 27 个 token 的得分
```

**类比**：
```
Transformer = 学生读题
- 输入：题目文本
- 输出：对题目的理解（16 维表示）

LM Head = 学生写答案
- 输入：对题目的理解（16 维）
- 输出：答案（27 个选项的得分）
```

---

## 🔬 实际例子

假设输入是 "ann"，当前处理第 2 个 'n'：

```python
# 步骤 1：Embedding
x = ['n' 的向量] + [位置 3 的向量]
# [0.5, -0.3, 0.8, ..., 0.2]

# 步骤 2：Transformer 处理
x = transformer_layer(x)
# [0.23, -0.15, 0.67, ..., 0.45]
# 这个向量现在包含了：
# - "前面有 'a', 'n', 'n'"
# - "名字可能是 'anna'"
# - "下一个字母很可能是 'a'"

# 步骤 3：LM Head 预测
logits = x @ W_lm_head
# [2.3, 0.1, -1.5, ..., 0.8]
#  'a'  'b'  'c'      'BOS'

# 步骤 4：Softmax 转概率
probs = softmax(logits)
# [0.35, 0.02, 0.01, ..., 0.25]
#  'a'  'b'  'c'      'BOS'
# ↑
# 概率最高！

# 预测：下一个字母是 'a'（概率 35%）
```

---

## 🎓 总结

| 组件 | 输入 | 输出 | 作用 |
|------|------|------|------|
| **Embedding** | token ID | 16 维向量 | 将 ID 转换为向量 |
| **Transformer** | 16 维向量 | 16 维向量 | 理解上下文，计算表示 |
| **LM Head** | 16 维向量 | 27 维得分 | 预测下一个 token |
| **Softmax** | 27 维得分 | 27 维概率 | 转换成概率分布 |

**核心要点**：

1. **Transformer 不直接预测**：它只是计算一个好的表示
2. **LM Head 做预测**：将表示转换成预测得分
3. **logits 不是概率**：是原始得分，需要 softmax 转成概率
4. **整个流程是可微分的**：可以端到端训练

**记忆公式**：
```
gpt() 函数 = 完整的预测流程

Embedding: ID → 向量
Transformer: 向量 → 更好的向量（理解）
LM Head: 向量 → 得分（预测）
Softmax: 得分 → 概率（采样）
```

---

## 快速问答

**Q: Transformer 的输出是什么？**
A: 16 维向量，是对当前上下文的理解

**Q: LM Head 的输出是什么？**
A: 27 维得分，是每个 token 的预测得分

**Q: logits 是概率吗？**
A: 不是！logits 是原始得分，需要 softmax 转成概率

**Q: 谁在做预测？**
A: LM Head 做预测，Transformer 提供理解

**Q: 整个流程是可训练的吗？**
A: 是的！从 embedding 到 LM Head，所有参数都可以训练
