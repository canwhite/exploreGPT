# "目标"（标签）是从哪里来的？

## ⭐ 核心概念：监督学习

**关键代码**（main.py:195）：
```python
token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
```

**一句话**：目标就是**输入序列中的下一个 token**！

---

## 🔍 详细解释

### 1️⃣ 数据准备阶段

```python
# 从数据集中取一个名字
doc = "anna"

# 编码成 token 序列
tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
#        [26,   0,  13, 13,   0,  26]
#        BOS,  'a', 'n', 'n', 'a', BOS
```

**这个序列就是我们的"训练数据"**！

---

### 2️⃣ 训练的目标是什么？

**GPT 的训练目标**：给定前面的 token，预测下一个 token

```
输入 "ann" → 预测下一个是什么？
        ↓
      答案：'a'

输入 "an" → 预测下一个是什么？
       ↓
      答案：'n'

输入 "a" → 预测下一个是什么？
      ↓
     答案：'n'
```

**关键**：答案就在原始数据中！

---

### 3️⃣ 如何构造训练样本？

```python
tokens = [BOS, 'a', 'n', 'n', 'a', BOS]
#         0    1   2   3   4   5

# 我们可以构造多个训练样本：
样本 1：输入 tokens[0] (BOS),   目标 tokens[1] ('a')
样本 2：输入 tokens[1] ('a'),  目标 tokens[2] ('n')
样本 3：输入 tokens[2] ('n'),  目标 tokens[3] ('n')
样本 4：输入 tokens[3] ('n'),  目标 tokens[4] ('a')
样本 5：输入 tokens[4] ('a'),  目标 tokens[5] (BOS)
```

**模式**：
- **输入**：当前位置的 token (`tokens[pos_id]`)
- **目标**：下一个位置的 token (`tokens[pos_id + 1]`)

---

### 4️⃣ 代码实现

```python
# main.py:194-199
for pos_id in range(n):  # 遍历每个位置
    # 关键！输入和目标都来自同一个序列
    token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
    #         ↑              ↑
    #      当前token         下一个token（目标！）

    # 前向传播：给定当前 token，预测所有可能的下一个 token
    logits = gpt(token_id, pos_id, keys, values)
    # 输出：27 个得分

    # 转成概率
    probs = softmax(logits)
    # 输出：27 个概率

    # 计算损失：模型预测的 vs 真实的
    loss_t = -probs[target_id].log()
    #            ↑
    #       真实的下一个 token 的概率
```

---

## 📊 可视化训练过程

### 完整例子：名字 "anna"

```python
# 步骤 1：准备数据
doc = "anna"
tokens = [BOS, 'a', 'n', 'n', 'a', BOS]
#         0    1   2   3   4   5

# 步骤 2：构造训练样本
┌─────────┬──────────┬──────────┬─────────────────┐
│ 样本    │ 输入     │ 真实目标 │ 模型应该学到   │
├─────────┼──────────┼──────────┼─────────────────┤
│ 样本 1  │ BOS      │ 'a'      │ 序列开始是 'a'  │
│ 样本 2  │ 'a'      │ 'n'      │ 'a' 后面是 'n'  │
│ 样本 3  │ 'n'      │ 'n'      │ 'n' 后面是 'n'  │
│ 样本 4  │ 'n'      │ 'a'      │ 'n' 后面是 'a'  │
│ 样本 5  │ 'a'      │ BOS      │ 序列结束        │
└─────────┴──────────┴──────────┴─────────────────┘

# 步骤 3：训练
for pos_id in range(5):
    token_id = tokens[pos_id]      # 输入
    target_id = tokens[pos_id + 1] # 目标（下一个！）

    # 模型预测
    logits = gpt(token_id, pos_id, keys, values)
    probs = softmax(logits)

    # 计算损失
    loss = -probs[target_id].log()
    # 比如样本 2：
    # 输入 'a'，真实目标是 'n'
    # 模型预测：P('a')=0.1, P('n')=0.6, P('b')=0.05, ...
    # 损失 = -log(P('n')) = -log(0.6) ≈ 0.51
```

---

## 🎯 为什么这样设计？

### 监督学习的本质

**传统监督学习**：
```
输入：猫的照片
目标：标签"猫"（人工标注）

问题：谁标注的？人类！
```

**GPT 的监督学习**：
```
输入："ann"
目标：'a'（来自数据本身！）

问题：谁标注的？不需要人工标注！
      目标就在输入数据中
```

**这叫"自监督学习"（Self-Supervised Learning）！**

---

## 💡 类比：填空题

```
题目：完成句子 "The cat ___ the mouse."

传统监督学习：
- 需要人工标注答案："ate"
- 昂贵且耗时

自监督学习（GPT）：
- 答案在原始文本中："The cat ate the mouse"
- 自动构造训练样本：
  输入："The cat " → 目标："ate"
  输入："The cat ___" → 目标："the"
- 不需要人工标注！
```

---

## 🔬 详细步骤追踪

让我们追踪一次完整的训练步骤：

```python
# 1. 取数据
doc = "anna"
tokens = [BOS, 'a', 'n', 'n', 'a', BOS]

# 2. 遍历每个位置
for pos_id in range(5):

    # === 样本 2：pos_id = 1 ===
    token_id = tokens[1]  # 'a'
    target_id = tokens[2] # 'n' ← 这就是目标！

    # 3. 前向传播
    logits = gpt(token_id=1, ...)  # 输入 'a'
    # 输出：logits = [2.3, 0.1, -1.5, ..., 0.8]
    #              'a'  'b'  'c'      'BOS'

    # 4. 转概率
    probs = softmax(logits)
    # 输出：probs = [0.35, 0.02, 0.01, ..., 0.25]
    #              'a'  'b'  'c'      'BOS'

    # 5. 计算损失
    # 真实目标是 'n' (token_id = 13)
    # 模型给 'n' 的概率是 probs[13] = 0.28
    loss = -probs[target_id].log()
         = -probs[13].log()
         = -0.28.log()
         = -(-1.27)
         = 1.27

    # 6. 反向传播
    loss.backward()
    # 梯度会告诉模型："给 'n' 的概率太低了，要提高！"
```

---

## 🎓 总结

| 问题 | 答案 |
|------|------|
| **目标从哪来？** | 从输入序列本身！ |
| **具体是什么？** | 下一个位置的 token |
| **谁标注的？** | 不需要人工标注 |
| **怎么构造？** | `target_id = tokens[pos_id + 1]` |
| **为什么有效？** | 目标就在数据中 |

**核心要点**：

1. **目标就在输入数据中**：
   ```python
   tokens = [BOS, 'a', 'n', 'n', 'a', BOS]
   # 输入：tokens[pos_id]
   # 目标：tokens[pos_id + 1]
   ```

2. **不需要人工标注**：
   - 传统监督学习需要人工标注
   - GPT 的目标自动从数据中构造

3. **这叫自监督学习**：
   - 用数据的一部分预测另一部分
   - 不需要外部标签

4. **训练过程**：
   ```
   给定 "ann" → 预测下一个 → 模型说 'n'
   真实答案 → 'a' → 不匹配！
   计算损失 → 反向传播 → 调整参数
   ```

**记忆公式**：
```
目标 = 下一个 token

tokens = [BOS, 'a', 'n', 'n', 'a', BOS]

位置 0：输入 BOS，  目标 'a'
位置 1：输入 'a',  目标 'n'
位置 2：输入 'n',  目标 'n'
位置 3：输入 'n',  目标 'a'
位置 4：输入 'a',  目标 BOS

目标 = 输入序列向右平移一位！
```

---

## 快速问答

**Q: 目标是从外部来的吗？**
A: 不是！目标就在输入序列中

**Q: 需要人工标注吗？**
A: 不需要！自动从数据构造

**Q: 为什么这样设计？**
A: 因为语言模型的任务就是"预测下一个token"

**Q: 这叫什么学习？**
A: 自监督学习（Self-Supervised Learning）

**Q: 为什么要用 `tokens[pos_id + 1]`？**
A: 因为下一个 token 就在序列的下一个位置！
