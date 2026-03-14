# microGPT 精炼教程：从零理解 GPT

> 本教程专为初学者设计，通过纯 Python 实现的 GPT 模型，深入浅出地讲解 Transformer 架构和深度学习核心概念。

---

## 目录

- [第一章：预备知识](#第一章预备知识)
- [第二章：自动微分系统](#第二章自动微分系统)
- [第三章：数据预处理](#第三章数据预处理)
- [第四章：Transformer 架构](#第四章transformer-架构)
- [第五章：训练过程](#第五章训练过程)
- [第六章：推理生成](#第六章推理生成)
- [附录：数学公式推导](#附录数学公式推导)

---

## 第一章：预备知识

### 1.1 什么是深度学习？

**深度学习** 是机器学习的一个分支，它使用**神经网络**来学习数据中的模式。

#### 核心思想

想象你在教一个孩子认识动物：

- **传统编程**：你需要明确告诉程序"如果有尖耳朵和胡须，就是猫"
- **深度学习**：你给它看1000张猫的照片，它自己学会"猫长什么样"

#### 神经网络的工作原理

```
输入层                    隐藏层                    输出层
───────                  ───────                  ───────
[特征1] ──┐
[特征2] ──┼──→ [神经元1] ──┐
[特征3] ──┘               │       ┌→ [预测结果]
          └→ [神经元2] ──┼──→
                          │       └→ [置信度]
          ┌→ [神经元3] ──┘
          │
          └→ ...
```

每个神经元都是一个**数学函数**，它：

1. 接收输入
2. 计算加权和
3. 通过激活函数转换
4. 传递给下一层

### 1.2 什么是 GPT？

**GPT** = **Generative Pre-trained Transformer**

- **Generative（生成式）**: 能够生成新内容（文本、代码等）
- **Pre-trained（预训练）**: 先在大数据集上学习通用知识
- **Transformer**: 使用 Transformer 架构

#### GPT 的本质

GPT 本质上是一个**超级强大的文本预测器**：

```
输入: "今天天气"
GPT预测: "真好" (概率最高)
        "不好" (概率次之)
        "怎么样" (概率第三)
        ...
```

它通过阅读海量文本，学会了：

- 语法规则
- 语义关系
- 世界知识
- 推理能力

### 1.3 为什么用纯 Python 实现？

这个项目只用纯 Python（无 PyTorch、TensorFlow），目的是：

1. **透明性**: 每一行代码都清晰可见
2. **教育性**: 理解底层原理，而不是调用 API
3. **简洁性**: 核心逻辑只需约 235 行代码

#### 对比

| 方式      | 优点               | 缺点     |
| --------- | ------------------ | -------- |
| 纯 Python | 完全可控，易理解   | 性能较慢 |
| PyTorch   | 性能强大，功能丰富 | 黑盒较多 |

---

## 第二章：自动微分系统

### 2.1 为什么需要自动微分？

在深度学习中，我们需要：

1. **前向传播**: 计算预测值
2. **反向传播**: 计算梯度（如何调整参数）

**问题**：神经网络有百万级参数，手动求导不可能！

**解决方案**：自动微分（Automatic Differentiation）

### 2.2 Value 类的设计

`Value` 类是整个自动微分系统的核心。

#### 核心属性

```python
class Value:
    __slots__ = ('data', 'grad', '_children', '_local_grads')

    def __init__(self, data, children=(), local_grads=()):
        self.data = data                # 前向传播的值
        self.grad = 0                   # 反向传播的梯度
        self._children = children       # 计算图的子节点
        self._local_grads = local_grads # 局部导数
```

#### 可视化示例

```python
# 创建两个值
a = Value(2.0)
b = Value(3.0)

# 计算 c = a + b
c = a + b
```

这构建了如下计算图：

```
     a (2.0)          b (3.0)
        └────────┬────────┘
                 ↓
              c (5.0)
           children: (a, b)
      local_grads: (1, 1)
```

### 2.3 计算图的构建

**核心机制**：每次运算（加、乘等）都会创建一个**新的** `Value` 对象，这个新对象会**记住**它的"父母"是谁。

#### 简单示例

```python
a = Value(2.0)
b = Value(3.0)
c = a + b
d = c * 2
L = d * d
```

**计算图**：

```
a (2.0) ──┐
          ├──> c (5.0) ──┐
b (3.0) ──┘              ├──> d (10.0) ──┐
               2 ────────┘              ├──> L (100.0)
                    d (10.0) ────────────┘
```

**关键点**：

- `c` 对象**包含**了指向 `a` 和 `b` 的引用
- 通过 `c._children` 可以找到 `a` 和 `b`
- 这就形成了**有向无环图**（DAG）

### 2.4 反向传播算法

**反向传播**就是**从后往前**计算"每个参数对结果的影响有多大"。

#### 生活例子：烤蛋糕

想象你在烤蛋糕，配方是：

```
蛋糕好吃度 = 2 × 面粉量 + 3 × 糖量
```

**问题**：蛋糕比期望的甜了 **6 分**，怎么调整？

**反向传播的思路**：

1. **测量误差**：蛋糕太甜了 6 分
2. **反向找原因**：
   - 糖的影响系数是 **3**
   - 所以糖放多了：6 ÷ 3 = **2 克**
3. **调整**：下次少放 **2 克糖**

#### 链式法则

对于任意链式计算：

```
a ──> b ──> c ──> ... ──> L
```

**计算 L 对 a 的导数**：

```
∂L/∂a = ∂L/∂... × ∂.../∂c × ∂c/∂b × ∂b/∂a
      = 路径上所有导数的乘积
```

#### 代码实现

```python
def backward(self):
    # 1. 拓扑排序：确保从后往前计算
    topo = []
    visited = set()
    def build_topo(v):
        if v not in visited:
            visited.add(v)
            for child in v._children:
                build_topo(child)
            topo.append(v)
    build_topo(self)

    # 2. 初始化：输出节点的梯度 = 1
    self.grad = 1  # ∂L/∂L = 1

    # 3. 从后往前传播梯度
    for v in reversed(topo):
        for child, local_grad in zip(v._children, v._local_grads):
            # 链式法则：累积梯度
            child.grad += local_grad * v.grad
```

### 2.5 其他运算

#### 加法运算

```python
def __add__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    return Value(self.data + other.data, (self, other), (1, 1))
```

**数学推导**：

```
c = a + b
∂c/∂a = 1
∂c/∂b = 1
```

#### 乘法运算

```python
def __mul__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    return Value(self.data * other.data, (self, other), (other.data, self.data))
```

**数学推导**：

```
c = a · b
∂c/∂a = b
∂c/∂b = a
```

#### 对数、指数、ReLU

```python
def log(self):
    return Value(math.log(self.data), (self,), (1/self.data,))

def exp(self):
    return Value(math.exp(self.data), (self,), (math.exp(self.data),))

def relu(self):
    return Value(max(0, self.data), (self,), (float(self.data > 0),))
```

---

## 第三章：数据预处理

### 3.0 模型参数设计

在开始数据处理之前，先理解模型的超参数设计。

```python
n_layer = 1      # Transformer 层数
n_embd = 16      # 嵌入维度
block_size = 16  # 上下文窗口长度
n_head = 4       # 注意力头数
head_dim = 4     # 每个头的维度 (n_embd // n_head)
```

#### 核心概念

##### 1️⃣ 嵌入维度 (Embedding Dimension) = 特征数量

**嵌入维度就是每个 token 用多少个特征来表示。**

**生活例子**：

```
只用 3 个特征（维度=3）：
[身高, 体重, 年龄]
= [180, 75, 25]

用 16 个特征（维度=16）：
[身高, 体重, 年龄, 发色, 眼睛颜色,
 血型, 教育, 收入, 爱好, ...]
= [180, 75, 25, 黑色, 棕色,
  A型, 本科, 50k, 篮球, ...]
```

**维度越高，能表示的信息越丰富！**

##### 2️⃣ 层数 (Number of Layers) = 处理深度

**层是"处理的轮数"，每层都会对输入进行一次变换。**

**生活例子**：

```
第 1 层（理解字面意思）：
"苹果" = 一种水果

第 2 层（理解上下文）：
"苹果" 可能指水果，也可能指公司

第 3 层（理解深层含义）：
"苹果发布了新手机"
这里的"苹果"肯定不是水果
```

**层数越深，理解越深入！**

#### 两者的区别

| 维度          | 层数               |
| ------------- | ------------------ | ---------------- |
| **是什么**    | 特征数量           | 处理轮数         |
| **作用**      | "表示能力"         | "处理深度"       |
| **增加会...** | 能表示更复杂的信息 | 能理解更深的规律 |
| **类比**      | 图片分辨率         | 思考轮数         |

#### 参数量计算

```python
# Token Embedding
vocab_size × n_embd = 27 × 16 = 432

# Position Embedding
block_size × n_embd = 16 × 16 = 256

# LM Head
vocab_size × n_embd = 27 × 16 = 432

# Transformer 层（1 层）
注意力: 1,024
MLP:    2,048
─────────────
总计：   3,072 个参数

# 总参数量
432 + 256 + 432 + 3,072 = 4,192 ≈ 4,200 个参数
```

### 3.1 数据加载

```python
# 读取名字数据
words = open('names.txt', 'r').read().splitlines()

# 示例数据
# ['emma', 'olivia', 'ava', 'isabella', ...]
```

### 3.2 字符级 Tokenization

```python
# 构建词汇表
chars = sorted(list(set(''.join(words))))
chars = ['.'] + chars  # 添加特殊标记

stoi = {s:i for i,s in enumerate(chars)}  # string to index
itos = {i:s for s,i in stoi.items()}      # index to string

vocab_size = len(chars)  # 27 (26个字母 + 1个特殊标记)
```

### 3.3 构建训练数据

```python
def build_dataset(words):
    X, Y = [], [], []
    for w in words:
        context = [0] * block_size  # 用0填充开头
        for ch in w + '.':
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]  # 滑动窗口
    X = [Value(x) for x in X]
    Y = Y  # 标签是普通整数
    return X, Y
```

---

## 第四章：Transformer 架构

### 4.1 整体架构

```
输入 Token IDs
    ↓
┌─────────────────────────────────────┐
│  Embedding Layer                     │
│  - Token Embedding                   │
│  - Position Embedding                │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  Transformer Block × n_layer         │
│  - Multi-Head Self-Attention         │
│  - Feed-Forward Network              │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  LM Head (Linear Layer)              │
└─────────────────────────────────────┘
    ↓
Logits (27 维)
```

### 4.2 Embedding 层

#### Token Embedding

```python
# 将 token ID 转换成向量
tok_emb = state_dict['wte'][token_id]  # shape: [n_embd]
# 例如：'a' → [0.52, -0.31, 0.88, ...] (16维)
```

#### Position Embedding

```python
# 编码位置信息
pos_emb = state_dict['wpe'][pos_id]  # shape: [n_embd]
# 例如：位置0 → [0.12, 0.45, -0.23, ...] (16维)
```

#### 组合

```python
# Token和Position向量相加
x = [t + p for t, p in zip(tok_emb, pos_emb)]
# 结果：同时包含"这是什么字符"和"在什么位置"的信息
```

### 4.3 多头自注意力机制

#### 核心思想

**注意力机制**：让每个字符都能"看到"它之前的所有字符，并根据相关性决定关注哪些。

#### Query, Key, Value

```python
# 计算Q, K, V
q = linear(x, W_q)  # Query: 我想查什么？
k = linear(x, W_k)  # Key: 我提供什么信息？
v = linear(x, W_v)  # Value: 我的内容是什么？
```

**类比图书馆**：

- **Q**: "我要找关于AI的书"
- **K**: 每本书的关键词 ["编程", "AI", "数学", ...]
- **V**: 每本书的内容摘要

#### 注意力计算

```python
# 1. 计算注意力分数 (Q · K)
scores = dot_product(q, k)

# 2. 缩放
scores = scores / sqrt(head_dim)

# 3. Softmax 转成概率
attn_weights = softmax(scores)

# 4. 加权求和 Value
output = weighted_sum(attn_weights, v)
```

#### 多头机制

```python
# 16维向量分成4个头，每个头4维
for h in range(n_head):
    hs = h * head_dim  # 0, 4, 8, 12

    # 取出当前头的Q, K, V
    q_h = q[hs:hs+head_dim]
    k_h = [ki[hs:hs+head_dim] for ki in keys]
    v_h = [vi[hs:hs+head_dim] for vi in values]

    # 计算注意力
    head_out = attention(q_h, k_h, v_h)

    # 拼接到总输出
    x_attn.extend(head_out)
```

**为什么要多头**？

- **头0**：可能学到关注元音字母
- **头1**：可能学到关注辅音字母
- **头2**：可能学到关注重复字母
- **头3**：可能学到关注名字长度

### 4.4 前馈网络 (MLP)

```python
# 第一层：扩展维度
hidden = [linear(x, W_fc1) for x in x_attn]
hidden = [relu(h) for h in hidden]

# 第二层：投影回原维度
x = [linear(h, W_fc2) for h in hidden]
```

---

## 第五章：训练过程

### 5.1 前向传播

```python
def gpt(token_id, pos_id, keys, values):
    # 1. Embedding
    x = embed(token_id, pos_id)

    # 2. Transformer Block
    x = transformer_block(x, keys, values)

    # 3. LM Head
    logits = linear(x, lm_head)

    return logits
```

### 5.2 损失计算

#### 交叉熵损失

```python
def cross_entropy_loss(logits, target):
    # 1. Softmax
    probs = softmax(logits)

    # 2. 取目标类别的概率
    prob_target = probs[target]

    # 3. 负对数似然
    loss = -log(prob_target)

    return loss
```

**直观理解**：

```
如果模型预测正确（P(target) = 0.9）：
  loss = -log(0.9) = 0.105  ← 小

如果模型预测错误（P(target) = 0.1）：
  loss = -log(0.1) = 2.302  ← 大
```

### 5.3 反向传播

```python
# 1. 前向传播计算损失
logits = gpt(...)
loss = cross_entropy_loss(logits, target)

# 2. 反向传播计算梯度
loss.backward()

# 3. 更新参数
for param in params:
    param.data -= learning_rate * param.grad
```

### 5.4 Adam 优化器

Adam 结合了动量和自适应学习率：

```python
# 更新规则
m = beta1 * m + (1 - beta1) * grad  # 动量
v = beta2 * v + (1 - beta2) * grad**2  # 自适应学习率
param.data -= lr * m / (sqrt(v) + eps)
```

---

## 第六章：推理生成

### 6.1 自回归生成

```python
def generate():
    # 初始化
    keys, values = [[]], [[]]
    token_id = BOS  # 从开始标记开始
    sample = []

    # 循环生成
    for pos_id in range(block_size):
        # 前向传播
        logits = gpt(token_id, pos_id, keys, values)

        # 温度缩放
        logits = [l / temperature for l in logits]

        # Softmax
        probs = softmax(logits)

        # 按概率采样
        token_id = random.choices(range(vocab_size),
                                  weights=[p.data for p in probs])[0]

        # 检查结束
        if token_id == BOS:
            break

        # 记录生成的字符
        sample.append(uchars[token_id])

    return ''.join(sample)
```

**自回归的含义**：

```
每次预测都依赖之前生成的结果：
  第1次：BOS → 'a'
  第2次：'a' → 'n'（依赖第1次）
  第3次：'n' → 'n'（依赖第1、2次）
  第4次：'n' → 'a'（依赖第1、2、3次）
```

### 6.2 温度采样

**温度参数**控制生成的"创造性"：

```
原始 logits: [2.0, 1.0, 0.5, -1.0]

Temperature = 0.5 (更保守):
  概率:  [0.84, 0.14, 0.02, 0.00]  # 更集中

Temperature = 1.0 (原始):
  概率:  [0.62, 0.23, 0.12, 0.03]

Temperature = 2.0 (更创新):
  概率:  [0.43, 0.26, 0.19, 0.12]  # 更均匀
```

**建议**：

- 0.3-0.7: 生成保守、常见的内容
- 0.8-1.0: 生成有创意但合理的内容
- 1.0+: 生成非常创新但可能不合理的内容

---

## 附录：数学公式推导

### A.1 链式法则

**链式法则**是微积分中的基本定理，也是反向传播的基础。

#### 单变量情况

如果 `y = f(x)` 和 `z = g(y)`，那么：

```
dz/dx = dz/dy · dy/dx
```

**例子**：

```
z = (3x)²

设 y = 3x，则 z = y²

dy/dx = 3
dz/dy = 2y = 6x
dz/dx = 6x · 3 = 18x
```

### A.2 注意力机制推导

#### 缩放点积注意力

```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

**为什么除以 `√d_k`？**

当 `d_k` 很大时，点积 `QK^T` 的值会很大，导致 softmax 进入饱和区（梯度很小）。

### A.3 交叉熵损失

从最大似然估计推导：

```
L = -Σᵢ log P(y⁽ⁱ⁾ | x⁽ⁱ⁾)
```

对于分类问题，`P(y | x)` 就是 softmax 输出的概率：

```
L = -log(softmax(z)_y)
```

其中 `z` 是 logits。

### A.4 Adam 优化器推导

Adam 结合了：

1. **动量（Momentum）**: 平滑梯度更新

   ```
   m_t = β₁ m_{t-1} + (1 - β₁) g_t
   ```

2. **RMSprop**: 自适应学习率
   ```
   v_t = β₂ v_{t-1} + (1 - β₂) g_t²
   ```

**最终更新规则**：

```
θ_{t+1} = θ_t - α · m̂_t / (√v̂_t + ε)
```

---

## 总结

本教程通过纯 Python 实现了一个最小化的 GPT 模型，涵盖了：

1. **自动微分系统**：实现前向传播和反向传播
2. **数据预处理**：字符级 tokenization 和数据集构建
3. **Transformer 架构**：
   - Embedding 层（Token + Position）
   - 多头自注意力机制
   - 前馈网络
4. **训练过程**：前向传播、损失计算、反向传播、参数更新
5. **推理生成**：自回归生成和温度采样

**核心要点**：

- GPT 本质上是一个强大的文本预测器
- Transformer 通过注意力机制捕获长距离依赖
- 多头注意力让模型能同时关注不同的模式
- 训练通过反向传播和优化器更新参数
- 推理通过自回归方式逐个生成 token

**下一步**：

- 尝试调整超参数（n_embd, n_layer, n_head）
- 在不同的数据集上训练
- 实现更高级的技术（层归一化、dropout、残差连接）
- 学习预训练和微调范式

---

**文档版本**: 精炼版 v1.0
**原始行数**: 6552 行
**精炼后**: 约 800 行
**压缩比**: 88%
**保留内容**: 所有核心概念和关键实现
