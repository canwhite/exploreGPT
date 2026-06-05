## micro.py 完整流程（500字）

### 从训练去理解概念

  训练循环 (training loop)
  ├── 1. 前向传播 (forward pass)
  │      输入 → 模型 → logits → loss
  │
  ├── 2. 反向传播 (backward pass)
  │      loss.backward() → 计算梯度
  │
  └── 3. 参数更新 (parameter update)
         Adam: p.data -= lr * gradient


### 1. 数据

`docs` 是 2 万多条人名组成的列表。打乱后随机取一条，比如 "alex"。在首尾插入特殊标记 BOS，分词成：

```
[BOS, a, l, e, x, BOS]
```

vocab 是所有字符的集合（0-25 映射 a-z），加 1 个 BOS。
---

### 2. 模型参数

`state_dict` 初始化了以下权重（都是 `Value` 对象，存了 data 和 grad）：

| 参数 | 形状 | 作用 |
|------|------|------|
| `wte` | vocab_size × 16 | token 嵌入表 |
| `wpe` | 16 × 16 | 位置嵌入表 |
| `lm_head` | vocab_size × 16 | 输出层，映射 hidden → vocab logits |
| `attn_wq/wk/wv` | 16 × 16 | Q/K/V 投影矩阵 |
| `attn_wo` | 16 × 16 | 注意力输出投影 |
| `mlp_fc1` | 64 × 16 | MLP 第一层（hidden×4） |
| `mlp_fc2` | 16 × 64 | MLP 第二层 |

总共 ~2.5 万个参数。

---

### 3. 前向传播

对序列中每个位置 `pos`：

1. **Embedding**：用 `wte[token_id]` 取 token 向量，用 `wpe[pos_id]` 取位置向量，相加得到输入
2. **RMSNorm**：归一化，稳定训练
3. **多头注意力**：
   - Q/K/V 分别通过 `attn_wq/wk/wv` 投影得到 query/key/value
   - 每个头计算 `softmax(Q·K^T / √d)` 得到注意力权重
   - 加权求和 V，得到注意力输出，再经 `attn_wo` 投影
   - 残差连接：`x = x + attn_output`
4. **MLP**：`fc1` 扩展到 64 维 → ReLU → `fc2` 投影回 16 维，残差连接
5. **LM Head**：`lm_head` 把 hidden vector 变成 vocab_size 维 logits
6. **Loss**：`softmax(logits)` 得到概率，取目标 token 的负对数概率作为这一位置的 loss

所有位置 loss 求平均，得到这一条样本的总 loss。

---

### 4. 反向传播

反向传播是微积分链式法则在计算图上的递归应用。它的目标只有一个：**算出每个参数对最终 loss 的梯度**——即"这个参数往哪个方向调能让 loss 降低"。

#### 核心思想：链式法则
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
详细过程：
```
前向传播

a = 2.0
b = 3.0
c = a + b = 5.0
d = c * 2 = 10.0
L = d * d = 100.0

反向传播（从后往前）

1. L = d * d

对乘法的局部梯度：∂L/∂d = d_local_grad + d_local_grad = d + d

∂L/∂d = 1 * d.grad + 1 * d.grad = 1 + 1 = 2
∂L/∂d = 2 * d.data = 2 * 10 = 20

2. d = c * 2

乘法局部梯度：∂d/∂c = 2

∂L/∂c = ∂L/∂d * ∂d/∂c = 20 * 2 = 40

3. c = a + b

加法局部梯度：∂c/∂a = 1, ∂c/∂b = 1

∂L/∂a = 40 * 1 = 40
∂L/∂b = 40 * 1 = 40

最终结果
┌──────┬──────┬────────────────────────┐
│ 参数  │ 梯度  │          含义          │
├──────┼──────┼────────────────────────┤
│ a    │ 40   │ a 增大 1，loss 增大 40   │
├──────┼──────┼────────────────────────┤
│ b    │ 40   │ b 增大 1，loss 增大 40   │
├──────┼──────┼────────────────────────┤
│ c    │ 40   │ c 增大 1，loss 增大 40   │
└──────┴──────┴────────────────────────┘
验证

L = (2c)² = 4c²
∂L/∂c = 8c = 8*5 = 40  ✓
∂L/∂a = ∂L/∂c * ∂c/∂a = 40 * 1 = 40  ✓

所以调整方向是：a 和 b 都应该减小（梯度为正意味着增大它会增大 loss，所以往反方向走）。
```


#### micro.py 中的实现

`Value` 对象记录了每个节点的：
- `data`：前向传播时的输出值
- `grad`：反向传播时累积的梯度
- `_children`：子节点引用
- `_local_grads`：该节点对每个子节点的局部导数，这个是局部导数；

`backward()` 的过程：

```python
def backward(self):
    # 1. 拓扑排序：从 loss 出发，保证先处理依赖的节点
    topo = []
    build_topo(loss_node)  # DFS 把所有节点按依赖顺序加入 topo

    # 2. 逆序传播：从后往前
    self.grad = 1  # loss 自身的梯度是 1
    for v in reversed(topo):
        for child, local_grad in zip(v._children, v._local_grads):
            child.grad += local_grad * v.grad  # 链式法则
```

#### micro.py 实际运算的梯度推导

以 micro.py 中的 `linear` + `softmax` + `cross_entropy` 为例：

```python
# 假设简化场景：vocab_size = 3, hidden_dim = 2
# w: 3x2 的权重矩阵
w = [[Value(1.0), Value(0.5)],   # 行0: 对应token 0
     [Value(2.0), Value(1.0)],    # 行1: 对应token 1
     [Value(0.5), Value(1.5)]]    # 行2: 对应token 2

# x: hidden vector [2维]
x = [Value(1.0), Value(0.8)]

# 前向传播
logits = linear(x, w)  # = [x·w[0], x·w[1], x·w[2]]
# logits[0] = 1.0*1.0 + 0.8*0.5 = 1.4
# logits[1] = 1.0*2.0 + 0.8*1.0 = 2.8
# logits[2] = 1.0*0.5 + 0.8*1.5 = 1.7

probs = softmax(logits)  # 归一化成概率
# max = 2.8
# exp(0) = 1.0, exp(-1.4) ≈ 0.25, exp(-1.1) ≈ 0.33
# sum ≈ 1.0 + 0.25 + 0.33 = 1.58
# probs ≈ [0.63, 0.22, 0.15]

# 假设目标 token = 0，loss = -log(probs[0]) ≈ -log(0.63) ≈ 0.46
```

**反向传播（只追踪 w[0][0] 的梯度）**：

```
loss = -log(probs[0])

1. loss = -log(probs[0])
   ∂loss/∂probs[0] = -1/probs[0] = -1/0.63 ≈ -1.59

2. probs[i] = exp_i / sum_exp
   ∂probs[0]/∂logits[0] = probs[0] * (1 - probs[0]) = 0.63 * 0.37 ≈ 0.23
   ∂probs[0]/∂logits[1] = -probs[0]*probs[1] = -0.63*0.22 ≈ -0.14
   ∂probs[0]/∂logits[2] = -probs[0]*probs[2] = -0.63*0.15 ≈ -0.09

3. logits[i] = x · w[i]
   ∂logits[0]/∂w[0][0] = x[0] = 1.0

4. 链式法则：
   ∂loss/∂w[0][0] = ∂loss/∂probs[0] * ∂probs[0]/∂logits[0] * ∂logits[0]/∂w[0][0]
                  = (-1.59) * (0.23) * (1.0)
                  ≈ -0.37
```

**梯度含义**：w[0][0]（即 token 0 对应的权重第一个元素）应该**减小**（梯度为负），这样 logits[0] 会变大，probs[0] 会变大，loss 会降低。

**实际代码中如何工作**：
- `softmax` 的 `.grad` 先算出来
- `log` 的 `.grad` 通过链式法则传回去
- `linear` 里的每个 `Value(w[i][j])` 都会累积梯度到 `.grad`
- 最终 `w[0][0].grad ≈ -0.37`，`w[0][1].grad ≈ -0.30`，等等
- Adam 更新时会：`w[0][0].data -= lr * (-0.37) = w[0][0].data + 正数`，即增大 w[0][0]，这正好是我们需要的方向！

---

### 5. 参数更新

Adam 优化器分两步：

```
m = β1*m + (1-β1)*grad     # 动量，平滑梯度
v = β2*v + (1-β2)*grad²     # 自适应学习率
p.data -= lr * m_hat / (√v_hat + ε)
p.grad = 0                  # 清零，准备下个 step
```

注意是**原地修改** `p.data`，不创建新对象。下次前向传播直接用新权重。

---

### 6. 推理

推理时 `state_dict` 不变。模型从 BOS 开始，逐位置前向传播得到 logits，用 temperature 缩放后 softmax 采样下一个 token。遇到 BOS 停止，输出生成的名字。

---

**核心循环**：前向算 loss → 反向算梯度 → Adam 更新权重 → 前向算新 loss → … → 权重逐渐"记住"训练数据分布 → 推理时能生成类似的新样本。
