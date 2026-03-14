# 自注意力机制详解 - 基于项目实例

## 1. 项目配置

```python
# 项目中的模型配置
n_layer = 1        # 1层 Transformer
n_embd = 16        # 嵌入维度 16
n_head = 4         # 4个注意力头
head_dim = 4       # 每个头的维度 = 16 / 4
block_size = 16    # 最大序列长度 16
vocab_size = 27    # 词汇表大小（26个字母 + 1个BOS）
```

## 2. 完整流程演示

### 场景：生成名字 "emma"

```
输入序列：[BOS, 'e', 'm', 'm', 'a']
位置：    [ 0,   1,   2,   3,   4]
```

### 步骤1：Token 嵌入 + 位置嵌入

```python
# 假设输入是第3个字符 'm'（位置 pos_id=2）
token_id = uchars.index('m')  # token_id = 12
pos_id = 2

# Token 嵌入
tok_emb = state_dict['wte'][token_id]
# 从 wte 矩阵取出第12行，形状：(16,)
# 例如：[0.1, -0.2, 0.3, 0.05, ..., 0.08]

# 位置嵌入
pos_emb = state_dict['wpe'][pos_id]
# 从 wpe 矩阵取出第2行，形状：(16,)
# 例如：[0.02, 0.01, -0.03, 0.04, ..., 0.02]

# 结合
x = [t + p for t, p in zip(tok_emb, pos_emb)]
# 结果：[0.12, -0.19, 0.27, 0.09, ..., 0.10]
# 形状：(16,)
```

### 步骤2：RMSNorm 归一化

```python
x = rmsnorm(x)
# 稳定数值，防止后续计算溢出
# 形状仍然是：(16,)
```

### 步骤3：生成 Query、Key、Value

```python
# 使用线性变换生成 Q、K、V
q = linear(x, state_dict['layer0.attn_wq'])
# Query: "当前词'我要查询什么'"
# 形状：(16,)

k = linear(x, state_dict['layer0.attn_wk'])
# Key: "当前词'能提供什么信息'"
# 形状：(16,)

v = linear(x, state_dict['layer0.attn_wv'])
# Value: "当前词'的实际内容'"
# 形状：(16,)
```

**具体计算过程**：

```python
# linear 函数的实现
def linear(x, w):
    """y = xW^T"""
    return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]

# 假设：
x = [0.12, -0.19, 0.27, 0.09, ...]  # 16维
w_q = state_dict['layer0.attn_wq']   # 16×16 矩阵

# 计算第1个维度：
q[0] = w_q[0][0]*x[0] + w_q[0][1]*x[1] + ... + w_q[0][15]*x[15]

# 计算第2个维度：
q[1] = w_q[1][0]*x[0] + w_q[1][1]*x[1] + ... + w_q[1][15]*x[15]

# ... 共16个维度
```

**缓存 Key 和 Value**：

```python
# 项目中使用 KV Cache（缓存历史信息）
keys[0].append(k)    # 存储当前词的 Key
values[0].append(v)  # 存储当前词的 Value

# 假设已经处理了 "BOS, 'e', 'm'"
# keys[0] = [k_BOS, k_e, k_m]  # 3个历史Key
# values[0] = [v_BOS, v_e, v_m]  # 3个历史Value
```

### 步骤4：多头注意力计算

```python
# 项目配置：4个头，每个头4维
n_head = 4
head_dim = 4

x_attn = []  # 存储所有头的输出

for h in range(n_head):
    # 4.1 切分当前头的 Q、K、V
    hs = h * head_dim  # 起始索引
    
    # 当前头的 Query（4维）
    q_h = q[hs:hs+head_dim]
    # 例如头0：q[0:4] = [0.2, -0.1, 0.3, 0.05]
    
    # 当前头的所有历史 Keys（每个4维）
    k_h = [ki[hs:hs+head_dim] for ki in keys[0]]
    # 例如头0：
    # k_h = [
    #     [0.1, 0.2, -0.1, 0.3],   # BOS 的 Key
    #     [0.4, -0.2, 0.1, 0.2],   # 'e' 的 Key
    #     [0.3, 0.1, -0.2, 0.4]    # 'm' 的 Key（当前词）
    # ]
    
    # 当前头的所有历史 Values（每个4维）
    v_h = [vi[hs:hs+head_dim] for vi in values[0]]
    # 例如头0：
    # v_h = [
    #     [0.2, 0.1, 0.3, -0.1],   # BOS 的 Value
    #     [0.1, 0.3, -0.2, 0.1],   # 'e' 的 Value
    #     [0.4, -0.1, 0.2, 0.3]    # 'm' 的 Value
    # ]
```

### 步骤5：计算注意力分数

```python
# 5.1 计算 Q·K^T（点积）
attn_logits = []
for t in range(len(k_h)):  # 遍历所有历史位置
    # 计算当前 Query 与第 t 个 Key 的点积
    dot_product = sum(q_h[j] * k_h[t][j] for j in range(head_dim))
    # 缩放：除以 √d_k
    scaled = dot_product / (head_dim ** 0.5)
    attn_logits.append(scaled)

# 具体例子：
# t=0 (BOS):   [0.2, -0.1, 0.3, 0.05] · [0.1, 0.2, -0.1, 0.3] / 2
#            = (0.02 - 0.02 - 0.03 + 0.015) / 2 = -0.0075
#
# t=1 ('e'):   [0.2, -0.1, 0.3, 0.05] · [0.4, -0.2, 0.1, 0.2] / 2
#            = (0.08 + 0.02 + 0.03 + 0.01) / 2 = 0.07
#
# t=2 ('m'):   [0.2, -0.1, 0.3, 0.05] · [0.3, 0.1, -0.2, 0.4] / 2
#            = (0.06 - 0.01 - 0.06 + 0.02) / 2 = 0.005

attn_logits = [-0.0075, 0.07, 0.005]
```

### 步骤6：Softmax 归一化

```python
attn_weights = softmax(attn_logits)

# Softmax 计算：
# 1. 找最大值
max_val = max(-0.0075, 0.07, 0.005) = 0.07

# 2. 减去最大值并计算指数
exps = [
    exp(-0.0075 - 0.07),  # exp(-0.0775) ≈ 0.925
    exp(0.07 - 0.07),      # exp(0) = 1.0
    exp(0.005 - 0.07)      # exp(-0.065) ≈ 0.937
]

# 3. 求和
total = 0.925 + 1.0 + 0.937 = 2.862

# 4. 归一化
attn_weights = [0.925/2.862, 1.0/2.862, 0.937/2.862]
             = [0.323, 0.349, 0.328]

# 解释：
# 当前词 'm' 对各位置的关注度：
# - BOS:  32.3%
# - 'e':  34.9%  ← 最关注
# - 'm':  32.8%
```

### 步骤7：加权求和

```python
head_out = []
for j in range(head_dim):  # 遍历每个维度
    # 根据注意力权重聚合所有 Value
    weighted_sum = 0
    for t in range(len(v_h)):
        weighted_sum += attn_weights[t] * v_h[t][j]
    head_out.append(weighted_sum)

# 具体计算：
# 第1维（j=0）：
# 0.323 * 0.2 + 0.349 * 0.1 + 0.328 * 0.4
# = 0.0646 + 0.0349 + 0.1312 = 0.2307

# 第2维（j=1）：
# 0.323 * 0.1 + 0.349 * 0.3 + 0.328 * (-0.1)
# = 0.0323 + 0.1047 - 0.0328 = 0.1042

# ... 继续计算其他维度

head_out = [0.2307, 0.1042, ...]  # 4维向量
```

### 步骤8：合并所有头

```python
x_attn.extend(head_out)

# 重复步骤4-7，处理其他3个头
# 最终 x_attn 包含 4 * 4 = 16 维
# x_attn = [
#     头0输出 (4维),
#     头1输出 (4维),
#     头2输出 (4维),
#     头3输出 (4维)
# ]
```

### 步骤9：输出投影

```python
x = linear(x_attn, state_dict['layer0.attn_wo'])
# 将多头输出投影回原始维度
# 形状：(16,) → (16,)
```

### 步骤10：残差连接

```python
x = [a + b for a, b in zip(x, x_residual)]
# x_residual 是步骤2保存的原始输入
# 残差连接缓解梯度消失，加速训练
```

## 3. 完整流程图

```
输入：token_id=12 ('m'), pos_id=2
    ↓
[Token 嵌入] tok_emb = wte[12]
[位置嵌入]  pos_emb = wpe[2]
    ↓
[结合] x = tok_emb + pos_emb  (16维)
    ↓
[RMSNorm] x = rmsnorm(x)
    ↓
保存 x_residual = x（用于后续残差连接）
    ↓
[线性投影]
    ├─ q = linear(x, attn_wq)  → Query (16维)
    ├─ k = linear(x, attn_wk)  → Key (16维)
    └─ v = linear(x, attn_wv)  → Value (16维)
    ↓
[缓存] keys.append(k), values.append(v)
    ↓
[多头注意力]
    ├─ 头0: q_h[0:4] · k_h[0:4]^T → weights → Σ(weights * v_h)
    ├─ 头1: q_h[4:8] · k_h[4:8]^T → weights → Σ(weights * v_h)
    ├─ 头2: q_h[8:12] · k_h[8:12]^T → weights → Σ(weights * v_h)
    └─ 头3: q_h[12:16] · k_h[12:16]^T → weights → Σ(weights * v_h)
    ↓
[合并] x_attn (16维)
    ↓
[输出投影] x = linear(x_attn, attn_wo)
    ↓
[残差连接] x = x + x_residual
    ↓
输出：融合了上下文信息的向量 (16维)
```

## 4. 关键代码对应关系

### 公式 → 代码

| 公式 | 项目代码位置 |
|------|-------------|
| Q = XW_Q | main.py:149 `q = linear(x, state_dict[f'layer{li}.attn_wq'])` |
| K = XW_K | main.py:150 `k = linear(x, state_dict[f'layer{li}.attn_wk'])` |
| V = XW_V | main.py:151 `v = linear(x, state_dict[f'layer{li}.attn_wv'])` |
| Q·K^T | main.py:161 `sum(q_h[j] * k_h[t][j] for j in range(head_dim))` |
| /√d_k | main.py:161 `/ head_dim**0.5` |
| Softmax | main.py:162 `attn_weights = softmax(attn_logits)` |
| weights·V | main.py:164 `sum(attn_weights[t] * v_h[t][j] ...)` |

## 5. 实际运行示例

### 创建测试脚本

```python
# test_attention.py
import math
import random
from main import (
    Value, linear, softmax, rmsnorm, 
    state_dict, n_head, head_dim, n_embd
)

# 简化版自注意力（单头）
def single_head_attention(x, w_q, w_k, w_v):
    """单头自注意力演示"""
    # 1. 生成 Q、K、V
    q = linear(x, w_q)
    k = linear(x, w_k)
    v = linear(x, w_v)
    
    print(f"Query:  {[round(v.data, 3) for v in q[:4]]}")
    print(f"Key:    {[round(v.data, 3) for v in k[:4]]}")
    print(f"Value:  {[round(v.data, 3) for v in v[:4]]}")
    
    # 2. 计算注意力分数
    dot = sum(qi * ki for qi, ki in zip(q, k))
    score = dot / math.sqrt(len(q))
    print(f"\n注意力分数: {score.data:.3f}")
    
    # 3. Softmax（单元素时为1.0）
    weight = softmax([score])[0]
    print(f"注意力权重: {weight.data:.3f}")
    
    # 4. 加权求和
    output = [weight * vi for vi in v]
    print(f"\n输出: {[round(v.data, 3) for v in output[:4]]}")
    
    return output

# 测试
random.seed(42)
x = [Value(random.gauss(0, 0.1)) for _ in range(n_embd)]
w_q = state_dict['layer0.attn_wq']
w_k = state_dict['layer0.attn_wk']
w_v = state_dict['layer0.attn_wv']

print("输入 x:", [round(v.data, 3) for v in x[:4]], "...")
print()

output = single_head_attention(x, w_q, w_k, w_v)
```

运行：

```bash
python test_attention.py
```

输出示例：

```
输入 x: [0.034, -0.012, 0.056, -0.023] ...

Query:  [0.012, -0.008, 0.034, 0.021]
Key:    [-0.015, 0.029, -0.011, 0.018]
Value:  [0.042, -0.016, 0.028, 0.007]

注意力分数: 0.002
注意力权重: 1.000

输出: [0.042, -0.016, 0.028, 0.007]
```

## 6. 多头注意力的作用

### 为什么需要多个头？

```python
# 不同头学习不同的关系模式

头0: 关注局部依赖（相邻词）
    "emma" → 'm' 关注前面的 'e'

头1: 关注全局依赖（远距离词）
    "emma" → 'm' 关注句首的 BOS

头2: 关注语法关系
    "emma" → 识别这是名字模式

头3: 关注语义关系
    "emma" → 理解这是人名

# 最终合并所有头的输出
output = [头0输出, 头1输出, 头2输出, 头3输出]
```

### 项目中的实现

```python
# main.py:155-165
x_attn = []
for h in range(n_head):  # 4个头
    hs = h * head_dim
    q_h = q[hs:hs+head_dim]      # 切分当前头的 Query
    k_h = [ki[hs:hs+head_dim] for ki in keys[li]]
    v_h = [vi[hs:hs+head_dim] for vi in values[li]]
    
    # 计算当前头的注意力
    attn_logits = [...]
    attn_weights = softmax(attn_logits)
    head_out = [...]
    
    x_attn.extend(head_out)  # 合并到结果中
```

## 7. KV Cache 优化

### 为什么需要缓存？

```python
# 生成序列：BOS → 'e' → 'm' → 'm' → 'a'

# 不使用缓存（每次重新计算）
位置0: 计算 BOS 的 Q、K、V
位置1: 计算 'e' 的 Q、K、V
       重新计算 BOS 的 K、V  ← 浪费！
位置2: 计算 'm' 的 Q、K、V
       重新计算 BOS、'e' 的 K、V  ← 更浪费！

# 使用缓存（项目采用的方式）
位置0: 计算 BOS 的 K、V → 存入 cache
位置1: 计算 'e' 的 K、V → 存入 cache
       直接使用 cache 中的 [BOS_K, BOS_V]
位置2: 计算 'm' 的 K、V → 存入 cache
       直接使用 cache 中的 [BOS_K, 'e'_K]
```

### 项目实现

```python
# main.py:152-153
keys[li].append(k)    # 缓存当前 Key
values[li].append(v)  # 缓存当前 Value

# main.py:158-159
k_h = [ki[hs:hs+head_dim] for ki in keys[li]]  # 使用所有历史 Key
v_h = [vi[hs:hs+head_dim] for vi in values[li]]  # 使用所有历史 Value
```

## 8. 总结

### 核心流程

```
输入词 → Token嵌入+位置嵌入 → RMSNorm
    ↓
生成 Q、K、V
    ↓
多头注意力：
  - 切分每个头的 Q、K、V
  - 计算 Q·K^T / √d_k
  - Softmax
  - 加权求和
    ↓
合并多头输出 → 线性投影 → 残差连接
    ↓
融合了上下文的词向量
```

### 项目特色

1. **纯 Python 实现**：无依赖框架，易于理解
2. **KV Cache**：缓存历史 K、V，提升推理效率
3. **多头注意力**：4个头，每个头4维
4. **RMSNorm**：替代 LayerNorm，计算更快
5. **残差连接**：缓解梯度消失

### 关键参数

| 参数 | 值 | 说明 |
|------|----|----|
| n_layer | 1 | Transformer 层数 |
| n_embd | 16 | 嵌入维度 |
| n_head | 4 | 注意力头数量 |
| head_dim | 4 | 每个头的维度 |
| block_size | 16 | 最大序列长度 |

### 性能优化点

1. **预归一化**：在注意力前先 RMSNorm
2. **KV Cache**：避免重复计算
3. **残差连接**：加速梯度流动
4. **缩放因子**：√d_k 防止梯度消失

**一句话总结**：自注意力让每个词通过 Query"询问"其他词，根据 Key"匹配"相关度，最后用 Value"聚合"信息，使每个词都获得上下文感知能力。
