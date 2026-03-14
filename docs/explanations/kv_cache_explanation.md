# KV Cache 详解

基于 main.py 的实现，解释 Transformer 推理中的 KV Cache 技术。

## 什么是 KV Cache？

KV Cache 是 Transformer 推理时的优化技术，用于**缓存已计算过的 Key 和 Value**，避免重复计算。

## main.py 中的 KV Cache 实现

### 初始化

```python
# main.py 中的关键代码

# 初始化空的 KV 缓存
keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]

# 推理循环
for pos_id in range(block_size):
    logits = gpt(token_id, pos_id, keys, values)  # 传入 keys, values
    # ...
```

### gpt 函数内部

```python
def gpt(token_id, pos_id, keys, values):
    # ...
    for li in range(n_layer):
        q = linear(x, state_dict[f'layer{li}.attn_wq'])  # 计算当前 Query
        k = linear(x, state_dict[f'layer{li}.attn_wk'])  # 计算当前 Key
        v = linear(x, state_dict[f'layer{li}.attn_wv'])  # 计算当前 Value

        keys[li].append(k)   # 缓存 Key
        values[li].append(v) # 缓存 Value

        # 使用所有缓存的 K, V 计算注意力
        for h in range(n_head):
            k_h = [ki[hs:hs+head_dim] for ki in keys[li]]   # 所有历史 Keys
            v_h = [vi[hs:hs+head_dim] for vi in values[li]] # 所有历史 Values
            # ... 注意力计算
```

## 为什么需要 KV Cache？

### 不使用 KV Cache（重复计算）

```
生成 "hello" 的过程:

Step 1: 输入 [BOS]
        计算 K1, V1
        注意力: Q1 · K1

Step 2: 输入 [BOS, h]
        重新计算 K1, K2  ← K1 重复计算了！
        重新计算 V1, V2  ← V1 重复计算了！
        注意力: Q2 · [K1, K2]

Step 3: 输入 [BOS, h, e]
        重新计算 K1, K2, K3  ← K1, K2 重复计算了！
        重新计算 V1, V2, V3  ← V1, V2 重复计算了！
        注意力: Q3 · [K1, K2, K3]
```

### 使用 KV Cache（增量计算）

```
生成 "hello" 的过程:

Step 1: 输入 [BOS]
        计算 K1, V1
        缓存: keys = [K1], values = [V1]
        注意力: Q1 · K1

Step 2: 输入 [h]
        只计算 K2, V2  ← 只计算新的！
        缓存: keys = [K1, K2], values = [V1, V2]
        注意力: Q2 · [K1, K2]

Step 3: 输入 [e]
        只计算 K3, V3  ← 只计算新的！
        缓存: keys = [K1, K2, K3], values = [V1, V2, V3]
        注意力: Q3 · [K1, K2, K3]
```

## 时间复杂度对比

| 方式        | 第 n 步计算量   | 总计算量 |
| ----------- | --------------- | -------- |
| 无 KV Cache | O(n) 次前向传播 | O(n²)    |
| 有 KV Cache | O(1) 次前向传播 | O(n)     |

## main.py 中的数据流示例

生成名字 "ana" 的过程：

```
pos_id=0: token=BOS
    ┌─────────────────────────────────┐
    │ gpt(BOS, 0, [], [])             │
    │   计算 K0, V0                   │
    │   keys[0] = [K0]                │
    │   values[0] = [V0]              │
    │   注意力: Q0 · K0               │
    └─────────────────────────────────┘
    输出: 预测 'a'

pos_id=1: token='a'
    ┌─────────────────────────────────┐
    │ gpt('a', 1, [K0], [V0])         │
    │   计算 K1, V1                   │
    │   keys[0] = [K0, K1]            │
    │   values[0] = [V0, V1]          │
    │   注意力: Q1 · [K0, K1]         │
    └─────────────────────────────────┘
    输出: 预测 'n'

pos_id=2: token='n'
    ┌─────────────────────────────────┐
    │ gpt('n', 2, [K0,K1], [V0,V1])   │
    │   计算 K2, V2                   │
    │   keys[0] = [K0, K1, K2]        │
    │   values[0] = [V0, V1, V2]      │
    │   注意力: Q2 · [K0, K1, K2]     │
    └─────────────────────────────────┘
    输出: 预测 'a'
```

## 内存占用计算

```python
# main.py 中的配置
n_layer = 1      # 层数
n_embd = 16      # 嵌入维度
block_size = 16  # 最大序列长度

# KV Cache 大小计算
# 每层: 2 * block_size * n_embd (K 和 V)
# 总计: n_layer * 2 * block_size * n_embd
#     = 1 * 2 * 16 * 16 = 512 个 Value 对象
```

## KV Cache 的关键点

| 要点           | 说明                  |
| -------------- | --------------------- |
| **缓存什么**   | Key 和 Value 向量     |
| **不缓存什么** | Query（每次都是新的） |
| **生命周期**   | 整个序列生成过程      |
| **清空时机**   | 开始新序列时          |
| **空间换时间** | 用内存换取计算速度    |

## 为什么只缓存 K 和 V？

### 自注意力的本质

```
注意力机制: Attention(Q, K, V) = softmax(Q·K^T / √d) · V

- Q (Query): 当前位置的查询向量，每次都不同
- K (Key): 历史位置的键向量，计算后不变
- V (Value): 历史位置的值向量，计算后不变
```

### 因果掩码（Causal Mask）

```
GPT 使用因果掩码，位置 i 只能看到位置 0..i

位置 0: 只看 [0]           → K0, V0
位置 1: 只看 [0, 1]        → K0, K1, V0, V1
位置 2: 只看 [0, 1, 2]     → K0, K1, K2, V0, V1, V2

K 和 V 是累积的，Q 每次都是新的！
```

## main.py vs 生产环境

| 方面     | main.py     | 生产环境 (如 vLLM)  |
| -------- | ----------- | ------------------- |
| 存储方式 | Python list | 连续内存块          |
| 内存管理 | 无限制      | PagedAttention 分页 |
| 多batch  | 不支持      | 支持批量缓存        |
| 量化     | 无          | KV Cache 量化       |

## 完整流程图

```
┌─────────────────────────────────────────────────────────────┐
│                    KV Cache 推理流程                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  初始化: keys = [[]], values = [[]]                          │
│                                                              │
│  Step 1:                                                     │
│  ┌──────────────┐                                            │
│  │ 输入 token_0 │                                            │
│  └──────┬───────┘                                            │
│         ↓                                                    │
│  ┌──────────────┐     ┌─────────────┐                       │
│  │ 计算 K0, V0  │ ──→ │ 缓存追加     │                       │
│  └──────┬───────┘     │ keys=[K0]   │                       │
│         │             │ values=[V0] │                       │
│         │             └─────────────┘                       │
│         ↓                                                    │
│  ┌──────────────┐                                            │
│  │ Q0 · K0      │                                            │
│  │ 输出 token_1 │                                            │
│  └──────────────┘                                            │
│                                                              │
│  Step 2:                                                     │
│  ┌──────────────┐                                            │
│  │ 输入 token_1 │                                            │
│  └──────┬───────┘                                            │
│         ↓                                                    │
│  ┌──────────────┐     ┌─────────────────┐                   │
│  │ 计算 K1, V1  │ ──→ │ 缓存追加         │                   │
│  └──────┬───────┘     │ keys=[K0,K1]    │                   │
│         │             │ values=[V0,V1]  │                   │
│         │             └─────────────────┘                   │
│         ↓                                                    │
│  ┌──────────────┐                                            │
│  │ Q1 · [K0,K1] │                                            │
│  │ 输出 token_2 │                                            │
│  └──────────────┘                                            │
│                                                              │
│  ...                                                         │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## 总结

KV Cache 是 Transformer 推理优化的核心技术：

1. **原理**：缓存已计算的 Key 和 Value，避免重复计算
2. **效果**：将推理复杂度从 O(n²) 降到 O(n)
3. **代价**：需要额外内存存储缓存
4. **适用**：所有自回归生成任务（GPT、LLaMA 等）
