# `local_grads` 详解：局部梯度（不是子节点的梯度）

## 核心概念

**`local_grads` = 当前节点对子节点的局部导数**

**重要**：它**不是**子节点的梯度（`child.grad`），而是**当前节点如何影响子节点**的数学关系。

---

## 1. 可视化理解

### 计算图示例

```python
a = Value(2.0)
b = Value(3.0)
c = a + b
```

**计算图**：

```
     a (2.0)          b (3.0)
        └────────┬────────┘
                 ↓
              c (5.0)
           children: (a, b)
      local_grads: (1, 1)  ← 关键！
```

---

### `_children` vs `_local_grads` 的区别

```python
# 对于节点 c = a + b

c._children = (a, b)           # 子节点是谁
c._local_grads = (1, 1)        # c 对 a, b 的局部导数

# 注意：local_grads 不是 a.grad, b.grad
#       而是 ∂c/∂a, ∂c/∂b
```

---

## 2. 数学定义

### `local_grads` 的含义

```python
# 对于 c = a + b
c._local_grads = (1, 1)

# 数学含义：
# 第1个 1: ∂c/∂a = 1  （c 对 a 的导数）
# 第2个 1: ∂c/∂b = 1  （c 对 b 的导数）
```

**关键**：
- `_local_grads[i]` = `∂当前节点/∂_children[i]`
- 它是**当前节点**对**子节点**的导数
- **不是**子节点自己的梯度！

---

## 3. 详细例子

### 例子1：加法

```python
a = Value(2.0)
b = Value(3.0)
c = a + b

# c 对象的状态：
c._children = (a, b)           # 子节点
c._local_grads = (1, 1)        # 局部导数

# 数学含义：
# c._local_grads[0] = 1  →  ∂c/∂a = 1
# c._local_grads[1] = 1  →  ∂c/∂b = 1

# 证明：
# c = a + b
# ∂c/∂a = 1  （a 增加 1，c 增加 1）
# ∂c/∂b = 1  （b 增加 1，c 增加 1）
```

---

### 例子2：乘法

```python
a = Value(2.0)
b = Value(3.0)
c = a * b  # c = 6

# c 对象的状态：
c._children = (a, b)           # 子节点
c._local_grads = (3, 2)        # 局部导数

# 数学含义：
# c._local_grads[0] = 3  →  ∂c/∂a = 3  (= b 的值)
# c._local_grads[1] = 2  →  ∂c/∂b = 2  (= a 的值)

# 证明：
# c = a × b
# ∂c/∂a = b = 3  （a 增加 1，c 增加 b）
# ∂c/∂b = a = 2  （b 增加 1，c 增加 a）
```

**代码对应**：

```python
def __mul__(self, other):
    return Value(
        self.data * other.data,           # 前向传播的值
        (self, other),                    # children
        (other.data, self.data)           # local_grads ← 这里！
    #             ↑         ↑
    #          ∂c/∂a      ∂c/∂b
    #          (= b)      (= a)
    )
```

---

### 例子3：链式计算

```python
a = Value(2.0)
b = Value(3.0)
c = a * b     # c = 6
d = c + 1     # d = 7
L = d * d     # L = 49
```

**计算图**：

```
a(2) ──┐
       ├──→ c(6) ──→ d(7) ──→ L(49)
b(3) ──┘              ↑
                   1(常量)
```

**每个节点的 `local_grads`**：

```python
# 节点 c = a * b
c._children = (a, b)
c._local_grads = (3, 2)  # ∂c/∂a = 3, ∂c/∂b = 2

# 节点 d = c + 1
d._children = (c, Value(1))
d._local_grads = (1, )   # ∂d/∂c = 1

# 节点 L = d * d
L._children = (d, d)     # ← 注意 d 出现两次！
L._local_grads = (7, 7)  # ∂L/∂d = 7, ∂L/∂d = 7
                        # (= d 的值)
```

---

## 4. 反向传播时如何使用 `local_grads`

### 反向传播代码

```python
def backward(self):
    # 拓扑排序...
    self.grad = 1  # dL/dL = 1

    for v in reversed(topo):
        for child, local_grad in zip(v._children, v._local_grads):
            child.grad += local_grad * v.grad
            #            ↑           ↑
            #         局部导数    从后面传来的梯度
```

---

### 逐步追踪

```python
# 初始化
a.grad = 0
b.grad = 0
c.grad = 0
d.grad = 0
L.grad = 1  # dL/dL = 1

# 第1步：处理 L = d * d
# L._local_grads = (7, 7)
# L._children = (d, d)

d.grad += 7 * L.grad  # d.grad = 0 + 7 * 1 = 7
d.grad += 7 * L.grad  # d.grad = 7 + 7 * 1 = 14

# 数学：∂L/∂d = ∂L/∂L × ∂L/∂d = 1 × 7 = 7
# 因为 d 出现两次，所以累积：7 + 7 = 14


# 第2步：处理 d = c + 1
# d._local_grads = (1, )
# d._children = (c, Value(1))

c.grad += 1 * d.grad  # c.grad = 0 + 1 * 14 = 14

# 数学：∂L/∂c = ∂L/∂d × ∂d/∂c = 14 × 1 = 14


# 第3步：处理 c = a * b
# c._local_grads = (3, 2)
# c._children = (a, b)

a.grad += 3 * c.grad  # a.grad = 0 + 3 * 14 = 42
b.grad += 2 * c.grad  # b.grad = 0 + 2 * 14 = 28

# 数学：
# ∂L/∂a = ∂L/∂c × ∂c/∂a = 14 × 3 = 42
# ∂L/∂b = ∂L/∂c × ∂c/∂b = 14 × 2 = 28
```

---

## 5. `local_grads` vs `child.grad` 对比

| 属性 | `local_grads` | `child.grad` |
|------|---------------|--------------|
| **含义** | 当前节点对子节点的导数 | 损失函数对子节点的梯度 |
| **数学符号** | ∂v/∂child | ∂L/∂child |
| **存储位置** | 父节点 `v._local_grads` | 子节点 `child.grad` |
| **何时设置** | 前向传播时 | 反向传播时 |
| **用途** | 链式法则的中间步骤 | 最终想要的梯度 |
| **示例** | `c._local_grads = (1, 1)` | `a.grad = 42` |

---

## 6. 为什么叫"局部"梯度？

### "局部"的含义

**局部** = 只看当前这一步，不考虑整个计算图

```python
# 全局视角（整个计算图）
L = ((a * b) + 1) ** 2
∂L/∂a = ?  # 需要链式法则，经过多个节点

# 局部视角（单个节点）
c = a + b
∂c/∂a = 1  # 只看这一步的导数
```

---

### 类比：接力赛

```
接力赛：
第1棒：a → c  （局部导数：3）
第2棒：c → d  （局部导数：1）
第3棒：d → L  （局部导数：7）

总导数（链式法则）：
∂L/∂a = 7 × 1 × 3 = 21

local_grads 存储的是每一步的"局部导数"：
- 第1棒的 local_grad = 3
- 第2棒的 local_grad = 1
- 第3棒的 local_grad = 7
```

---

## 7. 完整示例演示

```python
import math

class Value:
    def __init__(self, data, children=(), local_grads=()):
        self.data = data
        self.grad = 0
        self._children = children
        self._local_grads = local_grads

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(
            self.data + other.data,
            (self, other),
            (1, 1)  # ∂(a+b)/∂a = 1, ∂(a+b)/∂b = 1
        )

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(
            self.data * other.data,
            (self, other),
            (other.data, self.data)  # ∂(ab)/∂a = b, ∂(ab)/∂b = a
        )

    def __repr__(self):
        return f"Value({self.data})"

    def show_details(self):
        print(f"  data: {self.data}")
        print(f"  grad: {self.grad}")
        print(f"  children: {[c.data for c in self._children]}")
        print(f"  local_grads: {self._local_grads}")


# 创建计算图
a = Value(2.0)
b = Value(3.0)
c = a * b
d = c + 1
L = d * d

print("=== 计算图结构 ===\n")

print("节点 a:")
a.show_details()
print()

print("节点 b:")
b.show_details()
print()

print("节点 c = a * b:")
c.show_details()
print(f"  说明: ∂c/∂a = {c._local_grads[0]} (= b的值)")
print(f"       ∂c/∂b = {c._local_grads[1]} (= a的值)")
print()

print("节点 d = c + 1:")
d.show_details()
print(f"  说明: ∂d/∂c = {d._local_grads[0]}")
print()

print("节点 L = d * d:")
L.show_details()
print(f"  说明: ∂L/∂d = {L._local_grads[0]} (= d的值)")
print(f"       ∂L/∂d = {L._local_grads[1]} (= d的值, d用了2次)")
```

**输出**：
```
=== 计算图结构 ===

节点 a:
  data: 2.0
  grad: 0
  children: []
  local_grads: ()

节点 b:
  data: 3.0
  grad: 0
  children: []
  local_grads: ()

节点 c = a * b:
  data: 6.0
  grad: 0
  children: [2.0, 3.0]
  local_grads: (3.0, 2.0)
  说明: ∂c/∂a = 3.0 (= b的值)
       ∂c/∂b = 2.0 (= a的值)

节点 d = c + 1:
  data: 7.0
  grad: 0
  children: [6.0, 1.0]
  local_grads: (1,)
  说明: ∂d/∂c = 1

节点 L = d * d:
  data: 49.0
  grad: 0
  children: [7.0, 7.0]
  local_grads: (7.0, 7.0)
  说明: ∂L/∂d = 7.0 (= d的值)
       ∂L/∂d = 7.0 (= d的值, d用了2次)
```

---

## 8. 常见运算的 `local_grads`

### 加法：`c = a + b`

```python
c._local_grads = (1, 1)
# 数学：∂c/∂a = 1, ∂c/∂b = 1
```

---

### 乘法：`c = a * b`

```python
c._local_grads = (b.data, a.data)
# 数学：∂c/∂a = b, ∂c/∂b = a
```

---

### 幂运算：`c = a ** 2`

```python
c._local_grads = (2 * a.data,)
# 数学：∂c/∂a = 2a
```

---

### 对数：`c = log(a)`

```python
c._local_grads = (1 / a.data,)
# 数学：∂c/∂a = 1/a
```

---

### 指数：`c = exp(a)`

```python
c._local_grads = (math.exp(a.data),)
# 数学：∂c/∂a = exp(a)
```

---

### ReLU：`c = relu(a) = max(0, a)`

```python
c._local_grads = (1.0 if a.data > 0 else 0.0,)
# 数学：∂c/∂a = 1 if a > 0 else 0
```

---

## 9. 总结

### 核心要点

1. **`local_grads` 不是子节点的梯度**
   - ❌ 不是 `child.grad`
   - ✅ 是 `∂当前节点/∂子节点`

2. **`local_grads` 是局部导数**
   - 只看当前这一步
   - 不考虑整个计算图

3. **`local_grads` 在前向传播时设置**
   - 根据数学公式
   - 依赖于运算类型（加、乘、log 等）

4. **`local_grads` 在反向传播时使用**
   - 通过链式法则计算最终梯度
   - `child.grad += local_grad * v.grad`

---

## 记忆口诀

```
local_grads = 局部导数 = 当前节点对子节点的导数

记忆公式：
local_grads[i] = ∂v/∂children[i]

类比接力赛：
local_grads = 每一棒的速度
child.grad   = 从起点到终点的总速度

关键区别：
- local_grads: 当前这一步的导数（局部）
- child.grad:  整个路径的导数（全局）
```

---

## 最终答案

**`local_grads` 不是子节点的梯度**，而是**当前节点对子节点的局部导数**。

**数学定义**：
```python
local_grads[i] = ∂v/∂children[i]
```

**作用**：
- 在前向传播时设置（根据数学公式）
- 在反向传播时使用（链式法则）

**例子**：
```python
c = a * b
c._local_grads = (3, 2)  # ∂c/∂a = 3, ∂c/∂b = 2
                        # 不是 a.grad, b.grad！
```
