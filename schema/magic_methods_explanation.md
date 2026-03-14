# Python 魔术方法详解：`__` 方法何时使用？

## 目录
- [什么是魔术方法？](#什么是魔术方法)
- [算术运算方法](#算术运算方法)
- [反向运算方法](#反向运算方法)
- [实际应用场景](#实际应用场景)
- [完整示例演示](#完整示例演示)

---

## 什么是魔术方法？

**魔术方法**（Magic Methods）是 Python 中以 `__` 开头和结尾的特殊方法，允许对象自定义其行为。

```python
# 普通方法
obj.method()  # 显式调用

# 魔术方法
obj + other   # 隐式调用 __add__
```

---

## 算术运算方法

### 1. `__add__` - 加法

**何时使用**：当对象出现在 `+` 运算符**左边**时

```python
class Value:
    def __add__(self, other):
        print("__add__ 被调用")
        return Value(self.data + other.data)

a = Value(2.0)
b = Value(3.0)

# 场景1: a + b
c = a + b
# 输出: __add__ 被调用
# 因为 a 在左边，所以调用 a.__add__(b)

# 场景2: a + 5
d = a + 5
# 输出: __add__ 被调用
# 调用 a.__add__(5)，然后 __add__ 内部将 5 转换成 Value(5)
```

**关键点**：
- `a + b` → `a.__add__(b)`
- `a` 是调用者（self），`b` 是参数（other）

---

### 2. `__mul__` - 乘法

**何时使用**：当对象出现在 `*` 运算符**左边**时

```python
# 场景1: a * b
c = a * b
# 调用 a.__mul__(b)

# 场景2: a * 2
d = a * 2
# 调用 a.__mul__(2)
# 内部处理: return Value(self.data * other.data, (self, other), (other.data, self.data))
```

**关键点**：
- `a * b` → `a.__mul__(b)`
- `a` 是调用者（self），`b` 是参数（other）

---

### 3. `__pow__` - 幂运算

**何时使用**：当对象出现在 `**` 运算符**左边**时

```python
# 场景: a ** 2
c = a ** 2
# 调用 a.__pow__(2)
# 计算: self.data ** other = 2.0 ** 2 = 4.0
```

---

### 4. `__neg__` - 取负

**何时使用**：当对象前面有 `-` 运算符时

```python
# 场景: -a
c = -a
# 调用 a.__neg__()
# 等价于: a * -1
```

**为什么这样实现**：
```python
def __neg__(self):
    return self * -1
# 复用了 __mul__，避免重复代码
```

---

## 反向运算方法（重要！）

### 为什么需要反向运算？

当你的对象在运算符**右边**，而左边是**不支持该运算的类型**时，Python 会尝试调用反向方法。

### 5. `__radd__` - 反向加法

**何时使用**：当 `左边的对象没有 __add__` 或者 `__add__ 返回 NotImplemented` 时

```python
class Value:
    def __add__(self, other):
        print(f"__add__: self={self.data}, other={other}")
        if isinstance(other, Value):
            return Value(self.data + other.data)
        else:
            return Value(self.data + other)

    def __radd__(self, other):
        print(f"__radd__: self={self.data}, other={other}")
        # other + self
        # 反过来调用 __add__
        return self.__add__(other)

a = Value(2.0)

# 场景1: 5 + a  ← 注意：a 在右边！
b = 5 + a
# 尝试流程：
# 1. 先尝试 int.__add__(5, a) → 整数不知道怎么加 Value
# 2. 返回 NotImplemented
# 3. Python 尝试调用 a.__radd__(5)
# 输出: __radd__: self=2.0, other=5
# 结果: 7.0
```

**对比**：
```python
# a 在左边：调用 __add__
a + 5  → a.__add__(5)

# a 在右边：调用 __radd__
5 + a  → a.__radd__(5)
```

---

### 6. `__rmul__` - 反向乘法

**何时使用**：当对象在 `*` 运算符**右边**时

```python
# 场景: 2 * a
b = 2 * a
# 尝试流程：
# 1. 先尝试 int.__mul__(2, a) → 失败
# 2. 调用 a.__rmul__(2)
# 结果: 4.0
```

---

### 7. `__rsub__` - 反向减法

**何时使用**：当对象在 `-` 运算符**右边**时

```python
class Value:
    def __sub__(self, other):
        # a - b
        return self + (-other)  # 等价于 a + (-b)

    def __rsub__(self, other):
        # 5 - a  ← other 在左边，self 在右边
        # 应该变成: other - self
        return other + (-self)

a = Value(2.0)

# 场景1: a - 5
b = a - 5
# 调用 a.__sub__(5)
# 结果: -3.0

# 场景2: 5 - a
c = 5 - a
# 调用 a.__rsub__(5)
# 结果: 3.0
```

**注意减法的方向**：
```python
a - b  → a.__sub__(b)   → self + (-other)
5 - a  → a.__rsub__(5)  → other + (-self)
```

---

### 8. `__truediv__` 和 `__rtruediv__` - 除法

**何时使用**：对象出现在 `/` 运算符的左边或右边

```python
class Value:
    def __truediv__(self, other):
        # a / b
        return self * other**-1  # 等价于 a * (b^-1)

    def __rtruediv__(self, other):
        # 5 / a
        return other * self**-1

a = Value(2.0)

# 场景1: a / 2
b = a / 2
# 调用 a.__truediv__(2)
# 计算: 2.0 * 2^(-1) = 2.0 * 0.5 = 1.0

# 场景2: 10 / a
c = 10 / a
# 调用 a.__rtruediv__(10)
# 计算: 10 * 2.0^(-1) = 10 * 0.5 = 5.0
```

---

## 实际应用场景

### 场景1：自动微分中的运算

```python
# 在 microGPT 中
a = Value(2.0)
b = Value(3.0)

# 前向传播
c = a + b  # 调用 a.__add__(b)，创建计算图节点
d = c * 2  # 调用 c.__mul__(2)
L = d * d  # 调用 d.__mul__(d)

# 反向传播
L.backward()  # 使用计算图中的 _children 和 _local_grads
```

**关键**：
- 每次运算都创建新的 `Value` 对象
- 记录 `_children`（父母节点）和 `_local_grads`（局部导数）
- 支持链式法则

---

### 场景2：混合运算（数字 + Value）

```python
a = Value(5.0)

# 这些都能工作，因为实现了反向方法：
b = a + 3   # a.__add__(3)
c = 3 + a   # a.__radd__(3)
d = a * 2   # a.__mul__(2)
e = 2 * a   # a.__rmul__(2)
f = a - 1   # a.__sub__(1)
g = 10 - a  # a.__rsub__(10)
h = a / 5   # a.__truediv__(5)
i = 10 / a  # a.__rtruediv__(10)
```

---

### 场景3：链式运算

```python
# 因为每个运算都返回新的 Value 对象
a = Value(2.0)
b = Value(3.0)
c = Value(4.0)

# 可以链式调用
result = ((a + b) * c) / 2
# 1. a + b → 调用 a.__add__(b)
# 2. (a+b) * c → 调用 (a+b).__mul__(c)
# 3. ((a+b)*c) / 2 → 调用 ((a+b)*c).__truediv__(2)
```

---

## 完整示例演示

```python
import math

class Value:
    def __init__(self, data):
        self.data = data

    def __repr__(self):
        return f"Value({self.data})"

    # ===== 加法 =====
    def __add__(self, other):
        print(f"  __add__: {self.data} + {other}")
        if isinstance(other, Value):
            return Value(self.data + other.data)
        return Value(self.data + other)

    def __radd__(self, other):
        print(f"  __radd__: {other} + {self.data}")
        return self.__add__(other)

    # ===== 乘法 =====
    def __mul__(self, other):
        print(f"  __mul__: {self.data} * {other}")
        if isinstance(other, Value):
            return Value(self.data * other.data)
        return Value(self.data * other)

    def __rmul__(self, other):
        print(f"  __rmul__: {other} * {self.data}")
        return self.__mul__(other)

    # ===== 减法 =====
    def __sub__(self, other):
        print(f"  __sub__: {self.data} - {other}")
        if isinstance(other, Value):
            return Value(self.data - other.data)
        return Value(self.data - other)

    def __rsub__(self, other):
        print(f"  __rsub__: {other} - {self.data}")
        return Value(other - self.data)

    # ===== 除法 =====
    def __truediv__(self, other):
        print(f"  __truediv__: {self.data} / {other}")
        if isinstance(other, Value):
            return Value(self.data / other.data)
        return Value(self.data / other)

    def __rtruediv__(self, other):
        print(f"  __rtruediv__: {other} / {self.data}")
        return Value(other / self.data)

    # ===== 幂运算 =====
    def __pow__(self, other):
        print(f"  __pow__: {self.data} ** {other}")
        return Value(self.data ** other)

    # ===== 取负 =====
    def __neg__(self):
        print(f"  __neg__: -{self.data}")
        return Value(-self.data)


# 测试
print("=== 测试 1: 基本加法 ===")
a = Value(5.0)
b = Value(3.0)
c = a + b
print(f"结果: {c}\n")

print("=== 测试 2: 反向加法 ===")
d = 10 + a
print(f"结果: {d}\n")

print("=== 测试 3: 混合运算 ===")
e = a * 2
f = 2 * a
print(f"a * 2 = {e}")
print(f"2 * a = {f}\n")

print("=== 测试 4: 减法 ===")
g = a - 2
h = 10 - a
print(f"a - 2 = {g}")
print(f"10 - a = {h}\n")

print("=== 测试 5: 除法 ===")
i = a / 2
j = 10 / a
print(f"a / 2 = {i}")
print(f"10 / a = {j}\n")

print("=== 测试 6: 复杂表达式 ===")
result = ((a + b) * 2) / 5
print(f"((a + b) * 2) / 5 = {result}")
```

**输出**：
```
=== 测试 1: 基本加法 ===
  __add__: 5.0 + 3.0
结果: Value(8.0)

=== 测试 2: 反向加法 ===
  __radd__: 10 + 5.0
  __add__: 5.0 + 10
结果: Value(15.0)

=== 测试 3: 混合运算 ===
  __mul__: 5.0 * 2
a * 2 = Value(10.0)
  __rmul__: 2 * 5.0
  __mul__: 5.0 * 2
2 * a = Value(10.0)

=== 测试 4: 减法 ===
  __sub__: 5.0 - 2
a - 2 = Value(3.0)
  __rsub__: 10 - 5.0
10 - a = Value(5.0)

=== 测试 5: 除法 ===
  __truediv__: 5.0 / 2
a / 2 = Value(2.5)
  __rtruediv__: 10 / 5.0
10 / a = Value(2.0)

=== 测试 6: 复杂表达式 ===
  __add__: 5.0 + 3.0
  __mul__: 8.0 * 2
  __truediv__: 16.0 / 5
((a + b) * 2) / 5 = Value(3.2)
```

---

## 总结表

| 运算符 | 正向方法 | 反向方法 | 使用场景 | 示例 |
|--------|---------|---------|---------|------|
| `+` | `__add__` | `__radd__` | 对象在左/右 | `a + b` / `2 + a` |
| `-` | `__sub__` | `__rsub__` | 对象在左/右 | `a - b` / `5 - a` |
| `*` | `__mul__` | `__rmul__` | 对象在左/右 | `a * b` / `2 * a` |
| `/` | `__truediv__` | `__rtruediv__` | 对象在左/右 | `a / b` / `10 / a` |
| `**` | `__pow__` | `__rpow__` | 对象在左边 | `a ** 2` |
| `-` | `__neg__` | 无 | 取负 | `-a` |

**记忆口诀**：

```
正向方法：对象在左边
反向方法：对象在右边（左边类型不支持时）

a + 2  → __add__   (a 在左)
2 + a  → __radd__  (a 在右)
```

---

**关键要点**：

1. **正向方法**（`__add__` 等）：对象在运算符**左边**时调用
2. **反向方法**（`__radd__` 等）：对象在运算符**右边**时调用
3. **反向方法的作用**：让对象能和普通数字自由混合运算
4. **在 microGPT 中**：这些方法让 `Value` 对象可以像普通数字一样进行数学运算，同时构建计算图
