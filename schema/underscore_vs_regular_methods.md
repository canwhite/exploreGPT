# 带下划线 vs 不带下划线的方法 + 元编程讲解

## 问题1：没有 `__` 的方法需要手动调用吗？

### 简短答案

**是的！** 不带 `__` 的普通方法需要**手动调用**。

---

## 1. 对比：魔术方法 vs 普通方法

### 魔术方法（自动调用）

```python
class Value:
    def __add__(self, other):  # ← 带 __
        return self.data + other.data

a = Value(5.0)
b = Value(3.0)

# 使用运算符 → 自动调用
c = a + b  # Python 自动调用 a.__add__(b)
```

**特点**：
- ✅ 名字固定（如 `__add__`, `__mul__`）
- ✅ 使用运算符自动触发
- ✅ 定义在 Python 语言规范中
- ✅ 由 Python 解释器调用

---

### 普通方法（手动调用）

```python
class Value:
    def add(self, other):  # ← 不带 __
        return self.data + other.data

a = Value(5.0)
b = Value(3.0)

# 必须手动调用
c = a.add(b)  # ← 手动调用，不能用 a + b
```

**特点**：
- ✅ 名字自定义（如 `add`, `calculate`, `process`）
- ✅ 必须**显式调用**
- ✅ 你自己定义的业务逻辑
- ✅ 由你（程序员）调用

---

## 2. 详细对比表

| 特性 | 魔术方法 (`__add__`) | 普通方法 (`add`) |
|------|---------------------|----------------|
| **命名** | 固定（`__add__`） | 自定义（`add`） |
| **调用方式** | `a + b`（自动） | `a.add(b)`（手动） |
| **调用者** | Python 解释器 | 程序员 |
| **用途** | 运算符重载 | 业务逻辑 |
| **数量** | 有限（语言定义） | 无限（自定义） |
| **文档位置** | Python 语言规范 | 你的代码文档 |

---

## 3. 实际例子对比

### 例子1：加法

```python
class Calculator:
    # 魔术方法：可以用 + 运算符
    def __add__(self, other):
        print("调用 __add__")
        return self.value + other.value

    # 普通方法：必须手动调用
    def add(self, other):
        print("调用 add")
        return self.value + other.value

    def __init__(self, value):
        self.value = value


a = Calculator(5.0)
b = Calculator(3.0)

# 使用魔术方法（自动）
c1 = a + b
# 输出: 调用 __add__

# 使用普通方法（手动）
c2 = a.add(b)
# 输出: 调用 add
```

---

### 例子2：字符串表示

```python
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    # 魔术方法：print() 自动调用
    def __str__(self):
        return f"Person({self.name})"

    # 普通方法：手动调用
    def describe(self):
        return f"{self.name} is {self.age} years old"


p = Person("Alice", 25)

# 使用魔术方法（自动）
print(p)  # 自动调用 p.__str__()
# 输出: Person(Alice)

# 使用普通方法（手动）
print(p.describe())  # 手动调用
# 输出: Alice is 25 years old
```

---

### 例子3：长度

```python
class MyList:
    def __init__(self, items):
        self.items = items

    # 魔术方法：len() 自动调用
    def __len__(self):
        return len(self.items)

    # 普通方法：手动调用
    def count(self):
        return len(self.items)


lst = MyList([1, 2, 3, 4, 5])

# 使用魔术方法（自动）
print(len(lst))  # 自动调用 lst.__len__()
# 输出: 5

# 使用普通方法（手动）
print(lst.count())  # 手动调用
# 输出: 5
```

---

## 4. 常见魔术方法 vs 普通方法

### 数学运算

```python
# 魔术方法（自动）
a + b     → a.__add__(b)
a - b     → a.__sub__(b)
a * b     → a.__mul__(b)
a / b     → a.__truediv__(b)
a ** b    → a.__pow__(b)

# 普通方法（手动）
a.add(b)
a.subtract(b)
a.multiply(b)
a.divide(b)
a.power(b)
```

### 比较

```python
# 魔术方法（自动）
a == b    → a.__eq__(b)
a > b     → a.__gt__(b)
a < b     → a.__lt__(b)

# 普通方法（手动）
a.equals(b)
a.greater_than(b)
a.less_than(b)
```

### 容器操作

```python
# 魔术方法（自动）
len(a)       → a.__len__()
a[i]         → a.__getitem__(i)
a[i] = x     → a.__setitem__(i, x)
for x in a:  → a.__iter__()

# 普通方法（手动）
a.length()
a.get(i)
a.set(i, x)
a.iterator()
```

---

## 5. 什么时候用哪种？

### 用魔术方法（`__`）当：

✅ 你想重载运算符
```python
class Vector:
    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y)

v1 = Vector(1, 2)
v2 = Vector(3, 4)
v3 = v1 + v2  # ← 自然的语法
```

✅ 你想自定义对象的行为
```python
class Person:
    def __str__(self):
        return f"My name is {self.name}"

print(p)  # ← 自定义输出
```

✅ 你想让对象像内置类型一样使用
```python
class MyList:
    def __getitem__(self, index):
        return self.items[index]

lst = MyList([1, 2, 3])
print(lst[0])  # ← 像列表一样使用
```

---

### 用普通方法（无 `__`）当：

✅ 实现业务逻辑
```python
class User:
    def send_email(self, message):
        # 发送邮件的逻辑
        pass

user.send_email("Hello!")
```

✅ 封装功能
```python
class Database:
    def connect(self):
        pass

    def query(self, sql):
        pass

    def close(self):
        pass
```

✅ 提供API
```python
class API:
    def get_user(self, user_id):
        pass

    def create_user(self, data):
        pass
```

---

## 6. 命名约定总结

### Python 的命名约定

```python
class MyClass:
    # 1. 无下划线：公共方法
    def public_method(self):
        pass

    # 2. 单下划线前缀：内部使用（约定）
    def _internal_method(self):
        pass

    # 3. 双下划线前缀：名称改写（私有）
    def __private_method(self):
        pass

    # 4. 双下划线前后：魔术方法
    def __magic_method__(self):
        pass
```

---

## 问题2：`__` 是元编程吗？

### 简短答案

**是的！** 魔术方法是元编程的一种形式。

---

## 1. 什么是元编程？

**元编程** = "编写能够操作代码的代码"

或者：**程序在运行时修改自己的行为**

---

## 2. 元编程的层次

### 层次0：普通编程
```python
# 普通代码：操作数据
a = 5
b = 3
c = a + b  # 操作数据
```

### 层次1：使用元编程工具
```python
# 装饰器：修改函数行为
@timer
def my_function():
    pass

# 元类：修改类的创建行为
class Meta(type):
    def __new__(cls, name, bases, dct):
        # 修改类的创建过程
        pass
```

### 层次2：魔术方法（轻量级元编程）
```python
class Value:
    def __add__(self, other):
        # 改变 + 运算符的行为
        return self.data + other.data

a = Value(5.0)
b = Value(3.0)
c = a + b  # + 的行为被改变了！
```

---

## 3. 为什么魔术方法是元编程？

### 原因1：改变了语言的基本行为

```python
# 默认行为
5 + 3  # → 8（整数加法）

# 自定义行为
class Value:
    def __add__(self, other):
        return "自定义加法"

a = Value()
b = Value()
a + b  # → "自定义加法"（行为被改变！）
```

**你改变了 `+` 运算符的含义！** → 这就是元编程

---

### 原因2：让自定义类型拥有内置类型的特性

```python
# 内置类型
len([1, 2, 3])  # → 3

# 自定义类型（通过元编程）
class MyList:
    def __len__(self):
        return 100

lst = MyList()
len(lst)  # → 100（自定义行为！）
```

**你让 `len()` 能作用于你的类型！** → 这就是元编程

---

### 原因3：动态定义对象行为

```python
class Dynamic:
    def __getattr__(self, name):
        # 动态生成方法
        def method(*args, **kwargs):
            return f"调用 {name}"
        return method


obj = Dynamic()
print(obj.any_method())  # → "调用 any_method"
# 这个方法根本不存在，但能调用！
```

**运行时动态创建行为！** → 这就是元编程

---

## 4. 元编程的类型对比

### 类型1：魔术方法（最简单）

```python
class Value:
    def __add__(self, other):
        return self.data + other.data
```

**特点**：
- ✅ 简单易懂
- ✅ Python 内置支持
- ✅ 改变运算符行为

---

### 类型2：装饰器

```python
def my_decorator(cls):
    # 修改类
    cls.new_attr = "added"
    return cls

@my_decorator
class MyClass:
    pass

print(MyClass.new_attr)  # → "added"
```

**特点**：
- ⚠️ 中等复杂度
- ⚠️ 修改函数/类行为
- ⚠️ 语法糖

---

### 类型3：元类

```python
class Meta(type):
    def __new__(cls, name, bases, dct):
        # 拦截类的创建
        dct['created_by'] = 'Meta'
        return super().__new__(cls, name, bases, dct)


class MyClass(metaclass=Meta):
    pass

print(MyClass.created_by)  # → "Meta"
```

**特点**：
- ❌ 复杂
- ❌ 控制类的创建
- ❌ 高级元编程

---

## 5. 元编程的层次图

```
普通编程
    ↓
装饰器（简单元编程）
    ↓
魔术方法（中等元编程）
    ↓
元类（高级元编程）
    ↓
代码生成（完全元编程）
```

---

## 6. 实际应用示例

### 应用1：ORM（对象关系映射）

```python
class Model:
    def __getattr__(self, name):
        # 动态生成数据库查询
        if name.startswith('get_by_'):
            field = name[6:]  # get_by_name → name
            def query(value):
                return f"SELECT * FROM {self.table} WHERE {field} = {value}"
            return query


class User(Model):
    table = "users"


user = User()
print(user.get_by_name("Alice"))
# → SELECT * FROM users WHERE name = Alice
```

**元编程**：动态生成方法！

---

### 应用2：API 客户端

```python
class API:
    def __getattr__(self, name):
        # 动态生成 API 调用
        def call(*args, **kwargs):
            endpoint = f"/{name}"
            return f"调用 API: {endpoint}"
        return call


api = API()
print(api.get_user(1))    # → 调用 API: /get_user
print(api.create_user())  # → 调用 API: /create_user
```

**元编程**：不需要显式定义每个 API 方法！

---

### 应用3：领域特定语言（DSL）

```python
class Query:
    def __init__(self):
        self.filters = []

    def __eq__(self, other):
        self.filters.append(f"= {other}")
        return self

    def __gt__(self, other):
        self.filters.append(f"> {other}")
        return self


# 使用 DSL
q = Query()
q.age == 25  # 不是比较，而是添加过滤器！
q.salary > 50000
print(q.filters)  # → ["= 25", "> 50000"]
```

**元编程**：创造自定义语法！

---

## 7. 总结

### 问题1答案：没有 `__` 的方法

```python
# 魔术方法：自动调用
a + b      → a.__add__(b)      ← Python 解释器调用
len(a)     → a.__len__()       ← Python 解释器调用
print(a)   → a.__str__()       ← Python 解释器调用

# 普通方法：手动调用
a.add(b)   ← 你手动调用
a.count()  ← 你手动调用
a.describe()  ← 你手动调用
```

**规则**：
- `__` 前后 = 魔术方法 = **自动调用**
- 无 `__` = 普通方法 = **手动调用**

---

### 问题2答案：`__` 是元编程吗？

**是的！** 魔术方法是元编程的一种形式。

**元编程层次**：
```
1. 装饰器      → 简单元编程
2. 魔术方法    → 中等级别（改变对象行为）
3. 元类        → 高级元编程（控制类创建）
4. 代码生成    → 完全元编程
```

**魔术方法是元编程，因为**：
- ✅ 改变了语言的基本行为（如 `+` 运算符）
- ✅ 让自定义类型拥有内置类型的特性
- ✅ 动态定义对象的行为

---

## 记忆口诀

```
魔术方法 vs 普通方法：

__method__  → Python 自动调用（运算符、内置函数）
method      → 你手动调用（业务逻辑）

元编程等级：

装饰器     → 改变函数行为
魔术方法   → 改变对象行为（轻量级元编程）
元类       → 改变类创建行为（重量级元编程）
```

---

**最终答案**：

1. **没有 `__` 的方法需要手动调用**，带 `__` 的魔术方法由 Python 自动调用
2. **`__` 是元编程的一种形式**，它允许你改变对象的行为，让自定义类型像内置类型一样工作
