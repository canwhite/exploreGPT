import math
import os
import random

# 设定随机种子
random.seed(42)

# ========== 1. 数据准备 (names.txt) ==========
if not os.path.exists('input.txt'):
    import urllib.request
    names_url = 'https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt'
    urllib.request.urlretrieve(names_url, 'input.txt')

docs = [line.strip() for line in open('input.txt') if line.strip()]
random.shuffle(docs)

uchars = sorted(set(''.join(docs)))
BOS = len(uchars)
vocab_size = len(uchars) + 1
char_to_it = {ch: i for i, ch in enumerate(uchars)}
it_to_char = {i: ch for i, ch in enumerate(uchars)}
it_to_char[BOS] = '.'

# ========== 2. 自动微分系统 (Micrograd) ==========
class Value:
    __slots__ = ('data', 'grad', '_children', '_local_grads')
    def __init__(self, data, children=(), local_grads=()):
        self.data, self.grad = data, 0
        self._children, self._local_grads = children, local_grads

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), (1, 1))

    def __radd__(self, other): return self + other
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), (other.data, self.data))
    def __rmul__(self, other): return self * other
    def __pow__(self, other):
        return Value(self.data**other, (self,), (other * self.data**(other-1),))
    def log(self): return Value(math.log(self.data + 1e-10), (self,), (1/(self.data + 1e-10),))
    def exp(self): return Value(math.exp(self.data), (self,), (math.exp(self.data),))
    def relu(self): return Value(max(0, self.data), (self,), (float(self.data > 0),))
    def __neg__(self): return self * -1
    def __sub__(self, other): return self + (-other)
    def __truediv__(self, other): return self * other**-1

    def backward(self):
        topo, visited = [], set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._children: build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1
        for v in reversed(topo):
            for child, local_grad in zip(v._children, v._local_grads):
                child.grad += local_grad * v.grad

# ========== 3. TTT-GPT 架构配置 ==========
n_layer = 1
n_embd = 16
learning_rate = 0.01 # Backbone 学习率
ttt_lr = 0.4         # TTT 动态更新步长

def init_matrix(nout, nin):
    return [[Value(random.gauss(0, 0.1)) for _ in range(nin)] for _ in range(nout)]

# 静态主干网 (长期记忆/通用知识)
state_dict = {
    'wte': init_matrix(vocab_size, n_embd),
    'lm_head': init_matrix(vocab_size, n_embd),
}
for i in range(n_layer):
    state_dict[f'layer{i}.mlp'] = init_matrix(n_embd, n_embd)
    state_dict[f'layer{i}.mlp_out'] = init_matrix(n_embd, n_embd)

# ========== 4. TTT-Layer 核心演算法 ==========

def rmsnorm(x):
    ms = sum((xi * xi for xi in x), Value(0)) / len(x)
    scale = (ms + 1e-5) ** -0.5
    return [xi * scale for xi in x]

def linear(x, w):
    return [sum((wi * xi for wi, xi in zip(wo, x)), Value(0)) for wo in w]

def ttt_layer_step(x, W_hidden):
    """
    真正的 TTT 算子：
    1. 线性变换作为推理 (RNN-like hidden state interaction)
    2. 自监督重建任务更新 W_hidden (Gradient descent as state transition)
    """
    # --- Step 1: 推理 (获取当前 token 的特征表示) ---
    y = linear(x, W_hidden)

    # --- Step 2: 学习 (将当前 token 存入权重) ---
    # 定义目标：W 应该能够重建 x
    x_hat = linear(x, W_hidden)
    ttt_loss = sum(((xh - xi)**2 for xh, xi in zip(x_hat, x)), Value(0))

    # 局部反向传播，仅针对隐藏矩阵 W_hidden
    ttt_loss.backward()

    # 更新隐藏状态权重 (这就是 TTT 的“状态转移”)
    for row in W_hidden:
        for p in row:
            p.data -= ttt_lr * p.grad
            p.grad = 0 # 必须清空，否则下一时刻梯度会累加

    return y

# ========== 5. 模型前向传播流程 (彻底无 KV Cache) ==========

def model_forward(tokens, ttt_states):
    """
    tokens: token 列表
    ttt_states: 每一层的权重矩阵列表 [n_layer, n_embd, n_embd]
    """
    logits_seq = []
    for token_id in tokens:
        x = state_dict['wte'][token_id]

        for i in range(n_layer):
            # TTT 模块 (替代 Attention，无 KV Cache 增长)
            x_res = x
            x = rmsnorm(x)
            x = ttt_layer_step(x, ttt_states[i])
            x = [ai + bi for ai, bi in zip(x, x_res)]

            # MLP 模块
            x_res = x
            x = rmsnorm(x)
            x = linear([xi.relu() for xi in linear(x, state_dict[f'layer{i}.mlp'])],
                       state_dict[f'layer{i}.mlp_out'])
            x = [ai + bi for ai, bi in zip(x, x_res)]

        logits = linear(x, state_dict['lm_head'])
        logits_seq.append(logits)

    return logits_seq

# ========== 6. 训练循环 (Pre-training Backbone) ==========

print("=== 开始预训练 TTT-Backbone (学习通用语言规律) ===")
for step in range(301):
    doc = docs[step % len(docs)]
    tokens = [char_to_it[c] for c in doc] + [BOS]

    # 每个序列开始时，初始化 TTT 矩阵为单位阵 (表示初始记忆为空)
    ttt_states = [[[Value(1.0 if i==j else 0.0) for j in range(n_embd)]
                    for i in range(n_embd)] for _ in range(n_layer)]

    # 前向计算
    logits_seq = model_forward(tokens[:-1], ttt_states)

    # 交叉熵损失
    total_loss = Value(0)
    for i, target in enumerate(tokens[1:]):
        logits = logits_seq[i]
        max_l = max(l.data for l in logits)
        sum_exp = sum(((l - max_l).exp() for l in logits), Value(0))
        loss = (sum_exp.log() + max_l) - logits[target]
        total_loss = total_loss + loss

    avg_loss = total_loss / (len(tokens) - 1)
    avg_loss.backward()

    # 更新静态 Backbone 参数 (SGD)
    for mat in state_dict.values():
        for row in mat:
            for p in row:
                p.data -= learning_rate * p.grad
                p.grad = 0

    if step % 50 == 0:
        print(f"Step {step:3d} | Loss: {avg_loss.data:.4f}")

# ========== 7. 最终推理生成 (完全无 KV 缓存模式) ==========

def generate(prefix_str, max_len=10):
    # 1. 状态初始化 (State as Weights)
    ttt_states = [[[Value(1.0 if i==j else 0.0) for j in range(n_embd)]
                    for i in range(n_embd)] for _ in range(n_layer)]

    # 2. 处理前缀并注入状态
    tokens = [char_to_it[c] for c in prefix_str]
    # 通过 model_forward 处理前缀，ttt_states 会在此过程中被实时“训练”
    _ = model_forward(tokens, ttt_states)

    result = prefix_str
    curr_token = tokens[-1]

    for _ in range(max_len):
        # 核心：只输入当前 token，记忆全在 ttt_states 矩阵里
        # 这就是线性递归 (Linear Recurrence) 的魅力
        logits = model_forward([curr_token], ttt_states)[0]

        # 贪婪采样
        next_id = 0
        mv = -1e10
        for i, v in enumerate(logits):
            if v.data > mv:
                mv = v.data
                next_id = i

        if next_id == BOS: break
        char = it_to_char[next_id]
        result += char
        curr_token = next_id

    return result

print("\n=== TTT 推理测试 (无 KV Cache) ===")
for p in ["ma", "li", "jo"]:
    print(f"前缀 '{p}' -> 生成结果: '{generate(p)}'")
