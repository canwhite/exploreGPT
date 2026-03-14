import math
import os
import random

# ========== 1. 核心微分引擎 (Micrograd) ==========
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

# ========== 2. 算子定义 ==========
def rmsnorm(x):
    ms = sum((xi * xi for xi in x), Value(0)) / len(x)
    return [xi * ((ms + 1e-5) ** -0.5) for xi in x]

def linear(x, w):
    return [sum((wi * xi for wi, xi in zip(wo, x)), Value(0)) for wo in w]

def softmax(logits):
    mv = max(val.data for val in logits)
    exps = [(val - mv).exp() for val in logits]
    total = sum(exps, Value(0))
    return [e / total for e in exps]

# ========== 3. Hybrid 层 (TTT & Attention) ==========
def ttt_step(x, W_hidden, ttt_lr=0.2):
    y = linear(x, W_hidden) # 这里的线性变换就是 RNN 式的状态提取
    # 自监督更新：让 W 学会重建 x
    x_hat = linear(x, W_hidden)
    loss = sum(((xh - xi)**2 for xh, xi in zip(x_hat, x)), Value(0))
    loss.backward()
    for row in W_hidden:
        for p in row:
            p.data -= ttt_lr * p.grad
            p.grad = 0
    return y

def attention_step(q, k_cache, v_cache):
    d_k = len(q)
    scores = [sum((qi * ki for qi, ki in zip(q, k)), Value(0)) / (d_k**0.5) for k in k_cache]
    weights = softmax(scores)
    return [sum((weights[t] * v_cache[t][j] for t in range(len(v_cache))), Value(0)) for j in range(d_k)]

# ========== 4. 模型参数初始化 ==========
random.seed(42)
n_embd = 12
vocab_size = 27 # a-z + BOS
char_to_it = {ch: i for i, ch in enumerate("abcdefghijklmnopqrstuvwxyz")}
it_to_char = {i: ch for i, ch in enumerate("abcdefghijklmnopqrstuvwxyz")}
char_to_it['.'] = 26; it_to_char[26] = '.'
BOS = 26

state_dict = {
    'wte': [[Value(random.gauss(0, 0.1)) for _ in range(n_embd)] for _ in range(vocab_size)],
    'lm_head': [[Value(random.gauss(0, 0.1)) for _ in range(n_embd)] for _ in range(vocab_size)],
    'attn_q': [[Value(random.gauss(0, 0.1)) for _ in range(n_embd)] for _ in range(n_embd)],
    'attn_k': [[Value(random.gauss(0, 0.1)) for _ in range(n_embd)] for _ in range(n_embd)],
    'attn_v': [[Value(random.gauss(0, 0.1)) for _ in range(n_embd)] for _ in range(n_embd)],
    'mlp': [[Value(random.gauss(0, 0.1)) for _ in range(n_embd)] for _ in range(n_embd)]
}

# ========== 5. 混合前向传播逻辑 ==========
def forward(token_id, ttt_states, kv_cache):
    x = state_dict['wte'][token_id]

    # Layer 0: TTT (长程记忆矩阵)
    x = [ai + bi for ai, bi in zip(ttt_step(rmsnorm(x), ttt_states[0]), x)]

    # Layer 1: Attention (短程查表)
    x_norm = rmsnorm(x)
    q = linear(x_norm, state_dict['attn_q'])
    k = linear(x_norm, state_dict['attn_k'])
    v = linear(x_norm, state_dict['attn_v'])
    kv_cache['k'].append(k); kv_cache['v'].append(v)
    x = [ai + bi for ai, bi in zip(attention_step(q, kv_cache['k'], kv_cache['v']), x)]

    # Layer 2: MLP
    x = [ai + bi for ai, bi in zip(linear([xi.relu() for xi in linear(rmsnorm(x), state_dict['mlp'])], state_dict['mlp']), x)]

    return linear(x, state_dict['lm_head'])

# ========== 6. 极速预训练 (过拟合几个单词以验证逻辑) ==========
dataset = ["zack", "jack", "mary"]
print("=== 开始 Hybrid 预训练 (微型过拟合测试) ===")
for step in range(151):
    word = random.choice(dataset)
    tokens = [char_to_it[c] for c in word] + [BOS]

    ttt_states = [[[Value(1.0 if i==j else 0.0) for j in range(n_embd)] for i in range(n_embd)]]
    kv_cache = {'k': [], 'v': []}

    loss = Value(0)
    for t in range(len(tokens)-1):
        logits = forward(tokens[t], ttt_states, kv_cache)
        probs = softmax(logits)
        loss = loss + (-probs[tokens[t+1]].log())

    loss = loss / (len(tokens)-1)
    loss.backward()

    for key in state_dict:
        for row in state_dict[key]:
            for p in row:
                p.data -= 0.05 * p.grad
                p.grad = 0
    if step % 50 == 0: print(f"Step {step} | Loss: {loss.data:.4f}")

# ========== 7. 推理生成 ==========
def generate(prefix):
    ttt_states = [[[Value(1.0 if i==j else 0.0) for j in range(n_embd)] for i in range(n_embd)]]
    kv_cache = {'k': [], 'v': []}

    tokens = [char_to_it[c] for c in prefix]
    for t in tokens[:-1]: forward(t, ttt_states, kv_cache)

    res = prefix
    curr = tokens[-1]
    for _ in range(6):
        logits = forward(curr, ttt_states, kv_cache)
        next_id = max(range(vocab_size), key=lambda i: logits[i].data)
        if next_id == BOS: break
        res += it_to_char[next_id]
        curr = next_id
    return res

print("\n=== 推理测试 ===")
for p in ["za", "ma", "ja"]:
    print(f"Prefix: '{p}' -> Result: '{generate(p)}'")
