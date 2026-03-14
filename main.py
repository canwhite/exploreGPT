import math
import os
import random

random.seed(42)

# ========== 1. 数据获取（参考 micro.py）==========
if not os.path.exists('input.txt'):
    import urllib.request
    names_url = 'https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt'
    urllib.request.urlretrieve(names_url, 'input.txt')

docs = [line.strip() for line in open('input.txt') if line.strip()]
random.shuffle(docs)
print(f"num docs: {len(docs)}")

# ========== 2. 分词器构建 ==========
uchars = sorted(set(''.join(docs)))
BOS = len(uchars)
vocab_size = len(uchars) + 1
char_to_it = {ch: i for i, ch in enumerate(uchars)}
char_to_it['.'] = BOS
it_to_char = {i: ch for i, ch in enumerate(uchars)}
it_to_char[BOS] = '.'
print(f"vocab size: {vocab_size}")

# ========== 3. 核心微分引擎 (Micrograd) ==========
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

# ========== 4. 算子定义 ==========
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

# ========== 5. Hybrid 层 (TTT & Attention) ==========
def ttt_step(x, W_hidden, ttt_lr=0.2):
    y = linear(x, W_hidden)
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

# ========== 6. 模型参数初始化 ==========
n_embd = 12
block_size = 16  # 最大上下文长度
matrix = lambda nout, nin, std=0.1: [[Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]

state_dict = {
    'wte': matrix(vocab_size, n_embd),
    'lm_head': matrix(vocab_size, n_embd),
    'attn_q': matrix(n_embd, n_embd),
    'attn_k': matrix(n_embd, n_embd),
    'attn_v': matrix(n_embd, n_embd),
    'mlp': matrix(n_embd, n_embd)
}

# 展平所有参数
params = [p for mat in state_dict.values() for row in mat for p in row]
print(f"num params: {len(params)}")

# ========== 7. 混合前向传播逻辑 ==========
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

# ========== 8. 训练循环（Adam优化器）==========
learning_rate, beta1, beta2, eps_adam = 0.01, 0.85, 0.99, 1e-8
m = [0.0] * len(params)
v = [0.0] * len(params)
num_steps = 500

print("=== 开始 Hybrid 预训练 ===")
for step in range(num_steps):
    # 取一个文档，分词
    doc = docs[step % len(docs)]
    tokens = [BOS] + [char_to_it[ch] for ch in doc if ch in char_to_it] + [BOS]
    n = min(block_size, len(tokens) - 1)

    ttt_states = [[[Value(1.0 if i==j else 0.0) for j in range(n_embd)] for i in range(n_embd)]]
    kv_cache = {'k': [], 'v': []}

    losses = []
    for t in range(n):
        logits = forward(tokens[t], ttt_states, kv_cache)
        probs = softmax(logits)
        loss_t = -probs[tokens[t+1]].log()
        losses.append(loss_t)

    # 计算平均损失
    total = Value(0)
    for loss_t in losses:
        total = total + loss_t
    loss = total / n

    loss.backward()

    # Adam优化器更新
    lr_t = learning_rate * (1 - step / num_steps)
    for i, p in enumerate(params):
        m[i] = beta1 * m[i] + (1 - beta1) * p.grad
        v[i] = beta2 * v[i] + (1 - beta2) * p.grad ** 2
        m_hat = m[i] / (1 - beta1 ** (step + 1))
        v_hat = v[i] / (1 - beta2 ** (step + 1))
        p.data -= lr_t * m_hat / (v_hat ** 0.5 + eps_adam)
        p.grad = 0

    if (step + 1) % 100 == 0:
        print(f"Step {step+1:4d} / {num_steps} | Loss: {loss.data:.4f}")

# ========== 9. 推理生成（批量）==========
temperature = 0.5
print(f"\n=== 推理生成 (temperature={temperature}) ===")

for sample_idx in range(20):
    ttt_states = [[[Value(1.0 if i==j else 0.0) for j in range(n_embd)] for i in range(n_embd)]]
    kv_cache = {'k': [], 'v': []}

    token_id = BOS
    sample = []
    for _ in range(block_size):
        logits = forward(token_id, ttt_states, kv_cache)
        probs = softmax([l / temperature for l in logits])
        token_id = random.choices(range(vocab_size), weights=[p.data for p in probs])[0]
        if token_id == BOS:
            break
        sample.append(it_to_char.get(token_id, '?'))

    print(f"sample {sample_idx+1:2d}: {''.join(sample)}")
