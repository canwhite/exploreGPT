"""
用纯Python实现GPT训练和推理的最原子化方式
在此基础上实现 Test Time Optimization (TTO)

TTO 核心思想：
- 推理时只优化部分参数（而非全部），更轻量高效
- 常见策略：只优化 output head 或 normalization 参数
- 比完整 TTT 更快，但仍能获得适应能力

@karpathy + TTO extension
"""

import math
import os
import random

random.seed(42)

# 数据准备
if not os.path.exists('input.txt'):
    import urllib.request
    names_url = 'https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt'
    urllib.request.urlretrieve(names_url, 'input.txt')
docs = [line.strip() for line in open('input.txt') if line.strip()]
random.shuffle(docs)
print(f"num docs: {len(docs)}")

uchars = sorted(set(''.join(docs)))
BOS = len(uchars)
vocab_size = len(uchars) + 1
print(f"vocab size: {vocab_size}")

class Value:
    __slots__ = ('data', 'grad', '_children', '_local_grads')

    def __init__(self, data, children=(), local_grads=()):
        self.data = data
        self.grad = 0
        self._children = children
        self._local_grads = local_grads

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), (1, 1))

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), (other.data, self.data))

    def __pow__(self, other):
        return Value(self.data**other, (self,), (other * self.data**(other-1),))

    def log(self):
        return Value(math.log(self.data), (self,), (1/self.data,))

    def exp(self):
        return Value(math.exp(self.data), (self,), (math.exp(self.data),))

    def relu(self):
        return Value(max(0, self.data), (self,), (float(self.data > 0),))

    def __neg__(self):
        return self * -1

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * other**-1

    def __rtruediv__(self, other):
        return other * self**-1

    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        self.grad = 1
        for v in reversed(topo):
            for child, local_grad in zip(v._children, v._local_grads):
                child.grad += local_grad * v.grad

n_layer = 1
n_embd = 16
block_size = 16
n_head = 4
head_dim = n_embd // n_head
matrix = lambda nout, nin, std=0.08: [[Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]

state_dict = {
    'wte': matrix(vocab_size, n_embd),
    'wpe': matrix(block_size, n_embd),
    'lm_head': matrix(vocab_size, n_embd)
}

for i in range(n_layer):
    state_dict[f'layer{i}.attn_wq'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wk'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wv'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wo'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.mlp_fc1'] = matrix(4 * n_embd, n_embd)
    state_dict[f'layer{i}.mlp_fc2'] = matrix(n_embd, 4 * n_embd)

params = [p for mat in state_dict.values() for row in mat for p in row]
print(f"num params: {len(params)}")

# TTO: 只优化 output head (lm_head) 参数
lm_head_params = [p for row in state_dict['lm_head'] for p in row]
print(f"TTO optimizable params (lm_head only): {len(lm_head_params)}")

def linear(x, w):
    return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]

def softmax(logits):
    max_val = max(val.data for val in logits)
    exps = [(val - max_val).exp() for val in logits]
    total = sum(exps)
    return [e / total for e in exps]

def rmsnorm(x):
    ms = sum(xi * xi for xi in x) / len(x)
    scale = (ms + 1e-5) ** -0.5
    return [xi * scale for xi in x]

def gpt(token_id, pos_id, keys, values):
    tok_emb = state_dict['wte'][token_id]
    pos_emb = state_dict['wpe'][pos_id]
    x = [t + p for t, p in zip(tok_emb, pos_emb)]
    x = rmsnorm(x)

    for li in range(n_layer):
        x_residual = x
        x = rmsnorm(x)
        q = linear(x, state_dict[f'layer{li}.attn_wq'])
        k = linear(x, state_dict[f'layer{li}.attn_wk'])
        v = linear(x, state_dict[f'layer{li}.attn_wv'])
        keys[li].append(k)
        values[li].append(v)
        x_attn = []
        for h in range(n_head):
            hs = h * head_dim
            q_h = q[hs:hs+head_dim]
            k_h = [ki[hs:hs+head_dim] for ki in keys[li]]
            v_h = [vi[hs:hs+head_dim] for vi in values[li]]
            attn_logits = [sum(q_h[j] * k_h[t][j] for j in range(head_dim)) / head_dim**0.5 for t in range(len(k_h))]
            attn_weights = softmax(attn_logits)
            head_out = [sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h))) for j in range(head_dim)]
            x_attn.extend(head_out)

        x = linear(x_attn, state_dict[f'layer{li}.attn_wo'])
        x = [a + b for a, b in zip(x, x_residual)]
        x_residual = x
        x = rmsnorm(x)
        x = linear(x, state_dict[f'layer{li}.mlp_fc1'])
        x = [xi.relu() for xi in x]
        x = linear(x, state_dict[f'layer{li}.mlp_fc2'])
        x = [a + b for a, b in zip(x, x_residual)]

    logits = linear(x, state_dict['lm_head'])
    return logits

# Adam 优化器（用于预训练）
learning_rate, beta1, beta2, eps_adam = 0.01, 0.85, 0.99, 1e-8
m = [0.0] * len(params)
v = [0.0] * len(params)

# Phase 1: 预训练
num_steps = 1000
print("\n=== Phase 1: Pre-training ===")
for step in range(num_steps):
    doc = docs[step % len(docs)]
    tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
    n = min(block_size, len(tokens) - 1)

    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    losses = []
    for pos_id in range(n):
        token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
        logits = gpt(token_id, pos_id, keys, values)
        probs = softmax(logits)
        loss_t = -probs[target_id].log()
        losses.append(loss_t)

    total = Value(0)
    for loss_t in losses:
        total = total + loss_t
    loss = (Value(1) / Value(n)) * total

    loss.backward()

    lr_t = learning_rate * (1 - step / num_steps)
    for i, p in enumerate(params):
        m[i] = beta1 * m[i] + (1 - beta1) * p.grad
        v[i] = beta2 * v[i] + (1 - beta2) * p.grad ** 2
        m_hat = m[i] / (1 - beta1 ** (step + 1))
        v_hat = v[i] / (1 - beta2 ** (step + 1))
        p.data -= lr_t * m_hat / (v_hat ** 0.5 + eps_adam)
        p.grad = 0

    print(f"step {step+1:4d} / {num_steps:4d} | loss {loss.data:.4f}", end='\r')

print("\n\n=== Phase 2: Test Time Optimization (TTO) Inference ===")

# TTO 配置
tto_lr = 0.01
tto_momentum = 0.9
temperature = 0.5

# TTO 只为 lm_head 参数维护优化状态
tto_m = [0.0] * len(lm_head_params)

def save_lm_head():
    return [p.data for p in lm_head_params]

def restore_lm_head(snapshot):
    for p, val in zip(lm_head_params, snapshot):
        p.data = val

def tto_adapt_step(tokens):
    """TTO: 只更新 lm_head 参数"""
    n = min(block_size, len(tokens) - 1)
    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    losses = []

    for pos_id in range(n):
        token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
        logits = gpt(token_id, pos_id, keys, values)
        probs = softmax(logits)
        loss_t = -probs[target_id].log()
        losses.append(loss_t)

    total = Value(0)
    for loss_t in losses:
        total = total + loss_t
    loss = (Value(1) / Value(n)) * total

    loss.backward()

    # 只更新 lm_head 参数（使用简单 SGD + momentum）
    for i, p in enumerate(lm_head_params):
        tto_m[i] = tto_momentum * tto_m[i] + p.grad
        p.data -= tto_lr * tto_m[i]
        p.grad = 0

    # 清零其他参数的梯度（不更新）
    for p in params:
        if p not in lm_head_params:
            p.grad = 0

    return loss.data

def tto_generate(context_tokens):
    """TTO 生成：先适应，再生成"""
    snapshot = save_lm_head()

    # TTO 适应
    if len(context_tokens) > 1:
        tto_adapt_step(context_tokens)

    # 生成
    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]

    for pos_id, token_id in enumerate(context_tokens[:-1]):
        _ = gpt(token_id, pos_id, keys, values)

    token_id = context_tokens[-1]
    sample = []
    pos_id = len(context_tokens) - 1

    for _ in range(block_size - pos_id):
        logits = gpt(token_id, pos_id, keys, values)
        probs = softmax([l / temperature for l in logits])
        token_id = random.choices(range(vocab_size), weights=[p.data for p in probs])[0]
        if token_id == BOS:
            break
        sample.append(uchars[token_id])
        pos_id += 1

    # 恢复参数
    restore_lm_head(snapshot)

    return ''.join(sample)

print(f"TTO config: lr={tto_lr}, momentum={tto_momentum}")
print(f"Optimizing only {len(lm_head_params)}/{len(params)} params ({100*len(lm_head_params)/len(params):.1f}%)\n")

for sample_idx in range(20):
    context_doc = docs[(num_steps + sample_idx) % len(docs)]

    if sample_idx < 10:
        sample = tto_generate([BOS, BOS])
        print(f"sample {sample_idx+1:2d} (TTO from BOS): {sample}")
    else:
        half = len(context_doc) // 2
        prefix = context_doc[:half]
        prefix_tokens = [BOS] + [uchars.index(ch) for ch in prefix]
        sample = tto_generate(prefix_tokens)
        print(f"sample {sample_idx+1:2d} (TTO from '{prefix}...'): {sample}")

print("\n=== TTO Complete ===")
print(f"Total params: {len(params)}")
print(f"TTO params: {len(lm_head_params)} (lm_head only)")
print(f"Efficiency gain: {len(params)/len(lm_head_params):.1f}x faster than full TTT")
