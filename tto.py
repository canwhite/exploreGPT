import math
import os
import random

# 设定随机种子，确保实验可重复
random.seed(42)

# ========== 1. 数据准备 (Karpathy makemore 风格) ==========
if not os.path.exists('input.txt'):
    import urllib.request
    names_url = 'https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt'
    urllib.request.urlretrieve(names_url, 'input.txt')

docs = [line.strip() for line in open('input.txt') if line.strip()]
random.shuffle(docs)

uchars = sorted(set(''.join(docs)))
BOS = len(uchars)  # 序列起始/结束符
vocab_size = len(uchars) + 1
char_to_it = {ch: i for i, ch in enumerate(uchars)}
it_to_char = {i: ch for i, ch in enumerate(uchars)}
it_to_char[BOS] = '.'

# ========== 2. 自动微分系统 (Micrograd) ==========
class Value:
    """ 最原子化的自动微分引擎 """
    __slots__ = ('data', 'grad', '_children', '_local_grads')

    def __init__(self, data, children=(), local_grads=()):
        self.data = data
        self.grad = 0
        self._children = children
        self._local_grads = local_grads

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), (1, 1))

    def __radd__(self, other): # 支持 sum([Value, Value], 0) 这种操作
        return self + other

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), (other.data, self.data))

    def __rmul__(self, other):
        return self * other

    def __pow__(self, other):
        return Value(self.data**other, (self,), (other * self.data**(other-1),))

    def log(self):
        return Value(math.log(self.data + 1e-10), (self,), (1/(self.data + 1e-10),))

    def exp(self):
        return Value(math.exp(self.data), (self,), (math.exp(self.data),))

    def relu(self):
        return Value(max(0, self.data), (self,), (float(self.data > 0),))

    def __neg__(self): return self * -1
    def __sub__(self, other): return self + (-other)
    def __rsub__(self, other): return other + (-self)
    def __truediv__(self, other): return self * other**-1

    def backward(self):
        # 拓扑排序确保梯度传播顺序
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

# ========== 3. 模型配置与参数初始化 ==========
n_layer = 1
n_embd = 16
block_size = 16 # 名字通常比较短
n_head = 4
head_dim = n_embd // n_head

def init_matrix(nout, nin, std=0.02):
    return [[Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]

# 定义静态 Backbone 参数
state_dict = {
    'wte': init_matrix(vocab_size, n_embd),
    'wpe': init_matrix(block_size, n_embd),
    'lm_head': init_matrix(vocab_size, n_embd)
}

for i in range(n_layer):
    state_dict[f'layer{i}.attn_wq'] = init_matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wk'] = init_matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wv'] = init_matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wo'] = init_matrix(n_embd, n_embd)
    state_dict[f'layer{i}.mlp_fc1'] = init_matrix(4 * n_embd, n_embd)
    state_dict[f'layer{i}.mlp_fc2'] = init_matrix(n_embd, 4 * n_embd)

# 展平所有参数以便优化
params = [p for mat in state_dict.values() for row in mat for p in row]

# TTO 目标参数：只优化输出头 (lm_head)
tto_params = [p for row in state_dict['lm_head'] for p in row]

# ========== 4. 核心算子 (算子化实现) ==========

def linear(x, w):
    return [sum((wi * xi for wi, xi in zip(wo, x)), Value(0)) for wo in w]

def softmax(logits):
    mv = max(val.data for val in logits)
    exps = [(val - mv).exp() for val in logits]
    total = sum(exps, Value(0))
    return [e / total for e in exps]

def rmsnorm(x):
    ms = sum((xi * xi for xi in x), Value(0)) / len(x)
    scale = (ms + 1e-5) ** -0.5
    return [xi * scale for xi in x]

def gpt_forward(token_id, pos_id, keys, values):
    """ GPT 单步前向传播（带 KV Cache 接口） """
    x = [t + p for t, p in zip(state_dict['wte'][token_id], state_dict['wpe'][pos_id])]

    for li in range(n_layer):
        # Attention 模块
        x_res = x
        x = rmsnorm(x)
        q = linear(x, state_dict[f'layer{li}.attn_wq'])
        k = linear(x, state_dict[f'layer{li}.attn_wk'])
        v = linear(x, state_dict[f'layer{li}.attn_wv'])
        keys[li].append(k); values[li].append(v)

        # 多头注意力拼接
        x_attn = []
        for h in range(n_head):
            hs = h * head_dim
            q_h = q[hs:hs+head_dim]
            k_h = [ki[hs:hs+head_dim] for ki in keys[li]]
            v_h = [vi[hs:hs+head_dim] for vi in values[li]]

            # Dot-product attention
            score = [sum((qj * kj for qj, kj in zip(q_h, kh)), Value(0)) / (head_dim**0.5) for kh in k_h]
            weights = softmax(score)
            context = [sum((weights[t] * v_h[t][j] for t in range(len(v_h))), Value(0)) for j in range(head_dim)]
            x_attn.extend(context)

        x = [a + b for a, b in zip(linear(x_attn, state_dict[f'layer{li}.attn_wo']), x_res)]

        # MLP 模块
        x_res = x
        x = rmsnorm(x)
        x = [xi.relu() for xi in linear(x, state_dict[f'layer{li}.mlp_fc1'])]
        x = [a + b for a, b in zip(linear(x, state_dict[f'layer{li}.mlp_fc2']), x_res)]

    return linear(x, state_dict['lm_head'])

# ========== 5. 训练与 TTO 核心逻辑 ==========

# Adam 优化器状态
m = [0.0] * len(params); v = [0.0] * len(params)

print("=== Phase 1: Pre-training (Backbone) ===")
for step in range(201):
    doc = docs[step % len(docs)]
    tokens = [BOS] + [char_to_it[c] for c in doc] + [BOS]
    n = min(block_size, len(tokens) - 1)

    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    total_loss = Value(0)
    for t in range(n):
        logits = gpt_forward(tokens[t], t, keys, values)
        probs = softmax(logits)
        total_loss = total_loss + (-probs[tokens[t+1]].log())

    loss = total_loss / n
    loss.backward()

    # 全量参数 Adam 更新
    for i, p in enumerate(params):
        m[i] = 0.9 * m[i] + 0.1 * p.grad
        v[i] = 0.999 * v[i] + 0.001 * (p.grad**2)
        p.data -= 0.01 * m[i] / (math.sqrt(v[i]) + 1e-8)
        p.grad = 0
    if step % 50 == 0: print(f"Step {step} | Loss: {loss.data:.4f}")

# ========== 6. Test Time Optimization (TTO) 推理 ==========

def tto_generate(prefix_str, tto_steps=1):
    # 1. 备份 lm_head 初始参数 (快照)
    snapshot = [p.data for p in tto_params]
    tto_m = [0.0] * len(tto_params) # TTO 局部动量

    tokens = [BOS] + [char_to_it[c] for c in prefix_str]

    # 2. TTO 适应阶段：根据前缀微调输出头
    if len(tokens) > 1:
        for _ in range(tto_steps):
            keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
            total_loss = Value(0)
            for t in range(len(tokens)-1):
                logits = gpt_forward(tokens[t], t, keys, values)
                probs = softmax(logits)
                total_loss = total_loss + (-probs[tokens[t+1]].log())

            (total_loss / (len(tokens)-1)).backward()

            # 只更新 tto_params (lm_head)
            for i, p in enumerate(tto_params):
                tto_m[i] = 0.9 * tto_m[i] + p.grad
                p.data -= 0.05 * tto_m[i] # 较大的 TTO 学习率

            # 清理所有梯度
            for p in params: p.grad = 0

    # 3. 生成阶段
    res = prefix_str
    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    # 预填充 KV Cache
    for t in range(len(tokens)-1): gpt_forward(tokens[t], t, keys, values)

    curr = tokens[-1]
    for i in range(block_size - len(tokens)):
        logits = gpt_forward(curr, len(tokens)-1 + i, keys, values)
        probs = softmax([l / 0.7 for l in logits]) # 带温度采样
        next_id = random.choices(range(vocab_size), weights=[p.data for p in probs])[0]
        if next_id == BOS: break
        res += it_to_char[next_id]
        curr = next_id

    # 4. 恢复快照：保证 TTO 只针对当前推理，不污染全局模型
    for p, val in zip(tto_params, snapshot): p.data = val
    return res

print("\n=== Phase 2: TTO Inference (Adapting only LM Head) ===")
for p in ["ma", "lu", "ka"]:
    print(f"Prefix '{p}' -> TTO Output: {tto_generate(p)}")
