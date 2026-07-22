"""
用纯Python实现最简 MoE GPT：与 micro_gpt.py 共享自动微分与训练框架，
只把每一层的前馈 MLP 替换为 Top-1 Router + 多个独立 MLP Experts。
教学目标：暴露 router / expert 的参数边界、单 token 数据流和梯度流。
"""

import math  # math.log, math.exp
import os  # os.path.exists
import random  # random.seed, random.choices, random.gauss, random.shuffle

random.seed(42) # 设定随机种子，确保实验可复现

# 准备数据集 `docs`: 文档列表（例如：人名列表）
if not os.path.exists('input.txt'):
    import urllib.request
    names_url = 'https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt'
    urllib.request.urlretrieve(names_url, 'input.txt')
docs = [line.strip() for line in open('input.txt') if line.strip()]
random.shuffle(docs)  # 打乱数据顺序，避免训练时的顺序偏差
print(f"num docs: {len(docs)}")

# 构建分词器：将字符串转换为整数序列（"tokens"）并支持反向转换
uchars = sorted(set(''.join(docs))) # 提取数据集中所有唯一字符，分配token id 0..n-1
BOS = len(uchars) # 定义特殊的序列开始标记
vocab_size = len(uchars) + 1 # 词汇表总大小 = 字符数 + 1个BOS标记
print(f"vocab size: {vocab_size}")

# 自动微分系统：通过计算图递归地应用链式法则进行反向传播
class Value:
    __slots__ = ('data', 'grad', '_children', '_local_grads') # Python内存优化

    def __init__(
        self,
        data: float | int,
        children: tuple['Value', ...] = (),
        local_grads: tuple[float | int, ...] = (),
    ) -> None:
        self.data = float(data)         # 前向传播时计算的标量值
        self.grad = 0.0                 # 损失函数对该节点的梯度（导数），在反向传播中计算
        self._children = children       # 计算图中的子节点
        self._local_grads = local_grads # 该节点对子节点的局部导数

    def __add__(self, other: 'Value | float | int') -> 'Value':
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), (1, 1))

    def __mul__(self, other: 'Value | float | int') -> 'Value':
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), (other.data, self.data))

    def __pow__(self, other: float | int) -> 'Value':
        return Value(self.data**other, (self,), (other * self.data**(other-1),))

    def log(self) -> 'Value':
        return Value(math.log(self.data), (self,), (1/self.data,))

    def exp(self) -> 'Value':
        return Value(math.exp(self.data), (self,), (math.exp(self.data),))

    def relu(self) -> 'Value':
        return Value(max(0, self.data), (self,), (float(self.data > 0),))

    def __neg__(self) -> 'Value':
        return self * -1

    def __radd__(self, other: float | int) -> 'Value':
        return self + other

    def __sub__(self, other: 'Value | float | int') -> 'Value':
        return self + (-other)

    def __rsub__(self, other: float | int) -> 'Value':
        return other + (-self)

    def __rmul__(self, other: float | int) -> 'Value':
        return self * other

    def __truediv__(self, other: 'Value | float | int') -> 'Value':
        return self * other**-1

    def __rtruediv__(self, other: float | int) -> 'Value':
        return other * self**-1

    def backward(self) -> None:
        """反向传播：计算所有参数的梯度"""
        topo: list[Value] = []  # 拓扑排序：确保在计算梯度时，先计算依赖的节点
        visited: set[Value] = set()
        #DFS添加所有节点
        def build_topo(v: Value) -> None:
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        self.grad = 1  # 输出节点的梯度初始化为1（dL/dL = 1）
        # 按拓扑排序的逆序，从后向前传播梯度
        for v in reversed(topo):
            for child, local_grad in zip(v._children, v._local_grads):
                child.grad += local_grad * v.grad  # 链式法则：累积梯度

# 模型配置：和 micro_gpt.py 共享，只是把 MLP 替换为 MoE
n_layer = 1        # Transformer 神经网络深度（层数）
n_embd = 16        # 网络宽度（嵌入维度）
block_size = 16    # 注意力窗口的最大上下文长度
n_head = 4         # 注意力头的数量
head_dim = n_embd // n_head
n_expert = 4       # 每层专家数量（教学值，足够展示稀疏性且不会太慢）
router_std = 0.2   # router 权重用更大初始化 std，避免初始就塌缩到单专家
def matrix(nout: int, nin: int, std: float = 0.08) -> list[list[Value]]:
    return [[Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]

# state_dict：与 micro_gpt.py 唯一的差异是 dense MLP 被 router + 多个 expert 替代
state_dict: dict[str, list[list[Value]]] = {
    'wte': matrix(vocab_size, n_embd),
    'wpe': matrix(block_size, n_embd),
    'lm_head': matrix(vocab_size, n_embd),
}
# wte: token嵌入矩阵, wpe: 位置嵌入矩阵, lm_head: 语言模型输出头
for i in range(n_layer):
    state_dict[f'layer{i}.attn_wq'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wk'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wv'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wo'] = matrix(n_embd, n_embd)
    # MoE：router 决定 token 去哪个 expert；每个 expert 是独立的两层 ReLU MLP
    state_dict[f'layer{i}.router'] = matrix(n_expert, n_embd, std=router_std)
    # 可以理解为各自的mlp
    for e in range(n_expert):
        state_dict[f'layer{i}.expert{e}.fc1'] = matrix(4 * n_embd, n_embd)  # 扩展 4 倍
        state_dict[f'layer{i}.expert{e}.fc2'] = matrix(n_embd, 4 * n_embd)  # 投影回原始维度
params = [p for mat in state_dict.values() for row in mat for p in row]
print(f"num params: {len(params)}")
# 期望参数量：wte + wpe + lm_head + n_layer * (4*attn + router + n_expert*2*mlp)
expected_params = (vocab_size * n_embd) + (block_size * n_embd) + (vocab_size * n_embd)
expected_params += n_layer * (4 * n_embd * n_embd + n_expert * n_embd + n_expert * 2 * (4 * n_embd * n_embd))
assert len(params) == expected_params, f"param count mismatch: {len(params)} vs {expected_params}"

# 定义模型架构
def linear(x: list[Value], w: list[list[Value]]) -> list[Value]:
    """线性变换：y = xW^T"""
    return [sum((wi * xi for wi, xi in zip(wo, x)), Value(0)) for wo in w]

def softmax(logits: list[Value]) -> list[Value]:
    """Softmax激活函数：将logits转换为概率分布"""
    max_val = max(val.data for val in logits)
    exps = [(val - max_val).exp() for val in logits]
    total = sum(exps, Value(0))
    return [e / total for e in exps]

def rmsnorm(x: list[Value]) -> list[Value]:
    """RMS归一化：稳定训练，防止数值溢出"""
    ms = sum((xi * xi for xi in x), Value(0)) / len(x)
    scale = (ms + 1e-5) ** -0.5
    return [xi * scale for xi in x]

def top1(probs: list[Value]) -> int:
    """在 probs（长度 n_expert）上选 argmax，返回专家索引。仅作为 Python 控制流，不进入计算图。"""
    best_i, best_p = 0, probs[0].data
    for i in range(1, len(probs)):
        if probs[i].data > best_p:
            best_i, best_p = i, probs[i].data
    return best_i

def moe(x: list[Value], layer_index: int) -> list[Value]:
    """最简 MoE 前馈：
    1) router 给出 n_expert 个 softmax 概率；
    2) 用 argmax 选 Top-1 专家（仅控制流，不入图）；
    3) 只对被选专家做两层 ReLU MLP；
    4) 用被选专家的 router 概率作为标量缩放输出，让 router 进入 loss 计算图。
    """
    # router_logits = linear(x, state_dict[...]) 就是 “拿 token 表示去和每个专家的偏好向量做点积，
    # 得出每个专家的原始得分”
    # 后面 softmax 转概率、top1 选第一、概率缩放专家输出，全依赖这一步得到的 router_logits。
    router_logits = linear(x, state_dict[f'layer{layer_index}.router'])
    probs = softmax(router_logits)
    #扫一遍概率，挑数值最大的位置，返回那个位置（专家索引）。
    expert_idx = top1(probs)
    selected_prob = probs[expert_idx]  # Value 节点，使 router 在反向传播中获梯度
    # 只为被选专家构造 Value 节点：未选专家不进入本次 loss 的计算图
    h = linear(x, state_dict[f'layer{layer_index}.expert{expert_idx}.fc1'])
    h = [xi.relu() for xi in h]
    out = linear(h, state_dict[f'layer{layer_index}.expert{expert_idx}.fc2'])
    # 把 router 概率乘到输出每个维度上：dL/drouter = Σ_j dL/dout_j * expert_out_j
    return [y * selected_prob for y in out]

def gpt(token_id: int, pos_id: int, keys: list[list[list[Value]]], values: list[list[list[Value]]]) -> list[Value]:
    """GPT前向传播：除 MLP 子块外，与 micro_gpt.py 字符级一致。"""
    tok_emb = state_dict['wte'][token_id]
    pos_emb = state_dict['wpe'][pos_id]
    x: list[Value] = [t + p for t, p in zip(tok_emb, pos_emb)]
    x = rmsnorm(x)

    for li in range(n_layer):
        # 1) 多头注意力
        x_residual = x
        x = rmsnorm(x)
        q = linear(x, state_dict[f'layer{li}.attn_wq'])
        k = linear(x, state_dict[f'layer{li}.attn_wk'])
        v = linear(x, state_dict[f'layer{li}.attn_wv'])
        keys[li].append(k)
        values[li].append(v)
        x_attn: list[Value] = []
        for h_idx in range(n_head):
            hs = h_idx * head_dim
            q_h = q[hs:hs+head_dim]
            k_h = [ki[hs:hs+head_dim] for ki in keys[li]]
            v_h = [vi[hs:hs+head_dim] for vi in values[li]]
            attn_logits = [sum((q_h[j] * k_h[t][j] for j in range(head_dim)), Value(0)) / head_dim**0.5 for t in range(len(k_h))]
            attn_weights = softmax(attn_logits)
            head_out = [sum((attn_weights[t] * v_h[t][j] for t in range(len(v_h))), Value(0)) for j in range(head_dim)]
            x_attn.extend(head_out)
        x = linear(x_attn, state_dict[f'layer{li}.attn_wo'])
        x = [a + b for a, b in zip(x, x_residual)]
        # 2) MoE 前馈（替换原 dense MLP）
        x_residual = x
        x = rmsnorm(x)
        # micro_moe.py 几乎是把 micro_gpt.py 复制一份，只把“dense MLP 三行”换成 moe() 调用，
        # 新增 router 和 expert{e} 两族参数；
        # 其余基础设施（Value、注意力、训练、推理）逐字复用，从而把变量隔离在“前馈子块”这一处。
        x = moe(x, li)
        x = [a + b for a, b in zip(x, x_residual)]

    logits = linear(x, state_dict['lm_head'])
    return logits

# Adam优化器及其缓存
learning_rate, beta1, beta2, eps_adam = 0.01, 0.85, 0.99, 1e-8
m = [0.0] * len(params)
v = [0.0] * len(params)

# 训练循环
num_steps = 1000
for step in range(num_steps):
    doc = docs[step % len(docs)]
    tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
    n = min(block_size, len(tokens) - 1)

    keys: list[list[list[Value]]] = [[] for _ in range(n_layer)]
    values: list[list[list[Value]]] = [[] for _ in range(n_layer)]
    losses: list[Value] = []
    for pos_id in range(n):
        token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
        logits = gpt(token_id, pos_id, keys, values)
        probs = softmax(logits)
        loss_t = -probs[target_id].log()
        losses.append(loss_t)
    total: Value = Value(0)
    for loss_t in losses:
        total = total + loss_t
    loss: Value = (Value(1) / Value(n)) * total

    loss.backward()
    # 训练早期输出一次 router 概率分布，肉眼确认路由不是单点
    if step == 0 or step == 50:
        router_w = state_dict['layer0.router']
        sample_probs = softmax(linear(state_dict['wte'][tokens[0]], router_w))
        print(f"\n[DEBUG-MOE] step {step} router probs @ token {tokens[0]}: {[round(p.data, 3) for p in sample_probs]}")

    lr_t = learning_rate * (1 - step / num_steps)
    for i, p in enumerate(params):
        m[i] = beta1 * m[i] + (1 - beta1) * p.grad
        v[i] = beta2 * v[i] + (1 - beta2) * p.grad ** 2
        m_hat = m[i] / (1 - beta1 ** (step + 1))
        v_hat = v[i] / (1 - beta2 ** (step + 1))
        p.data -= lr_t * m_hat / (v_hat ** 0.5 + eps_adam)
        p.grad = 0

    print(f"step {step+1:4d} / {num_steps:4d} | loss {loss.data:.4f}", end='\r')

# 推理
temperature = 0.5
print("\n--- inference (new, hallucinated names) ---")
for sample_idx in range(20):
    keys = [[] for _ in range(n_layer)]
    values = [[] for _ in range(n_layer)]
    token_id = BOS
    sample = []
    for pos_id in range(block_size):
        logits = gpt(token_id, pos_id, keys, values)
        probs = softmax([l / temperature for l in logits])
        token_id = random.choices(range(vocab_size), weights=[p.data for p in probs])[0]
        if token_id == BOS:
            break
        sample.append(uchars[token_id])
    print(f"sample {sample_idx+1:2d}: {''.join(sample)}")
