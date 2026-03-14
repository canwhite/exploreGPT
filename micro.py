"""
用纯Python实现GPT训练和推理的最原子化方式
这是完整的算法实现
其他所有优化都是为了效率
@karpathy
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

    def __init__(self, data, children=(), local_grads=()):
        self.data = data                # 前向传播时计算的标量值
        self.grad = 0                   # 损失函数对该节点的梯度（导数），在反向传播中计算
        self._children = children       # 计算图中的子节点
        self._local_grads = local_grads # 该节点对子节点的局部导数

    def __add__(self, other):  # type: ignore[no-untyped-def]
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), (1, 1))

    def __mul__(self, other):  # type: ignore[no-untyped-def]
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), (other.data, self.data))

    def __pow__(self, other):  # type: ignore[no-untyped-def]
        return Value(self.data**other, (self,), (other * self.data**(other-1),))

    def log(self):  # type: ignore[no-untyped-def]
        return Value(math.log(self.data), (self,), (1/self.data,))

    def exp(self):  # type: ignore[no-untyped-def]
        return Value(math.exp(self.data), (self,), (math.exp(self.data),))

    def relu(self):  # type: ignore[no-untyped-def]
        return Value(max(0, self.data), (self,), (float(self.data > 0),))

    def __neg__(self):  # type: ignore[no-untyped-def]
        return self * -1

    def __radd__(self, other):  # type: ignore[no-untyped-def]
        return self + other

    def __sub__(self, other):  # type: ignore[no-untyped-def]
        return self + (-other)

    def __rsub__(self, other):  # type: ignore[no-untyped-def]
        return other + (-self)

    def __rmul__(self, other):  # type: ignore[no-untyped-def]
        return self * other

    def __truediv__(self, other):  # type: ignore[no-untyped-def]
        return self * other**-1

    def __rtruediv__(self, other):  # type: ignore[no-untyped-def]
        return other * self**-1

    def backward(self):
        """反向传播：计算所有参数的梯度"""
        topo = []  # 拓扑排序：确保在计算梯度时，先计算依赖的节点
        visited = set()
        #DFS添加所有节点
        def build_topo(v):
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

# 初始化模型参数，用于存储模型的知识
n_layer = 1     # Transformer神经网络深度（层数）
n_embd = 16     # 网络宽度（嵌入维度）
block_size = 16 # 注意力窗口的最大上下文长度（注：最长名字是15个字符）
n_head = 4      # 注意力头的数量
head_dim = n_embd // n_head # 每个注意力头的维度
matrix = lambda nout, nin, std=0.08: [[Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]
state_dict = {'wte': matrix(vocab_size, n_embd), 'wpe': matrix(block_size, n_embd), 'lm_head': matrix(vocab_size, n_embd)}
# wte: token嵌入矩阵, wpe: 位置嵌入矩阵, lm_head: 语言模型输出头
for i in range(n_layer):
    state_dict[f'layer{i}.attn_wq'] = matrix(n_embd, n_embd)  # Query权重矩阵
    state_dict[f'layer{i}.attn_wk'] = matrix(n_embd, n_embd)  # Key权重矩阵
    state_dict[f'layer{i}.attn_wv'] = matrix(n_embd, n_embd)  # Value权重矩阵
    state_dict[f'layer{i}.attn_wo'] = matrix(n_embd, n_embd)  # Attention输出权重矩阵
    # 这块儿是先放大，再缩小吗
    state_dict[f'layer{i}.mlp_fc1'] = matrix(4 * n_embd, n_embd)  # MLP第一层（扩展4倍）
    state_dict[f'layer{i}.mlp_fc2'] = matrix(n_embd, 4 * n_embd)  # MLP第二层（投影回原始维度）
params = [p for mat in state_dict.values() for row in mat for p in row] # 将所有参数展平成单个list[Value]
print(f"num params: {len(params)}")

# 定义模型架构：将tokens和参数映射到下一个token的logits
# 参考GPT-2架构，做了一些微调：layernorm改为rmsnorm，无bias，GeLU改为ReLU
def linear(x, w):
    """线性变换：y = xW^T"""
    return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]

def softmax(logits):
    """Softmax激活函数：将logits转换为概率分布"""
    max_val = max(val.data for val in logits)
    exps = [(val - max_val).exp() for val in logits]
    total = sum(exps)
    return [e / total for e in exps]

def rmsnorm(x):
    """RMS归一化：稳定训练，防止数值溢出"""
    ms = sum(xi * xi for xi in x) / len(x)
    scale = (ms + 1e-5) ** -0.5
    return [xi * scale for xi in x]

def gpt(token_id, pos_id, keys, values):
    """GPT模型前向传播：核心函数"""
    tok_emb = state_dict['wte'][token_id] # token嵌入：将token ID转换为向量
    pos_emb = state_dict['wpe'][pos_id] # 位置嵌入：编码位置信息
    x = [t + p for t, p in zip(tok_emb, pos_emb)] # 结合token和位置信息
    x = rmsnorm(x) # 预归一化（注：由于残差连接的反向传播，这里不冗余）

    for li in range(n_layer):
        # 1) 多头注意力机制块
        x_residual = x  # 保存残差连接
        x = rmsnorm(x)
        q = linear(x, state_dict[f'layer{li}.attn_wq'])  # Query：当前token想要查询什么
        k = linear(x, state_dict[f'layer{li}.attn_wk'])  # Key：其他token提供的键
        v = linear(x, state_dict[f'layer{li}.attn_wv'])  # Value：其他token的值
        keys[li].append(k)  # 缓存key，用于自注意力
        values[li].append(v)  # 缓存value
        x_attn = []
        for h in range(n_head):  # 遍历每个注意力头
            hs = h * head_dim
            q_h = q[hs:hs+head_dim]  # 当前头的query
            k_h = [ki[hs:hs+head_dim] for ki in keys[li]]  # 当前头的历史keys
            v_h = [vi[hs:hs+head_dim] for vi in values[li]]  # 当前头的历史values
            # 计算注意力分数：Q·K^T / sqrt(d_k)
            attn_logits = [sum(q_h[j] * k_h[t][j] for j in range(head_dim)) / head_dim**0.5 for t in range(len(k_h))]
            attn_weights = softmax(attn_logits)  # 转换为概率分布
            # 加权求和：根据注意力权重聚合values
            head_out = [sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h))) for j in range(head_dim)]
            x_attn.extend(head_out)

        x = linear(x_attn, state_dict[f'layer{li}.attn_wo'])  # 输出投影
        x = [a + b for a, b in zip(x, x_residual)]  # 残差连接
        # 2) 前馈神经网络块
        x_residual = x
        x = rmsnorm(x)
        x = linear(x, state_dict[f'layer{li}.mlp_fc1'])  # 扩展层：4倍维度
        x = [xi.relu() for xi in x]  # ReLU激活
        x = linear(x, state_dict[f'layer{li}.mlp_fc2'])  # 投影层：恢复原始维度
        x = [a + b for a, b in zip(x, x_residual)]  # 残差连接

    logits = linear(x, state_dict['lm_head'])
    return logits

# Adam优化器及其缓存
learning_rate, beta1, beta2, eps_adam = 0.01, 0.85, 0.99, 1e-8
m = [0.0] * len(params) # 一阶矩缓存（梯度均值）
v = [0.0] * len(params) # 二阶矩缓存（梯度平方的均值）

# 训练循环：重复执行以下步骤
num_steps = 1000 # 训练步数
for step in range(num_steps):

    # 取一个文档，分词，用BOS特殊标记包围
    doc = docs[step % len(docs)]
    tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]  # 例如：[BOS, a, n, a, BOS]
    n = min(block_size, len(tokens) - 1)

    # 前向传播：将token序列输入模型，构建计算图直到loss
    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    losses: list['Value'] = []  # 显式声明类型
    for pos_id in range(n):
        token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
        logits = gpt(token_id, pos_id, keys, values)  # GPT前向传播
        probs = softmax(logits)  # 转换为概率
        loss_t = -probs[target_id].log()  # 交叉熵损失：-log(预测正确token的概率)
        losses.append(loss_t)
    # 手动累加以保持Value类型
    total: 'Value' = Value(0)
    for loss_t in losses:
        total = total + loss_t
    loss: 'Value' = (Value(1) / Value(n)) * total # 文档序列的最终平均损失。愿你的loss很低。

    # 反向传播：计算所有模型参数的梯度
    loss.backward()

    # Adam优化器更新：根据梯度更新模型参数
    lr_t = learning_rate * (1 - step / num_steps) # 线性学习率衰减
    for i, p in enumerate(params):
        m[i] = beta1 * m[i] + (1 - beta1) * p.grad  # 更新一阶矩估计（动量）
        v[i] = beta2 * v[i] + (1 - beta2) * p.grad ** 2  # 更新二阶矩估计
        m_hat = m[i] / (1 - beta1 ** (step + 1))  # 偏差
        v_hat = v[i] / (1 - beta2 ** (step + 1))
        p.data -= lr_t * m_hat / (v_hat ** 0.5 + eps_adam)  # 参数更新
        p.grad = 0  # 清零梯度

    print(f"step {step+1:4d} / {num_steps:4d} | loss {loss.data:.4f}", end='\r')

# 推理：让模型生成新样本
temperature = 0.5 # 温度参数 (0, 1]，控制生成文本的"创造性"，低=保守，高=创新
print("\n--- inference (new, hallucinated names) ---")
for sample_idx in range(20):
    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    token_id = BOS  # 从BOS开始
    sample = []
    for pos_id in range(block_size):
        logits = gpt(token_id, pos_id, keys, values)  # 前向传播
        probs = softmax([l / temperature for l in logits])  # 温度缩放后应用softmax
        token_id = random.choices(range(vocab_size), weights=[p.data for p in probs])[0]  # 按概率采样
        if token_id == BOS:  # 遇到BOS则停止生成
            break
        sample.append(uchars[token_id])  # 将token ID转换回字符
    print(f"sample {sample_idx+1:2d}: {''.join(sample)}")
