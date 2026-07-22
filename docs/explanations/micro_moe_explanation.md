# micro_moe.py 实现总结

> `micro_moe.py` 是与 [`micro_gpt.py`](../../micro_gpt.py) 平行的一份教学脚本：在保留原版自动微分、注意力、训练和推理的基础上，**只把每一层的前馈 MLP 替换为 Top-1 Router + 多个独立 MLP Experts**。本文档讲清它的实现、参数、单 token 数据流和梯度边界。

## 1. 整体定位

- **教学目标**：在一份可独立运行的脚本里，把“router 决定谁、`Value` 计算图决定谁学、专家参数怎么被稀疏更新”三件事同时展示出来。
- **与原 dense GPT 的关系**：除前馈子块外，其余代码字符级与 `micro_gpt.py` 一致——`Value`、注意力、Adam、采样、训练循环全部复用，刻意把变量隔离在一处。
- **约束**：不引入第三方 MoE 库；不引入生产级特性（Top-k>1、aux loss、容量限制、专家并行）。所有 MoE 逻辑约 30 行。

## 2. 参数结构（`state_dict`）

```
wte / wpe / lm_head              # 与 micro_gpt.py 共享
layer{i}.attn_wq / wk / wv / wo  # 与 micro_gpt.py 共享
layer{i}.router                  # 新增：n_expert × n_embd
layer{i}.expert{e}.fc1           # 新增：4*n_embd × n_embd
layer{i}.expert{e}.fc2           # 新增：n_embd × 4*n_embd
```

| 参数 | 形状 | 数量（`n_expert=4`） | 说明 |
|------|------|---------------------|------|
| `router` | `n_expert × n_embd` | 4 × 16 = 64 | 用 `std=0.2` 初始化，避免初始塌缩 |
| `expert{e}.fc1` | `4n × n` | 4 × (64 × 16) = 4096 | 扩展 4 倍 |
| `expert{e}.fc2` | `n × 4n` | 4 × (16 × 64) = 4096 | 投影回 `n_embd` |
| 单层前馈总参数 | — | 64 + 4·4096·2 ≈ **16 448** | dense MLP 参数量约为 `4n²` · 2 = 2048 |

脚本顶部用 `expected_params` 断言 `len(params)` 匹配，**保证所有 router / expert 参数都进了 Adam 训练**。

## 3. 前向数据流（单 token，单层）

```
x ── rmsnorm ──►
                ├─► q,k,v ──► attention ──► + residual
                │
                └─ rmsnorm ──►
                               router = x @ W_router
                               probs  = softmax(router)
                               k      = argmax(probs)        # Top-1，控制流
                               h      = relu(x @ expert_k.fc1)
                               y      = h @ expert_k.fc2
                               out    = y * probs[k]         # Value 节点
                ◄────────────── + residual
```

要点：

- `top1(probs)` 是 **Python 控制流**，读 `probs[i].data` 选 argmax，**不返回 `Value`**，因此不进入计算图。
- 只为被选专家构造 `Value` 节点：未选专家的参数在本次 loss 中**不会被读取**，也谈不上反向传播。
- `probs[k]` 是 `Value`，它与 `y` 一起参与最终乘法，所以 router 路径在计算图上。

## 4. 反向传播与梯度边界

`Value.__mul__` 会把两个子节点都登记到计算图，因此一次 MoE 前向结束后的反向传播会沿两条路径传梯度：

1. **专家路径**：`out = y * p` → `∂L/∂y = p · ∂L/∂out`。
   - 只有被选专家 `k` 的 `fc1/fc2` 参数会获得梯度；
   - 其他 `e ≠ k` 的专家参数在该 token 上 `grad = 0`（**稀疏更新**）。
2. **Router 路径**：`p = probs[k]` 是 softmax 输出，反向会经 softmax 反传。
   - `∂L/∂W_router` 的第 `k` 行接收来自该 token 的非零梯度；
   - 其他行的梯度通过 softmax 的耦合项也可能非零，但只有“被选 token 经过的行”才真正决定路由。

直觉总结：

- 选中的专家必须学得准（梯度来自 `p · ∂L/∂out`）。
- Router 必须让 `p` 增大，以让被选专家的输出整体更被信任；其他专家因为没进入 `Value` 图，没有“被惩罚”的直接信号——这正是为什么生产 MoE 需要 auxiliary loss 抑制专家塌缩。

## 5. 训练 / 推理与原版的差异

| 项 | micro_gpt | micro_moe |
|----|-----------|-----------|
| 训练步数 | 1000 | 1000 |
| 学习率 / Adam β / eps | 完全一致 | 完全一致 |
| 推理采样 | temperature 0.5 × 20 | 同 |
| 额外可观测输出 | — | step 0 / 50 打印 router 概率分布，**人为检查塌缩** |

实际运行结果（人名数据集，相同 seed=42）：

- 末步 loss ≈ 2.59（与 dense 同一量级）；
- router 分布在 step 0 ≈ `[0.279, 0.235, 0.252, 0.234]`，step 50 ≈ `[0.289, 0.226, 0.262, 0.223]`，未见单专家塌缩；
- 推理样本：`daran / azixi / viera / anna / sena / kerie / sarian / haynan` 等，符合英文人名形态。

## 6. 与生产 MoE 的差异

| 特性 | micro_moe | Mixtral / DeepSeek-MoE 等 |
|------|-----------|---------------------------|
| 路由 | Top-1（硬） | Top-k（常 2） |
| 训练目标 | 纯交叉熵 | + 负载均衡辅助 loss（switch / aux-loss-free 等） |
| 专家容量 | 无限制 | capacity factor + token dropping |
| 表达 | `y * probs[k]` 缩放 | 多专家概率加权求和 |
| 并行 | 单进程循环 | 专家并行 / EP 通信 |
| 数值 | 纯 Python `Value` | GPU kernel + bf16/fp8 |

教学侧强调：先理解“路由 → 选专家 → 缩放 → 残差”的最小闭环，再去看 production 版本里那些为 GPU/通信/数值稳定而生的工程化结构。

## 7. 阅读路线建议

1. 先看 `moe()`（约 12 行）—— 整个文件的核心；
2. 再看 `state_dict` 的初始化，确认 `mlp_fc1/fc2` 已彻底消失、`router` 和 `expert{e}.fc1/fc2` 已就位；
3. 对比 `gpt()`：除“前馈子块”三行外，与 `micro_gpt.py` 字符级一致——这是变量隔离的关键；
4. 跑一次完整 1000 步，观察 router 概率分布是否随时间移动，验证路由在“动”而不是“死锁”到初始 argmax。

## 8. 已知风险与改进方向

- **专家塌缩**：本实现无 aux loss，长期训练可能让 router 集中到一个专家。改进：加 `n_expert * Σ probs_i` 形式的均衡 loss，或直接换 Top-2。
- **Top-1 抖动**：当 router 概率在两个专家之间 0.5/0.5 切换时，被选专家会跳变，loss 出现跳点。改进：换 Top-2 平滑，或加 router 温度。
- **未选专家零梯度**：稀疏训练天然成立，但“未选专家完全学不到”也可能让某些专家永远冷启动。改进：aux loss 或随机扰动路由。
- **效率**：本脚本每个 token 都遍历 `n_expert` 算一次 softmax，再 argmax，再做 1 次两层 MLP。生产环境会按 token 批量分桶到专家设备上并行计算。

---

**TL;DR**：`micro_moe.py` = `micro_gpt.py` + `top1` + `moe`，用最少的额外代码（≈30 行）暴露 MoE 的三个关键属性——**路由是控制流不是图、专家是稀疏可微、router 通过 softmax 反向获得梯度**。
