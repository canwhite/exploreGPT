# Plan: Minimal MoE GPT

> 在保留 `micro_gpt.py` 教学风格和纯 Python 自动微分实现的基础上，新增一个可独立训练、推理并易于对照理解的最简稀疏 MoE GPT。

## Context

当前 `micro_gpt.py` 使用标准 Transformer 前馈网络：`n_embd → 4*n_embd → n_embd`。本次将该前馈块替换为最小可理解的 Mixture-of-Experts：一个路由器为每个 token 选择专家，多个结构相同但参数独立的 MLP 专家处理 token，再由路由权重合并结果。

约束：

- 不引入第三方 MoE 库或新依赖。
- 保留现有标量级 `Value` 自动微分，使路由器和专家均可端到端训练。
- 优先教学清晰度，而非批处理效率、分布式训练或生产级负载均衡。
- 不破坏原始 `micro_gpt.py`，以便逐行对照普通 MLP 与 MoE。

## Goal

新增 `micro_moe.py`，满足：

1. 复用并保留原实现的数据、自动微分、注意力、训练和推理流程。
2. 将每层标准 MLP 替换为包含 router、多个专家和 Top-1 稀疏选择的 MoE 前馈块。
3. router 概率参与最终输出计算，确保被选专家和对应路由分数能获得梯度。
4. 通过严格类型检查、字节码编译和完整训练/推理运行。
5. 最终解释参数结构、单 token 数据流、梯度流，以及这个最简版本与生产 MoE 的差异。

## Plan

1. 以 `micro_gpt.py` 为基线新增 `micro_moe.py`，保留原文件作为 dense GPT 对照组。
2. 增加最小 MoE 配置：专家数量 `n_expert`，每个 token 激活专家数固定为 Top-1。
3. 为每层新增 router 权重矩阵（命名 `layer{i}.router`，形状 `n_embd × n_expert`），以及每个专家各自的 `fc1`（`4*n_embd × n_embd`）、`fc2`（`n_embd × 4*n_embd`）参数矩阵；删除该层原来的共享 dense MLP 参数。
4. 实现带完整类型标注的 `moe(x, layer_index) -> list[Value]`：
   - 用 `linear(x, state_dict[f'layer{i}.router'])` 计算 router logits（长度 `n_expert`），再 softmax 得到概率 `probs`；
   - 显式 `def top1(probs: list[Value]) -> int: ...` 选择 Top-1 专家索引；
   - 用被选专家的 `fc1`/`fc2` 构造 MLP，**只**为选中专家创建 `Value` 节点；
   - 用 `selected_prob = probs[expert_idx]` 缩放专家输出（每个维度都乘这个 `Value` 标量），保证 router 路径在 loss 计算图上；
   - 返回与输入同维度的向量供残差连接使用。
5. 在 `gpt()` 中除 MLP 子块外字符级复制 `micro_gpt.py`，仅把原 dense MLP 替换为 `moe()`；其他部分（注意力、RMSNorm、残差、采样、训练循环）保持一致。
6. router 初始化用略大 `std`（如 0.2），降低初始化即塌缩概率；并在脚本顶部打印初始 router 概率分布便于观察。
7. 参数展平沿用 `micro_gpt.py` 的 `params = [p for mat in state_dict.values() for row in mat for p in row]`；插入一段 `print` 断言 `len(params)` 等于按当前形状算出的期望值，验证 router 与全部 expert 都被纳入训练。
8. 执行验证（mypy strict → py_compile → 训练+推理运行），并按"未通过"的具体错误**只**修正类型或 MoE 逻辑，不做无关重构。
9. 向用户解释：dense MLP 与 MoE 的参数结构对照；一次 token 前向传播全过程；Top-1 + soft 缩放下的梯度边界（含 router 获得梯度、未选专家不获梯度）；与生产 MoE 的差异及已知风险（专家塌缩、Top-1 抖动等）。

## Think — Debug Methodology

- 此脚本没有框架路由边界；关键边界是 `gpt() → moe() → selected expert`。若运行异常，在 `moe()` 入口记录输入长度、router 概率、选中专家和输出长度。
- 首先检查完整 traceback 和第一个异常，不通过反复调整结构猜测原因。
- 自动微分行为以本项目 `Value` 运算实现为准；确认 router 权重是否进入 loss 计算图时，直接检查一次反向传播后的 router gradient。
- 使用统一 `[DEBUG-MOE]` 前缀；定位完成后全局搜索并删除所有临时调试输出。
- 从上游向下游验证：输入维度 → router logits → softmax → expert index → expert output → residual output。

## Do — Verification Strategy

- **Build**: `uv run python -m py_compile micro_moe.py` — 必须通过。项目当前未在 `pyproject.toml` 中定义 `uv run build` 命令，因此以 Python 编译作为可用构建门槛。
- **Static analysis / type check**: `python -m mypy micro_moe.py --strict` — 必须零错误。
- **Runtime verification**: `uv run python micro_moe.py` — 完成全部训练步骤并生成样本，无异常或非有限 loss。
- **Logic correctness**:
  - 普通路径：router 产生 `n_expert` 个概率并选择合法专家索引；输出长度始终为 `n_embd`。
  - 梯度路径：选中专家参数和被选 router 概率对应参数获得梯度；未选专家在该 token 上不执行。
  - 残差路径：MoE 输出可与输入逐元素相加，类型均为 `Value`。
  - 多层/多专家参数路径：所有 router 和专家矩阵进入 `state_dict`、`params` 和 Adam 更新。
  - 数值路径：训练 loss 为有限数值，推理采样索引始终落在词表范围。

## Adjust — Rollback and Global Scan

- **Rollback plan**: 删除新增的 `micro_moe.py` 即可完整回滚；原始 `micro_gpt.py` 不变。
- **Global scan**: 搜索所有原 dense MLP 参数名和调用，确认 MoE 文件中没有同时保留未使用的共享 MLP；检查所有 `state_dict` 参数都被展平并训练。
- **Backwards compatibility**: N/A — 新增独立教学脚本，不改变原脚本接口或行为。

## Open Questions

- 默认采用 Top-1 路由和少量专家，以保持实现最小；专家数量在实现时选择一个能清楚展示稀疏激活且运行成本较低的值。
- 最简版本是否需要辅助负载均衡损失：默认不加入，因为它会显著增加解释和实现复杂度；会明确说明专家塌缩风险。

## Out of Scope

- Top-k（k > 1）加权聚合。
- 专家容量限制、token dropping、负载均衡辅助损失。
- noisy gating、共享专家、专家并行或分布式通信。
- 批处理、GPU kernel 优化和生产级吞吐性能。
- 修改原始 `micro_gpt.py`。

---

> Next step: run `/pre-mortem docs/plan-minimal-moe.md` to identify failure modes and harden this plan.

---

## Pre-Mortem Risks

### [Risk] router 概率不参与损失计算，router 权重无梯度
**Severity**: 5 | **Likelihood**: 4 | **Detectability**: 4
**Risk Score**: 16.0

**Failure Scenario**：`argmax`/`max` 是非可微操作；若实现只取 `argmax` 选专家并用 1 缩放专家输出，router 权重全程不进入 loss 计算图，router 永远不被训练，模型退化为固定硬路由。

**Mitigation**:
- 在 `moe()` 中用 softmax 概率数值选出 Top-1 专家，并将该专家的 softmax 概率作为标量乘到专家输出上。
- 验证：反向传播后检查 `state_dict['layer0.router']` 的部分行 grad 非零，且在多次迭代中数值发生变化。
- 实现中显式写出 `selected_prob = probs[expert_idx]`，避免在乘到输出时被误写成常量 1。

### [Risk] 未选专家因共享计算图而获得意外梯度
**Severity**: 4 | **Likelihood**: 3 | **Detectability**: 3
**Risk Score**: 12.0

**Failure Scenario**：如果 router 写成 `expert_outputs[expert_idx] * probs[expert_idx]`，但仍对全部专家执行前向（哪怕最后不读），`Value` 会因为构造的中间节点把未选专家的路径也带到计算图里，导致“稀疏”失效并让全部专家同时被训练。

**Mitigation**:
- 在 `moe()` 中仅构造被选专家的 `Value` 节点，未选专家不进入 `Value` 计算图。
- 验证：在一次反向传播后，断言未选专家参数在当前 token 上的 grad 为 0。
- 调整 `state_dict` 中对应层的参数顺序，使 router 先于 experts，方便逐层核对梯度流。

### [Risk] 专家塌缩导致训练崩溃或无效
**Severity**: 4 | **Likelihood**: 4 | **Detectability**: 2
**Risk Score**: 12.0

**Failure Scenario**：Top-1 + 无负载均衡损失时，softmax 容易把所有 token 路由到同一专家，其他专家权重完全不更新；训练 loss 可能停在高位或出现 NaN。

**Mitigation**:
- 初始化时给 router 权重加小幅噪声（增大 `std`），避免一开始就饱和。
- 训练前先打印初始 router 概率分布，并在前几步中肉眼确认路由分布不是单点。
- 在最终解释中明确说明这一风险，并指出改进方向（aux loss / noisy gating / top-k）。
- 如发现某专家概率在 50 步后仍 < 0.01，停止当前实现并在该文件中标注警告，不静默运行。

### [Risk] 参数展平和 Adam 更新遗漏 router 或专家
**Severity**: 5 | **Likelihood**: 2 | **Detectability**: 3
**Risk Score**: 10.0

**Failure Scenario**：若 `state_dict` 中 router/专家参数名拼写不一致，或 `params` 展平不彻底，Adam 不会更新它们，模型等价于无 MoE。

**Mitigation**:
- 统一 router 命名（如 `layer{i}.router`），专家统一为 `layer{i}.expert{j}.fc1/fc2`。
- 显式断言 `len(params) == (vocab_size + block_size + vocab_size + n_layer * (n_embd * n_embd * 4 + n_expert * n_embd * 4 * 2 + n_embd * n_expert))`，并把这条断言作为 `print` 输出，便于人眼核对。
- 验证：训练结束后采样模型输出，若结果与 dense 版本几乎无差异，说明 MoE 路径未生效，回查参数列表。

### [Risk] micro_gpt.py 残留的 dense MLP 痕迹被无意中保留
**Severity**: 3 | **Likelihood**: 3 | **Detectability**: 2
**Risk Score**: 9.0

**Failure Scenario**：从原文件复制后忘记删除共享 `attn/mlp` 参数和对应前向代码，导致 MoE 与原 dense MLP 并存，计算图变慢、解释混乱。

**Mitigation**:
- 复制后立即逐个删除 `attn_wq/attn_wk/attn_wv/attn_wo` 之外的 dense MLP 参数声明，仅保留 `attn_*` 和 `router/expert` 系列。
- 在 `gpt()` 中确认 MLP 子块已替换为 `moe()` 调用，dense MLP 不再被调用。
- 全局搜索 `mlp_fc1/mlp_fc2` 应在 `micro_moe.py` 中 0 命中。

### [Risk] 严格类型注解与生成/采样代码冲突
**Severity**: 3 | **Likelihood**: 3 | **Detectability**: 2
**Risk Score**: 9.0

**Failure Scenario**：`mypy --strict` 对 `argmax`、`max` 返回索引、`probs[expert_idx]` 的元素类型推断敏感；若函数返回值类型不显式标注，编译和类型检查会同时失败。

**Mitigation**:
- 为 `moe()`、`router()` 显式声明 `-> list[Value]` 和 `-> int`（专家索引）。
- 对 `argmax` 包装为显式 `def top1(probs: list[float]) -> int: ...`，避免推断为 `Any`。
- 实现完成后立即跑 `mypy --strict` 验证；任一报错都属于实现未完成。

### [Risk] 教学解释过度简略或与实现脱节
**Severity**: 3 | **Likelihood**: 3 | **Detectability**: 2
**Risk Score**: 9.0

**Failure Scenario**：用户期望“清楚解释”但实现与解释脱节（如解释里写 Top-k 但代码是 Top-1），反而比 dense GPT 更难理解。

**Mitigation**:
- 解释必须严格匹配实现：写 Top-1 + 缩放概率，不引入文档中未实现的概念。
- 在解释中并列给出 dense MLP 与 MoE 的伪代码对比，而不是只描述 MoE。
- 明确列出最简实现与生产 MoE 的差异（容量限制、aux loss、并行），与计划中的 Out of Scope 对齐。

### [Risk] 训练/推理循环改动引入与 MoE 无关的回归
**Severity**: 4 | **Likelihood**: 2 | **Detectability**: 3
**Risk Score**: 8.0

**Failure Scenario**：在 `gpt()`、训练循环、采样代码中除了替换 MLP 之外还顺手改了别处，导致 loss 下降但与 MoE 无关，验证失去隔离性。

**Mitigation**:
- 除替换 MLP 子块外，`gpt()` 其他部分字符级复制 `micro_gpt.py`。
- 训练超参（学习率、步数、衰减、初始化 std）保持一致，便于和 dense 结果对照。
- 跑通后用 git diff 复核 `gpt()` 主体差异是否仅在 MLP 子块。

### [Risk] Top-1 选中分支影响反向传播稳定性
**Severity**: 3 | **Likelihood**: 2 | **Detectability**: 3
**Risk Score**: 6.0

**Failure Scenario**：`argmax` 选出的专家若在某次迭代发生改变，loss 出现抖动；如果 router 概率在边界值（如 0.5/0.5）附近反复切换，反向梯度也会震荡。

**Mitigation**:
- 不在 router 端加额外噪声（除一次性初始化），降低训练随机性。
- 在解释中明确说明这是 Top-1 + soft 缩放的已知行为，并把缓解方向指向 top-k 或 aux loss。
- 训练时观察 loss 曲线是否稳定；如 NaN，立即按上一个高风险项处理（停止并标注）。
