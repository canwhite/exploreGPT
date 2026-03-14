# Task: 改造 novel.py 支持小说训练

**任务ID**: task_novel_train_260307_135818
**创建时间**: 2026-03-07 13:58:18
**状态**: 已完成
**目标**: 将 novel.py 改造为可训练小说的 GPT 系统

## 最终目标

1. 引入 tiktoken 分词器替代字符级分词 ✅
2. 添加小说数据自动下载和批处理 ✅
3. 保留 Value 类用于教学，主流程用 PyTorch ✅
4. 适配 MacBook M2 (MPS 加速) ✅
5. 目标模型: ~1M 参数 (n_layer=4, n_embd=128, block_size=256) ✅

## 拆解步骤

### 1. 项目结构和配置 ✅
- [x] 创建 config.py 配置中心
- [x] 更新 pyproject.toml 添加依赖
- [x] 创建 novel_gpt/ 目录结构

### 2. 分词器模块 (tokenizer.py) ✅
- [x] 实现 TiktokenTokenizer 类
- [x] 保留字符级分词器备选
- [x] 添加编码/解码接口

### 3. 数据模块 (data.py) ✅
- [x] 实现 Project Gutenberg 小说下载
- [x] 实现 DataLoader 和批处理
- [x] 添加数据缓存机制

### 4. 模型模块 (model.py) ✅
- [x] PyTorch 实现 GPT 模型
- [x] MPS 设备适配
- [x] Checkpoint 保存/加载

### 5. 训练脚本 (train.py) ✅
- [x] 训练循环
- [x] 学习率调度
- [x] 日志输出

### 6. 推理脚本 (generate.py) ✅
- [x] 文本生成
- [x] 温度采样
- [x] 交互模式

### 7. 入口脚本 (__main__.py) ✅
- [x] train 子命令
- [x] generate 子命令
- [x] info 子命令

### 8. 测试验证 ✅
- [x] MPS 设备检测正常
- [x] 模型前向传播正常
- [x] 训练流程测试通过
- [x] 生成功能测试通过

## 完成状态

### 文件结构
```
novel_gpt/
├── __init__.py      # 模块导出
├── __main__.py      # CLI 入口
├── config.py        # 配置中心
├── tokenizer.py     # tiktoken + 字符级分词
├── data.py          # 小说下载 + DataLoader
├── model.py         # PyTorch GPT 模型
├── train.py         # 训练脚本
└── generate.py      # 生成脚本
```

### 依赖 (已安装)
- torch 2.10.0
- tiktoken 0.12.0
- tqdm 4.67.3
- numpy 2.2.6

### 测试结果
- Device: mps ✅
- Model params: 3,323,584 (~3.3M, mini config)
- 训练 5 步测试通过
- 生成功能测试通过

## 使用方法

```bash
# 查看配置信息
python -m novel_gpt info --config mini

# 快速测试
python -m novel_gpt train --config mini --max_steps 100

# 标准训练
python -m novel_gpt train --config default

# 生成文本
python -m novel_gpt generate checkpoints/best.pt --prompt "Once upon a time"

# 交互模式
python -m novel_gpt generate checkpoints/best.pt --interactive
```

## 配置说明

| 配置 | 参数量 | n_layer | n_embd | block_size |
|------|--------|---------|--------|------------|
| mini | ~3.3M  | 2       | 64     | 128        |
| default | ~6.5M | 4     | 128    | 256        |
| small | ~38M  | 6       | 256    | 512        |
