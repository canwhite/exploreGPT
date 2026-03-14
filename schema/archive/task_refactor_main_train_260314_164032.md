# Task: 改造 main.py 训练数据获取

**任务ID**: task_refactor_main_train_260314_164032
**创建时间**: 2026-03-14
**状态**: 已完成
**目标**: 参考 micro.py 的数据获取方法，改造 main.py 的训练流程，保持 main features 不变

## 最终目标

1. 数据获取：下载 karpathy names.txt，构建完整分词器 ✅
2. 训练循环：500步，使用 Adam 优化器 ✅
3. 推理生成：批量生成多个样本 ✅

## 拆解步骤

### 1. 数据获取改造 ✅

- [x] 添加 input.txt 下载逻辑（参考 micro.py）
- [x] 构建完整分词器（uchars, BOS, char_to_it, it_to_char）
- [x] 打乱数据顺序

### 2. 训练循环改造 ✅

- [x] 调整训练步数为 500
- [x] 添加 Adam 优化器（m, v 缓存）
- [x] 添加学习率衰减

### 3. 推理生成改造 ✅

- [x] 批量生成多个样本（20个）
- [x] 添加温度采样

## 运行结果

```
num docs: 32033
vocab size: 27
num params: 1224
=== 开始 Hybrid 预训练 ===
Step  100 / 500 | Loss: 3.2978
Step  200 / 500 | Loss: 2.4301
Step  300 / 500 | Loss: 2.4887
Step  400 / 500 | Loss: 2.2724
Step  500 / 500 | Loss: 2.3534

=== 推理生成 (temperature=0.5) ===
sample  1: manli
sample  2: dia
...
sample 20: bakili
```

Loss 从 3.3 下降到 2.3，成功生成 20 个人名样本。
