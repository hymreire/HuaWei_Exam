# 深度学习

## 高频考点

### 神经网络基础

- 前向传播
- 反向传播
- 激活函数：Sigmoid、Tanh、ReLU、GELU、Softmax
- 损失函数：MSE、交叉熵
- BatchNorm、LayerNorm、Dropout

### CNN

- 卷积核、步长、填充
- 输出尺寸计算
- 池化
- 感受野
- 常见结构：LeNet、AlexNet、VGG、ResNet

### RNN 与序列模型

- RNN
- LSTM
- GRU
- 梯度消失与梯度爆炸

### Transformer

- Self-Attention
- Multi-Head Attention
- Q/K/V
- Position Encoding
- Encoder、Decoder
- Mask

### 分布式训练

- 数据并行 DDP
- 梯度同步
- All-Reduce
- Broadcast、All-Gather、Send/Recv 的区别

## 题目归档

## 题目：DDP 每轮迭代后的梯度同步通信

- 时间：2026-04-15 T1
- 类型：选择题
- 分类：分布式训练 / 深度学习工程 / DDP 通信机制
- 答案：D. All-Reduce

### 题目摘要

在标准数据并行 DDP 训练中，每个 GPU 独立计算梯度后，为了保持模型权重一致，每次迭代结束时需要执行什么通信操作？

### 正确思路

DDP 中每张 GPU 都保存一份完整模型副本，并处理不同的数据子批次。每张 GPU 完成前向传播和反向传播后，会得到自己的本地梯度。

为了让所有 GPU 后续更新出的模型参数保持一致，需要在反向传播后同步梯度。标准 DDP 的核心通信操作是 `All-Reduce`：所有 GPU 对对应位置的梯度做规约操作，通常是求和或等价平均，再把聚合后的结果返回给每张 GPU。这样每张 GPU 拿到一致梯度，再执行相同优化器更新，模型参数就能继续保持一致。

### 选项辨析

- `Broadcast`：一个源节点把数据发给所有节点。常用于初始化参数或同步公共状态，但不能聚合多张卡的梯度。
- `All-Gather`：每个节点贡献自己的数据，每个节点拿到完整拼接结果。它保留各节点原始数据，不负责对应元素求和或平均。
- `Send/Recv`：点对点通信，常见于手写通信逻辑或流水线并行，不是标准 DDP 的每轮梯度同步方案。
- `All-Reduce`：所有节点共同参与规约，再让所有节点都拿到规约结果。它正是 DDP 梯度同步的典型操作。

### 易错点

- DDP 每轮训练同步的重点是梯度，不是重新广播参数。
- All-Gather 是收集完整数据，All-Reduce 是聚合对应位置的数据。
- Send/Recv 能拼出复杂通信逻辑，但不是标准 DDP 的常规答案。

### 需要记住

- `Broadcast`：一个发给所有人。
- `All-Gather`：所有人交数据，每个人拿全套。
- `Send/Recv`：两点之间传输。
- `All-Reduce`：所有人一起聚合，再把结果返回给所有人。
- 一句话：标准 DDP 的关键通信操作是对梯度做 `All-Reduce`。

### 模板

```markdown
## 题目：xxx

- 类型：选择题 / 代码题
- 考点：
- 结论：

### 思路

### 易错点

### 需要记住
```
