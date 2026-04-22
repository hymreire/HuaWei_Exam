# Numpy 代码题

## 高频能力

- 数组创建
- reshape、transpose、squeeze、expand_dims
- 广播机制
- fancy indexing
- mask 过滤
- axis 参数
- 矩阵乘法
- 归一化、标准化
- softmax、cross entropy
- 梯度下降手写

## 常见易错点

- `np.array` 与 Python list 的行为差异
- `*` 是逐元素乘法，`@` / `np.matmul` 才是矩阵乘法
- `axis=0` 按列聚合，`axis=1` 按行聚合
- `keepdims=True` 可减少广播错误
- softmax 要减去最大值防止溢出
- 训练/测试集不能混用统计量，避免数据泄漏

## 必背模板

### 稳定 Softmax

```python
import numpy as np

def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
```

### 交叉熵

```python
import numpy as np

def cross_entropy(probs, y):
    eps = 1e-12
    probs = np.clip(probs, eps, 1.0 - eps)
    return -np.mean(np.log(probs[np.arange(len(y)), y]))
```

### 标准化

```python
import numpy as np

def standardize_train_test(x_train, x_test):
    mean = np.mean(x_train, axis=0, keepdims=True)
    std = np.std(x_train, axis=0, keepdims=True)
    std = np.where(std == 0, 1, std)
    return (x_train - mean) / std, (x_test - mean) / std
```

## 题目归档

### 模板

```markdown
## 题目：xxx

- 类型：Numpy 代码题
- 考点：
- 结论：

### 思路

### 易错点

### 代码模板
```

