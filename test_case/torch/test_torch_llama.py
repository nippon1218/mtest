import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest
import math

# ---------------------
# 正向传播测试
# ---------------------

def test_add():
    """测试 torch.add 在正向传播中的加法操作"""
    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([4.0, 5.0, 6.0])
    result = torch.add(a, b)
    expected = torch.tensor([5.0, 7.0, 9.0])
    assert torch.allclose(result, expected)

def test_matmul():
    """测试 torch.matmul（或 torch.mm）在正向传播中的矩阵乘法"""
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    b = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
    result = torch.matmul(a, b)
    expected = torch.tensor([[19.0, 22.0], [43.0, 50.0]])
    assert torch.allclose(result, expected)

def test_div():
    """测试 torch.div 在正向传播中的除法操作"""
    a = torch.tensor([10.0, 20.0, 30.0])
    b = torch.tensor([2.0, 4.0, 5.0])
    result = torch.div(a, b)
    expected = torch.tensor([5.0, 5.0, 6.0])
    assert torch.allclose(result, expected)

def test_linear():
    """测试 nn.Linear 层（内部调用矩阵乘法和加法）"""
    linear = nn.Linear(3, 2)
    # 固定权重和偏置以保证测试可复现
    with torch.no_grad():
        linear.weight.copy_(torch.tensor([[1.0, 2.0, 3.0],
                                          [4.0, 5.0, 6.0]]))
        linear.bias.copy_(torch.tensor([0.5, -0.5]))
    input_tensor = torch.tensor([[1.0, 0.0, -1.0]])
    # 手工计算: 第一行: 1*1 + 0*2 + (-1)*3 + 0.5 = -1.5；第二行: 1*4 + 0*5 + (-1)*6 -0.5 = -2.5
    expected = torch.tensor([[-1.5, -2.5]])
    output = linear(input_tensor)
    assert torch.allclose(output, expected, atol=1e-4)

def test_layernorm():
    """测试 LayerNorm 层：验证输入经过归一化后符合 (x - μ)/σ 的计算结果"""
    x = torch.tensor([[1.0, 2.0, 3.0],
                      [2.0, 4.0, 6.0]], dtype=torch.float32)
    layernorm = nn.LayerNorm(3, elementwise_affine=False)
    
    # 手动计算
    mean = x.mean(dim=1, keepdim=True)
    var = x.var(dim=1, keepdim=True, unbiased=False)  # 匹配PyTorch实现
    std = torch.sqrt(var + layernorm.eps)
    expected = (x - mean) / std
    
    output = layernorm(x)
    assert torch.allclose(output, expected, atol=1e-4)

def test_relu():
    """测试 ReLU 激活函数"""
    x = torch.tensor([-1.0, 0.0, 1.0, 2.0])
    result = F.relu(x)
    expected = torch.tensor([0.0, 0.0, 1.0, 2.0])
    assert torch.allclose(result, expected)

def test_softmax():
    """测试 softmax 计算：输出应为概率分布，和为 1"""
    x = torch.tensor([1.0, 2.0, 3.0])
    result = F.softmax(x, dim=0)
    assert torch.allclose(result.sum(), torch.tensor(1.0), atol=1e-4)

def test_reshape():
    """测试张量变形操作（reshape/view）"""
    x = torch.arange(12)
    y = x.reshape(3, 4)
    assert y.shape == (3, 4)
    z = x.view(4, 3)
    assert z.shape == (4, 3)

def test_embedding():
    """测试嵌入操作"""
    num_embeddings = 10
    embedding_dim = 4
    embedding = nn.Embedding(num_embeddings, embedding_dim)
    input_indices = torch.tensor([1, 3, 5, 7])
    output = embedding(input_indices)
    assert output.shape == (4, embedding_dim)

def test_dropout():
    """测试 Dropout：训练模式下部分元素应为零，评估模式下应不做 dropout"""
    dropout = nn.Dropout(p=0.5)
    dropout.train()  # 训练模式下
    x = torch.ones(1000)
    y = dropout(x)
    num_zeros = (y == 0).sum().item()
    # 预期大约有 50% 元素被置零，允许一定波动
    assert abs(num_zeros - 500) < 100
    dropout.eval()  # 评估模式下
    y_eval = dropout(x)
    assert torch.allclose(x, y_eval)

def test_matmul_special_cases():
    """测试矩阵乘法的特殊输入情况"""
    # 零矩阵相乘
    a = torch.zeros(2,3)
    b = torch.zeros(3,4)
    assert torch.allclose(torch.matmul(a,b), torch.zeros(2,4))
    
    # 单位矩阵特性
    identity = torch.eye(3)
    random_mat = torch.randn(3,5)
    assert torch.allclose(torch.matmul(identity, random_mat), random_mat)
    
    # 非对齐维度（触发异常）
    with pytest.raises(RuntimeError):
        torch.matmul(torch.randn(2,3), torch.randn(4,5))

def test_embedding_edge_cases():
    """测试Embedding层的边界条件"""
    # 索引越界
    embedding = nn.Embedding(10, 8)
    with pytest.raises(IndexError):
        embedding(torch.tensor([-1, 10]))
    
    # 空输入张量
    empty_input = torch.tensor([], dtype=torch.long)
    assert embedding(empty_input).shape == (0,8)
    
    # 全相同索引
    same_indices = torch.tensor([5,5,5])
    output = embedding(same_indices)
    assert torch.allclose(output[0], output[1])

def test_gelu():
    """测试GELU激活函数的基础实现"""
    x = torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0])
    expected = torch.tensor([
        -0.00404951,  # -3.0
        -0.15865529,  # -1.0
        0.0,          # 0.0
        0.8413447,    # 1.0
        2.9959507     # 3.0
    ])
    output = F.gelu(x)
    assert torch.allclose(output, expected, atol=1e-6)

@pytest.mark.parametrize("approximate", ["none", "tanh"])
def test_gelu_approximate(approximate):
    """测试不同近似算法的GELU实现"""
    x = torch.randn(10)
    exact = F.gelu(x, approximate="none")
    approx = F.gelu(x, approximate=approximate)
    
    if approximate == "none":
        assert torch.allclose(exact, approx)
    else:
        # 允许近似方法有一定误差
        assert torch.allclose(exact, approx, atol=1e-3)

def test_gelu_grad():
    """测试GELU的梯度计算"""
    x = torch.tensor([-2.0, 0.5, 3.0], requires_grad=True)
    y = F.gelu(x)
    y.sum().backward()
    
    # 手动计算期望梯度
    def gelu_grad(x):
        return 0.5 * (1 + torch.erf(x / math.sqrt(2))) + \
               (x * torch.exp(-0.5 * x**2)) / math.sqrt(2 * math.pi)
    
    expected_grad = gelu_grad(x)
    assert torch.allclose(x.grad, expected_grad, atol=1e-4)

# ---------------------
# 反向传播测试（由 Autograd 自动生成的梯度算子）
# ---------------------

def test_backward_add():
    """测试加法的反向传播：验证梯度计算正确性"""
    a = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    b = torch.tensor([4.0, 5.0, 6.0])
    c = torch.add(a, b)
    out = c.sum()
    out.backward()
    # 对于 c = a + b, ∂c/∂a 为 1, 故 a.grad 应全为 1
    assert torch.allclose(a.grad, torch.ones_like(a))

if __name__ == "__main__":
    pytest.main([__file__])
