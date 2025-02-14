import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest
import math
import allure
import time

# ---------------------
# 正向传播测试
# ---------------------

@allure.epic('llama算子正反向测试')
@allure.story('基础算子测试')
@allure.title('加法运算测试')
def test_add(device):
    """测试 torch.add 在正向传播中的加法操作，包含多种测试场景"""
    print(f"\n在test_add中使用的设备: {device}")
    print(f"CUDA是否可用: {torch.cuda.is_available()}")
    
    # 1. 基本加法测试
    a = torch.tensor([1.0, 2.0, 3.0], device=device)
    b = torch.tensor([4.0, 5.0, 6.0], device=device)
    result = torch.add(a, b)
    expected = torch.tensor([5.0, 7.0, 9.0], device=device)
    assert torch.allclose(result, expected)
    print(f"基本加法测试通过: a={a.device}, b={b.device}, result={result.device}")
    
    # 2. 不同数据类型测试
    a_int = torch.tensor([1, 2, 3], dtype=torch.int32, device=device)
    b_float = torch.tensor([4.0, 5.0, 6.0], dtype=torch.float32, device=device)
    result = torch.add(a_int, b_float)
    assert result.dtype == torch.float32  # 结果应该是float类型
    assert torch.allclose(result, torch.tensor([5.0, 7.0, 9.0], device=device))
    
    # 3. 广播机制测试
    a_broadcast = torch.tensor([[1.0], [2.0]], device=device)  # shape: (2,1)
    b_broadcast = torch.tensor([3.0, 4.0, 5.0], device=device)  # shape: (3,)
    result = torch.add(a_broadcast, b_broadcast)  # 结果shape应该是(2,3)
    assert result.shape == (2, 3)
    expected = torch.tensor([[4.0, 5.0, 6.0], [5.0, 6.0, 7.0]], device=device)
    assert torch.allclose(result, expected)
    
    # 4. 标量加法测试
    scalar = 2.0
    result = torch.add(a, scalar)
    expected = torch.tensor([3.0, 4.0, 5.0], device=device)
    assert torch.allclose(result, expected)
    
    # 5. 零维张量测试
    scalar_tensor = torch.tensor(2.0, device=device)
    result = torch.add(a, scalar_tensor)
    assert torch.allclose(result, expected)
    
    # 6. 边界值测试
    max_val = torch.finfo(torch.float32).max
    min_val = torch.finfo(torch.float32).min
    a_edge = torch.tensor([max_val, min_val], device=device)
    b_edge = torch.tensor([1.0, -1.0], device=device)
    result = torch.add(a_edge, b_edge)
    assert not torch.isnan(result).any()  # 确保结果不包含NaN
    
    # 7. alpha参数测试
    result = torch.add(a, b, alpha=2.0)  # 等价于 a + 2.0 * b
    expected = torch.tensor([9.0, 12.0, 15.0], device=device)
    assert torch.allclose(result, expected)

@allure.epic('llama算子正反向测试')
@allure.story('基础算子测试')
@allure.title('矩阵乘法测试')
def test_matmul(device):
    """
    测试PyTorch的matmul算子在不同场景下的表现，包括：
    1. 基本矩阵乘法测试
    2. 向量运算测试
    3. 广播机制测试
    4. 批处理测试
    5. 特殊情况测试
    6. 数值精度测试
    7. 梯度测试
    8. 性能测试
    """
    print(f"\n在test_matmul中使用的设备: {device}")
    
    # 1. 基本矩阵乘法测试
    # 1.1 2D矩阵乘法
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=device)
    b = torch.tensor([[5.0, 6.0], [7.0, 8.0]], device=device)
    result = torch.matmul(a, b)
    expected = torch.tensor([[19.0, 22.0], [43.0, 50.0]], device=device)
    assert torch.allclose(result, expected)
    print("基本矩阵乘法测试通过")
    
    # 2. 向量运算测试
    # 2.1 向量点积
    v1 = torch.tensor([1.0, 2.0, 3.0], device=device)
    v2 = torch.tensor([4.0, 5.0, 6.0], device=device)
    result = torch.matmul(v1, v2)
    assert torch.allclose(result, torch.tensor(32.0, device=device))
    
    # 3. 广播机制测试
    # 3.1 简单广播
    a = torch.randn(3, 4, 5, device=device)
    b = torch.randn(5, 2, device=device)
    result = torch.matmul(a, b)
    assert result.shape == (3, 4, 2)
    
    # 3.2 复杂广播
    a = torch.randn(2, 3, 4, 5, device=device)
    b = torch.randn(2, 1, 5, 6, device=device)
    result = torch.matmul(a, b)
    assert result.shape == (2, 3, 4, 6)
    print("广播机制测试通过")
    
    # 4. 批处理测试
    # 4.1 批量矩阵乘法
    batch_size = 10
    a = torch.randn(batch_size, 3, 4, device=device)
    b = torch.randn(batch_size, 4, 5, device=device)
    result = torch.matmul(a, b)
    assert result.shape == (batch_size, 3, 5)
    
    # 4.2 不同批大小的广播
    a = torch.randn(10, 1, 3, 4, device=device)
    b = torch.randn(1, 5, 4, 2, device=device)
    result = torch.matmul(a, b)
    assert result.shape == (10, 5, 3, 2)
    print("批处理测试通过")
    
    # 5. 特殊情况测试
    # 5.1 空维度
    a = torch.randn(0, 2, 3, device=device)
    b = torch.randn(3, 4, device=device)
    result = torch.matmul(a, b)
    assert result.shape == (0, 2, 4)
    
    # 5.2 1x1矩阵
    a = torch.tensor([[2.0]], device=device)
    b = torch.tensor([[3.0]], device=device)
    result = torch.matmul(a, b)
    assert torch.allclose(result, torch.tensor([[6.0]], device=device))
    
    # 5.3 特殊值测试
    print("开始特殊值测试...")
    
    # 5.3.1 NaN测试
    print("5.3.1 开始 NaN 测试...")
    
    # 创建包含NaN的矩阵
    nan_matrix = torch.tensor([
        [float('nan'), 1.0],
        [2.0, 3.0]
    ], device=device)
    
    normal_matrix = torch.tensor([
        [1.0, 2.0],
        [3.0, 4.0]
    ], device=device)
    
    # NaN与正常数相乘
    result = torch.matmul(nan_matrix, normal_matrix)
    assert torch.isnan(result[0,0]) and torch.isnan(result[0,1]), "NaN行与正常矩阵相乘应该得到NaN"
    assert not torch.isnan(result[1,0]) and not torch.isnan(result[1,1]), "正常行与正常矩阵相乘不应该得到NaN"
    
    # 全NaN矩阵测试
    all_nan = torch.tensor([[float('nan')]] * 4, device=device).reshape(2, 2)
    result = torch.matmul(all_nan, normal_matrix)
    assert torch.all(torch.isnan(result)), "全NaN矩阵与正常矩阵相乘应该得到全NaN结果"
    
    print("5.3.1 NaN 测试通过")
    
    # 5.3.2 Inf测试
    print("5.3.2 开始 Inf 测试...")
    
    # 测试正无穷
    pos_inf_matrix = torch.tensor([
        [float('inf'), 0.0],
        [1.0, 2.0]
    ], device=device)
    
    pos_result = torch.matmul(pos_inf_matrix, normal_matrix)
    assert torch.isinf(pos_result[0,0]) and pos_result[0,0] > 0, "正无穷与正数相乘应该得到正无穷"
    assert torch.isinf(pos_result[0,1]) and pos_result[0,1] > 0, "正无穷与正数相乘应该得到正无穷"
    assert not torch.any(torch.isinf(pos_result[1])), "正常行与正常矩阵相乘不应该得到无穷"
    
    # 测试负无穷
    neg_inf_matrix = torch.tensor([
        [float('-inf'), 0.0],
        [1.0, 2.0]
    ], device=device)
    
    neg_result = torch.matmul(neg_inf_matrix, normal_matrix)
    assert torch.isinf(neg_result[0,0]) and neg_result[0,0] < 0, "负无穷与正数相乘应该得到负无穷"
    assert torch.isinf(neg_result[0,1]) and neg_result[0,1] < 0, "负无穷与正数相乘应该得到负无穷"
    
    # 测试 Inf * 0
    zero_matrix = torch.zeros(2, 2, device=device)
    zero_result = torch.matmul(pos_inf_matrix, zero_matrix)
    assert torch.all(torch.isnan(zero_result[0])), "Inf * 0 应该得到NaN"
    assert not torch.any(torch.isnan(zero_result[1])), "正常数 * 0 应该得到 0"
    
    # 测试正负无穷相加
    mixed_inf_matrix = torch.tensor([
        [float('inf'), float('-inf')],
        [1.0, 2.0]
    ], device=device)
    
    mixed_result = torch.matmul(mixed_inf_matrix, normal_matrix)
    assert torch.all(torch.isnan(mixed_result[0])), "正负无穷相加应该得到NaN"
    assert not torch.any(torch.isnan(mixed_result[1])), "正常行不应该得到NaN"
    
    print("5.3.2 Inf 测试通过")
    
    # 5.3.4 大数与特殊值测试
    print("5.3.4 开始大数与特殊值测试...")
    
    # 1. 大数与小数的测试
    # 使用较小的大数，避免溢出
    large_num = 1e30
    small_num = 1e-30
    
    large_small = torch.tensor([
        [large_num, small_num],
        [1.0, 2.0]
    ], device=device)
    
    normal_2 = torch.tensor([
        [1.0, 2.0],
        [3.0, 4.0]
    ], device=device)
    
    result_1 = torch.matmul(large_small, normal_2)
    assert torch.isfinite(result_1[0,0]), "较大的数与正常数相乘应该在有限范围内"
    assert torch.isfinite(result_1[0,1]), "小数与正常数相乘应该在有限范围内"
    
    # 2. 极大数与特殊值的测试
    max_float = torch.finfo(torch.float32).max
    min_float = torch.finfo(torch.float32).min
    
    extreme_special = torch.tensor([
        [max_float, float('inf')],
        [min_float, float('nan')]
    ], device=device)
    
    small_matrix = torch.tensor([
        [1e-10, 2e-10],
        [3e-10, 4e-10]
    ], device=device)
    
    result_2 = torch.matmul(extreme_special, small_matrix)
    assert torch.isinf(result_2[0,0]), "极大数与小数相乘可能会溢出成无穷"
    assert torch.isinf(result_2[0,1]), "Inf与正常数相乘应该得到Inf"
    assert torch.isnan(result_2[1,0]), "极小负数与小数相乘可能会下溢成NaN"
    assert torch.isnan(result_2[1,1]), "NaN与正常数相乘应该得到NaN"
    
    # 3. 下溢测试
    tiny_num = torch.finfo(torch.float32).tiny  # 最小的正规化浮点数
    underflow_matrix = torch.tensor([
        [tiny_num, tiny_num],
        [1.0, 2.0]
    ], device=device)
    
    result_3 = torch.matmul(underflow_matrix, small_matrix)
    assert torch.all(result_3[0] == 0), "非常小的数相乘应该下溢成0"
    assert not torch.any(result_3[1] == 0), "正常数相乘不应该下溢"
    
    print("5.3.4 大数与特殊值测试通过")
    
    # 5.3.3 混合特殊值测试
    print("5.3.3 开始混合特殊值测试...")
    
    # 测试 Inf 和 NaN 的不同组合
    # 1. Inf在第一个元素，NaN在第二个元素
    mixed_1 = torch.tensor([
        [float('inf'), float('nan')],
        [1.0, 2.0]
    ], device=device)
    
    result_1 = torch.matmul(mixed_1, normal_matrix)
    assert torch.all(torch.isnan(result_1[0])), "Inf和NaN混合的行与正常矩阵相乘应该得到NaN"
    assert not torch.any(torch.isnan(result_1[1])), "正常行不应该得到NaN"
    
    # 2. NaN在第一个元素，Inf在第二个元素
    mixed_2 = torch.tensor([
        [float('nan'), float('inf')],
        [1.0, 2.0]
    ], device=device)
    
    result_2 = torch.matmul(mixed_2, normal_matrix)
    assert torch.all(torch.isnan(result_2[0])), "NaN和Inf混合的行与正常矩阵相乘应该得到NaN"
    assert not torch.any(torch.isnan(result_2[1])), "正常行不应该得到NaN"
    
    # 3. 测试正负无穷和NaN的组合
    mixed_3 = torch.tensor([
        [float('inf'), float('-inf'), float('nan')],
        [1.0, 2.0, 3.0]
    ], device=device)
    
    normal_3 = torch.tensor([
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0]
    ], device=device)
    
    result_3 = torch.matmul(mixed_3, normal_3)
    assert torch.all(torch.isnan(result_3[0])), "正负无穷和NaN混合的行与正常矩阵相乘应该得到NaN"
    assert not torch.any(torch.isnan(result_3[1])), "正常行不应该得到NaN"
    
    print("5.3.3 混合特殊值测试通过")
    

    
    print("特殊值测试全部通过")
    print("特殊情况测试通过")
    
    # 6. 数值精度测试
    # 6.1 不同数据类型
    dtypes = [torch.float32, torch.float64]
    for dtype in dtypes:
        a = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=dtype, device=device)
        b = torch.tensor([[5.0, 6.0], [7.0, 8.0]], dtype=dtype, device=device)
        result = torch.matmul(a, b)
        assert result.dtype == dtype
    
    # 6.2 大数值
    a = torch.tensor([[1e10, 1e-10], [1e-10, 1e10]], device=device)
    b = torch.tensor([[1e10, 1e-10], [1e-10, 1e10]], device=device)
    result = torch.matmul(a, b)
    assert not torch.any(torch.isinf(result))
    print("数值精度测试通过")
    
    # 7. 梯度测试
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True, device=device)
    b = torch.tensor([[5.0, 6.0], [7.0, 8.0]], requires_grad=True, device=device)
    result = torch.matmul(a, b)
    loss = result.sum()
    loss.backward()
    
    # 梯度形状检查
    assert a.grad.shape == a.shape
    assert b.grad.shape == b.shape
    
    # 梯度值检查
    expected_a_grad = torch.tensor([[11.0, 15.0], [11.0, 15.0]], device=device)
    expected_b_grad = torch.tensor([[4.0, 4.0], [6.0, 6.0]], device=device)
    assert torch.allclose(a.grad, expected_a_grad)
    assert torch.allclose(b.grad, expected_b_grad)
    print("梯度测试通过")
    
    # 8. 性能测试
    import time
    
    # 8.1 大矩阵乘法性能
    matrix_size = 1000
    a = torch.randn(matrix_size, matrix_size, device=device)
    b = torch.randn(matrix_size, matrix_size, device=device)
    
    # 预热
    _ = torch.matmul(a, b)
    torch.cuda.synchronize() if device.type == 'cuda' else None
    
    # 计时
    start_time = time.time()
    result = torch.matmul(a, b)
    torch.cuda.synchronize() if device.type == 'cuda' else None
    end_time = time.time()
    
    print(f"大规模矩阵乘法性能测试:")
    print(f"    矩阵大小: {matrix_size} x {matrix_size}")
    print(f"    计算时间: {end_time - start_time:.4f} 秒")
    print(f"    计算设备: {device}")
    # 检查结果是否有效（不包含NaN或无穷大）
    assert not torch.isnan(result).any() and not torch.isinf(result).any()
    print("性能测试通过")

@allure.epic('llama算子正反向测试')
@allure.story('基础算子测试')
@allure.title('除法运算测试')
def test_div(device):
    """测试 torch.div 在正向传播中的除法操作"""
    a = torch.tensor([10.0, 20.0, 30.0], device=device)
    b = torch.tensor([2.0, 4.0, 5.0], device=device)
    result = torch.div(a, b)
    expected = torch.tensor([5.0, 5.0, 6.0], device=device)
    assert torch.allclose(result, expected)

@allure.epic('llama算子正反向测试')
@allure.story('神经网络层测试')
@allure.title('线性层测试')
def test_linear(device):
    """测试 nn.Linear 层（内部调用矩阵乘法和加法）"""
    linear = nn.Linear(3, 2).to(device)  # 将模型移动到指定设备
    # 固定权重和偏置以保证测试可复现
    with torch.no_grad():
        linear.weight.copy_(torch.tensor([[1.0, 2.0, 3.0],
                                          [4.0, 5.0, 6.0]], device=device))
        linear.bias.copy_(torch.tensor([0.5, -0.5], device=device))
    # 使用固定的输入值
    input_tensor = torch.tensor([[1.0, 0.0, -1.0]], device=device)
    # 手工计算: 第一行: 1*1 + 0*2 + (-1)*3 + 0.5 = -1.5；第二行: 1*4 + 0*5 + (-1)*6 -0.5 = -2.5
    expected = torch.tensor([[-1.5, -2.5]], device=device)
    output = linear(input_tensor)
    assert torch.allclose(output, expected, atol=1e-4)

@allure.epic('llama算子正反向测试')
@allure.story('规范化层测试')
@allure.title('LayerNorm测试')
def test_layernorm(device):
    """测试 LayerNorm 层：验证输入经过归一化后符合 (x - μ)/σ 的计算结果"""
    x = torch.tensor([[1.0, 2.0, 3.0],
                      [2.0, 4.0, 6.0]], device=device, dtype=torch.float32)
    layernorm = nn.LayerNorm(3, elementwise_affine=False)
    
    # 手动计算
    mean = x.mean(dim=1, keepdim=True)
    var = x.var(dim=1, keepdim=True, unbiased=False)  # 匹配PyTorch实现
    std = torch.sqrt(var + layernorm.eps)
    expected = (x - mean) / std
    
    output = layernorm(x)
    assert torch.allclose(output, expected, atol=1e-4)

@allure.epic('llama算子正反向测试')
@allure.story('激活函数测试')
@allure.title('ReLU测试')
def test_relu(device):
    """测试 ReLU 激活函数"""
    x = torch.tensor([-1.0, 0.0, 1.0, 2.0], device=device)
    result = F.relu(x)
    expected = torch.tensor([0.0, 0.0, 1.0, 2.0], device=device)
    assert torch.allclose(result, expected)

@allure.epic('llama算子正反向测试')
@allure.story('激活函数测试')
@allure.title('Softmax测试')
def test_softmax(device):
    """测试 softmax 计算的各种场景，包括：
    1. 基本功能测试
    2. 多维张量测试
    3. 数值稳定性测试
    4. 批处理维度测试
    5. 数学特性验证
    """
    # 1. 基本功能测试
    x = torch.tensor([1.0, 2.0, 3.0], device=device)
    result = F.softmax(x, dim=0)
    # 验证输出和为1
    assert torch.allclose(result.sum(), torch.tensor(1.0, device=device))
    # 验证输出在[0,1]范围内
    assert torch.all(result >= 0) and torch.all(result <= 1)
    
    # 2. 多维张量测试
    x_2d = torch.tensor([[1.0, 2.0, 3.0],
                        [4.0, 5.0, 6.0]], device=device)
    # 在不同维度上测试softmax
    result_dim0 = F.softmax(x_2d, dim=0)  # 在第0维上做softmax
    result_dim1 = F.softmax(x_2d, dim=1)  # 在第1维上做softmax
    # 验证每个维度的和都为1
    assert torch.allclose(result_dim0.sum(dim=0), torch.ones(3, device=device))
    assert torch.allclose(result_dim1.sum(dim=1), torch.ones(2, device=device))
    
    # 3. 数值稳定性测试
    # 测试大数
    x_large = torch.tensor([1000.0, 2000.0, 3000.0], device=device)
    result_large = F.softmax(x_large, dim=0)
    assert not torch.any(torch.isnan(result_large))
    assert not torch.any(torch.isinf(result_large))
    
    # 测试小数
    x_small = torch.tensor([-1000.0, -2000.0, -3000.0], device=device)
    result_small = F.softmax(x_small, dim=0)
    assert not torch.any(torch.isnan(result_small))
    assert not torch.any(torch.isinf(result_small))
    
    # 4. 批处理维度测试
    batch_size = 3
    seq_len = 4
    hidden_size = 5
    x_3d = torch.randn(batch_size, seq_len, hidden_size, device=device)
    # 在最后一个维度上做softmax（常见于注意力机制）
    result_3d = F.softmax(x_3d, dim=-1)
    # 验证每个样本、每个位置的softmax和都为1
    assert torch.allclose(result_3d.sum(dim=-1), 
                         torch.ones(batch_size, seq_len, device=device))
    
    # 5. 特殊值测试
    # 测试相等的输入值
    x_equal = torch.ones(5, device=device)
    result_equal = F.softmax(x_equal, dim=0)
    expected_equal = torch.full_like(x_equal, 1.0/5)
    assert torch.allclose(result_equal, expected_equal)
    assert torch.allclose(result.sum(), torch.tensor(1.0, device=device), atol=1e-4)

@allure.epic('llama算子正反向测试')
@allure.story('张量操作测试')
@allure.title('张量变形测试')
def test_reshape(device):
    """测试张量变形操作（reshape/view）"""
    x = torch.arange(12, device=device)
    y = x.reshape(3, 4)
    assert y.shape == (3, 4)
    z = x.view(4, 3)
    assert z.shape == (4, 3)

@allure.epic('llama算子正反向测试')
@allure.story('嵌入层测试')
@allure.title('Embedding基础测试')
def test_embedding(device):
    """测试嵌入操作"""
    num_embeddings = 10
    embedding_dim = 4
    embedding = nn.Embedding(num_embeddings, embedding_dim).to(device)
    input_indices = torch.tensor([1, 3, 5, 7], device=device)
    output = embedding(input_indices)
    assert output.shape == (4, embedding_dim)

@allure.epic('llama算子正反向测试')
@allure.story('正则化测试')
@allure.title('Dropout测试')
def test_dropout(device):
    """测试 Dropout：训练模式下部分元素应为零，评估模式下应不做 dropout"""
    dropout = nn.Dropout(p=0.5)
    dropout.train()  # 训练模式下
    x = torch.ones(1000, device=device)
    y = dropout(x)
    num_zeros = (y == 0).sum().item()
    # 预期大约有 50% 元素被置零，允许一定波动
    assert abs(num_zeros - 500) < 100
    dropout.eval()  # 评估模式下
    y_eval = dropout(x)
    assert torch.allclose(x, y_eval)

@allure.epic('llama算子正反向测试')
@allure.story('边界条件测试')
@allure.title('矩阵乘法特殊场景测试')
def test_matmul_special_cases(device):
    """测试矩阵乘法的特殊输入情况"""
    # 零矩阵相乘
    a = torch.zeros(2,3, device=device)
    b = torch.zeros(3,4, device=device)
    assert torch.allclose(torch.matmul(a,b), torch.zeros(2,4, device=device))
    
    # 单位矩阵特性
    identity = torch.eye(3, device=device)
    random_mat = torch.randn(3,5, device=device)
    assert torch.allclose(torch.matmul(identity, random_mat), random_mat)
    
    # 非对齐维度（触发异常）
    with pytest.raises(RuntimeError):
        torch.matmul(torch.randn(2,3, device=device), torch.randn(4,5, device=device))

@allure.epic('llama算子正反向测试')
@allure.story('边界条件测试')
@allure.title('Embedding边界测试')
def test_embedding_edge_cases(device):
    """测试Embedding层的边界条件"""
    # 索引越界
    embedding = nn.Embedding(10, 8).to(device)
    if device.type == 'cuda':
        # 在CUDA设备上，我们跳过越界测试，因为它会触发device-side assert
        pytest.skip('Skipping index out of bounds test on CUDA device')
    else:
        # CPU设备上会抛出IndexError
        with pytest.raises(IndexError):
            embedding(torch.tensor([-1, 10], device=device))
    
    # 空输入张量
    empty_input = torch.tensor([], dtype=torch.long, device=device)
    assert embedding(empty_input).shape == (0,8)
    
    # 全相同索引
    same_indices = torch.tensor([5,5,5], device=device)
    output = embedding(same_indices)
    assert torch.allclose(output[0], output[1])

# @allure.epic('llama算子正反向测试')
# @allure.story('激活函数测试')
# @allure.title('GELU基础测试')
# def test_gelu(device):
#     """测试GELU激活函数的基础实现"""
#     x = torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0], device=device)
#     expected = torch.tensor([
#         -0.00404951,  # -3.0
#         -0.15865529,  # -1.0
#         0.0,          # 0.0
#         0.8413447,    # 1.0
#         2.9959507     # 3.0
#     ], device=device)
#     output = F.gelu(x)
#     assert torch.allclose(output, expected, atol=1e-6)

# @allure.epic('llama算子正反向测试')
# @allure.story('激活函数测试')
# @allure.title('GELU近似算法测试')
# @pytest.mark.parametrize("approximate", ["none", "tanh"])
# def test_gelu_approximate(device, approximate):
#     """测试不同近似算法的GELU实现"""
#     x = torch.randn(10, device=device)
#     exact = F.gelu(x, approximate="none")
#     approx = F.gelu(x, approximate=approximate)
#     
#     if approximate == "none":
#         assert torch.allclose(exact, approx)
#     else:
#         # 允许近似方法有一定误差
#         assert torch.allclose(exact, approx, atol=1e-3)

# @allure.epic('llama算子正反向测试')
# @allure.story('梯度计算测试')
# @allure.title('GELU梯度测试')
# def test_gelu_grad(device):
#     """测试GELU的梯度计算"""
#     x = torch.tensor([-2.0, 0.5, 3.0], device=device, requires_grad=True)
#     y = F.gelu(x)
#     y.sum().backward()
#     
#     # 手动计算期望梯度
#    def gelu_grad(x):
#        return 0.5 * (1 + torch.erf(x / math.sqrt(2))) + \
#               (x * torch.exp(-0.5 * x**2)) / math.sqrt(2 * math.pi)
#    
#    expected_grad = gelu_grad(x)
#    assert torch.allclose(x.grad, expected_grad, atol=1e-4)

# ---------------------
# 反向传播测试（由 Autograd 自动生成的梯度算子）
# ---------------------

@allure.epic('llama算子正反向测试')
@allure.story('反向传播测试')
@allure.title('加法反向传播测试')
def test_backward_add(device):
    """测试加法的反向传播：验证梯度计算正确性"""
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True, device=device)
    b = torch.tensor([[5.0, 6.0], [7.0, 8.0]], requires_grad=True, device=device)
    c = torch.add(a, b)
    out = c.sum()
    out.backward()
    # 对于 c = a + b, ∂c/∂a 为 1, 故 a.grad 应全为 1
    assert torch.allclose(a.grad, torch.ones_like(a))

@allure.epic('llama算子正反向测试')
@allure.story('RoPE位置编码测试')
@allure.title('RoPE算子测试')
def test_rope(device):
    """测试 Rotary Position Embedding (RoPE) 算子"""
    # 设置参数
    batch_size = 2
    seq_len = 4
    num_heads = 2
    head_dim = 32
    dim = num_heads * head_dim
    
    # 创建输入张量
    x = torch.randn(batch_size, seq_len, dim, device=device)
    
    # 计算 RoPE 的位置编码
    def get_rope_cache(seq_len: int, n_elem: int, device: torch.device) -> torch.Tensor:
        """生成 RoPE 的位置编码缓存"""
        # 计算角度频率
        theta = 1.0 / (10000 ** (torch.arange(0, n_elem, 2, device=device).float() / n_elem))
        seq_idx = torch.arange(seq_len, device=device).float()
        
        # 计算角度
        idx_theta = torch.outer(seq_idx, theta)
        cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)
        
        # 展平为 [seq_len, n_elem]
        cache = cache.view(seq_len, n_elem//2, 2)
        return cache
    
    def apply_rope(x: torch.Tensor, rope_cache: torch.Tensor) -> torch.Tensor:
        """应用 RoPE 位置编码"""
        # 重整形状为 [batch_size, seq_len, num_heads, head_dim]
        x = x.view(batch_size, seq_len, num_heads, head_dim)
        
        # 将每个头的维度拆分为两半，以便于旋转
        x_reshape = x.view(batch_size, seq_len, num_heads, head_dim//2, 2)
        
        # 应用旋转
        cos, sin = rope_cache[..., 0], rope_cache[..., 1]
        cos = cos.view(seq_len, 1, head_dim//2)
        sin = sin.view(seq_len, 1, head_dim//2)
        
        x_out = torch.empty_like(x_reshape)
        x_out[..., 0] = x_reshape[..., 0] * cos - x_reshape[..., 1] * sin
        x_out[..., 1] = x_reshape[..., 0] * sin + x_reshape[..., 1] * cos
        
        # 还原形状
        return x_out.view(batch_size, seq_len, dim)
    
    # 获取 RoPE 缓存
    rope_cache = get_rope_cache(seq_len, head_dim, device)
    
    # 应用 RoPE
    output = apply_rope(x, rope_cache)
    
    # 验证输出形状
    assert output.shape == (batch_size, seq_len, dim)
    
    # 验证不同位置的编码是不同的
    pos1_encoding = output[:, 0, :]
    pos2_encoding = output[:, 1, :]
    assert not torch.allclose(pos1_encoding, pos2_encoding)
    
    # 验证编码后的值仍然在合理范围内
    assert torch.all(torch.isfinite(output))
    
    # 验证旋转不改变向量长度
    input_norms = torch.norm(x, dim=-1)
    output_norms = torch.norm(output, dim=-1)
    assert torch.allclose(input_norms, output_norms, rtol=1e-5)

@allure.epic('llama算子正反向测试')
@allure.story('Fused Adam优化器测试')
@allure.title('Fused Adam算子测试')
def test_fused_adam(device):
    """测试 Fused Adam 优化器"""
    if device.type != 'cuda':
        pytest.skip('Fused Adam only works on CUDA device')
        
    try:
        from apex.optimizers import FusedAdam
    except ImportError:
        pytest.skip('apex is not installed')
    
    # 设置参数
    input_size = 128
    hidden_size = 64
    output_size = 32
    batch_size = 16
    num_steps = 5
    
    # 创建一个简单的神经网络
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.fc2 = nn.Linear(hidden_size, output_size)
            
        def forward(self, x):
            x = F.relu(self.fc1(x))
            return self.fc2(x)
    
    # 创建模型和优化器
    model = SimpleModel().to(device)
    
    # 创建两个相同的模型副本，一个用普通 Adam，一个用 Fused Adam
    model_adam = copy.deepcopy(model)
    model_fused = copy.deepcopy(model)
    
    # 创建优化器
    optimizer_adam = torch.optim.Adam(model_adam.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8)
    optimizer_fused = FusedAdam(model_fused.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8)
    
    # 生成测试数据
    torch.manual_seed(42)
    for step in range(num_steps):
        # 生成随机输入和目标
        x = torch.randn(batch_size, input_size, device=device)
        target = torch.randn(batch_size, output_size, device=device)
        
        # 普通 Adam 前向和反向传播
        output_adam = model_adam(x)
        loss_adam = F.mse_loss(output_adam, target)
        optimizer_adam.zero_grad()
        loss_adam.backward()
        optimizer_adam.step()
        
        # Fused Adam 前向和反向传播
        output_fused = model_fused(x)
        loss_fused = F.mse_loss(output_fused, target)
        optimizer_fused.zero_grad()
        loss_fused.backward()
        optimizer_fused.step()
        
        # 验证两个优化器产生的梯度更新是相似的
        for p1, p2 in zip(model_adam.parameters(), model_fused.parameters()):
            # 由于浮点数计算的差异，我们使用一个相对较大的容差
            assert torch.allclose(p1.data, p2.data, rtol=1e-3, atol=1e-3)
    
    # 验证两个模型的最终损失值相近
    assert abs(loss_adam.item() - loss_fused.item()) < 1e-2

@allure.epic('llama算子正反向测试')
@allure.story('基础运算测试')
@allure.title('乘法算子测试')
def test_mul(device):
    """测试乘法算子的各种情况，包括标量乘法、向量乘法、广播乘法、特殊值处理等"""
    print(f"\n在test_mul中使用的设备: {device}")
    
    # 1. 标量乘法测试
    a = torch.tensor(2.5, device=device)
    b = torch.tensor(3.0, device=device)
    result = torch.mul(a, b)
    expected = torch.tensor(7.5, device=device)
    assert torch.allclose(result, expected)
    print("标量乘法测试通过")
    
    # 2. 向量乘法测试（逐元素）
    a = torch.tensor([1.0, 2.0, 3.0], device=device)
    b = torch.tensor([4.0, 5.0, 6.0], device=device)
    result = torch.mul(a, b)
    expected = torch.tensor([4.0, 10.0, 18.0], device=device)
    assert torch.allclose(result, expected)
    print("向量乘法测试通过")
    
    # 3. 不同数据类型测试
    a_int = torch.tensor([1, 2, 3], dtype=torch.int32, device=device)
    b_float = torch.tensor([1.5, 2.5, 3.5], dtype=torch.float32, device=device)
    result = torch.mul(a_int, b_float)
    assert result.dtype == torch.float32  # 结果应该是float类型
    expected = torch.tensor([1.5, 5.0, 10.5], device=device)
    assert torch.allclose(result, expected)
    print("不同数据类型乘法测试通过")
    
    # 4. 广播乘法测试
    a = torch.tensor([[1.0], [2.0], [3.0]], device=device)  # shape: (3,1)
    b = torch.tensor([2.0, 3.0, 4.0], device=device)        # shape: (3,)
    result = torch.mul(a, b)  # 结果shape应该是(3,3)
    expected = torch.tensor(
        [[2.0, 3.0, 4.0],
         [4.0, 6.0, 8.0],
         [6.0, 9.0, 12.0]], device=device)
    assert torch.allclose(result, expected)
    print("广播乘法测试通过")
    
    # 5. 特殊值测试
    print("开始特殊值测试...")
    
    # 5.1 基本特殊值测试
    special_values = torch.tensor([
        0.0,                    # 零
        1.0,                    # 单位元
        -1.0,                   # 负单位元
        float('inf'),           # 正无穷
        float('-inf'),          # 负无穷
        float('nan')            # NaN
    ], device=device)
    
    # 测试与1相乘
    result = torch.mul(special_values, torch.tensor(1.0, device=device))
    assert torch.all(torch.isfinite(result[:-3]))  # 前三个值应该是有限的
    assert torch.isinf(result[-2]) and torch.isinf(result[-3])  # 无穷值
    assert torch.isnan(result[-1])  # NaN值
    print("5.1 基本特殊值测试通过")
    
    # 5.2 NaN与各种值的乘法测试
    nan = torch.tensor(float('nan'), device=device)
    test_values = torch.tensor([0.0, 1.0, -1.0, float('inf'), float('-inf')], device=device)
    for val in test_values:
        result = torch.mul(nan, val)
        assert torch.isnan(result), f"NaN * {val} 应该返回NaN"
    print("5.2 NaN乘法测试通过")
    
    # 5.3 Inf与各种值的乘法测试
    inf = torch.tensor(float('inf'), device=device)
    neg_inf = torch.tensor(float('-inf'), device=device)
    
    # Inf * 0 = NaN
    assert torch.isnan(torch.mul(inf, torch.tensor(0.0, device=device)))
    assert torch.isnan(torch.mul(neg_inf, torch.tensor(0.0, device=device)))
    
    # Inf * Inf = Inf
    assert torch.isinf(torch.mul(inf, inf)) and torch.mul(inf, inf) > 0
    
    # Inf * (-Inf) = -Inf
    assert torch.isinf(torch.mul(inf, neg_inf)) and torch.mul(inf, neg_inf) < 0
    
    # Inf * 正数 = Inf
    assert torch.isinf(torch.mul(inf, torch.tensor(2.0, device=device))) and torch.mul(inf, torch.tensor(2.0, device=device)) > 0
    
    # Inf * 负数 = -Inf
    assert torch.isinf(torch.mul(inf, torch.tensor(-2.0, device=device))) and torch.mul(inf, torch.tensor(-2.0, device=device)) < 0
    print("5.3 Inf乘法测试通过")
    
    # 5.4 数值稳定性测试
    tiny = torch.finfo(torch.float32).tiny  # 最小正数
    huge = torch.finfo(torch.float32).max   # 最大有限数
    
    # 测试极小数乘极大数
    result = torch.mul(torch.tensor(tiny, device=device), torch.tensor(huge, device=device))
    assert torch.isfinite(result), "极小数乘极大数应该仍然在有限范围内"
    
    # 测试下溢
    result = torch.mul(torch.tensor(tiny, device=device), torch.tensor(tiny, device=device))
    assert result == 0.0, "两个极小数相乘应该下溢为0"
    
    # 测试上溢
    result = torch.mul(torch.tensor(huge, device=device), torch.tensor(huge, device=device))
    assert torch.isinf(result) and result > 0, "两个极大数相乘应该上溢为正无穷"
    print("5.4 数值稳定性测试通过")
    
    print("特殊值测试全部通过")
    
    # 6. 大数乘法测试
    print("开始大数乘法测试...")
    
    # 6.1 基本大数测试
    max_float = torch.finfo(torch.float32).max  # 最大浮点数
    min_float = torch.finfo(torch.float32).min  # 最小浮点数
    eps = torch.finfo(torch.float32).eps        # 最小精度
    
    # 测试大数与小数相乘
    a = torch.tensor([max_float, min_float, max_float/2], device=device)
    b = torch.tensor([0.5, -0.5, 2.0], device=device)
    result = torch.mul(a, b)
    assert not torch.isnan(result).any()  # 确保结果不包含NaN
    assert torch.isfinite(result[2])      # 确保较小的结果是有限的
    print("6.1 基本大数测试通过")
    
    # 6.2 数值范围農界测试
    large_nums = torch.tensor([
        max_float,           # 最大值
        max_float - eps,     # 最大值减去最小精度
        max_float * 0.99,    # 接近最大值
        min_float,           # 最小值
        min_float + eps,     # 最小值加上最小精度
        min_float * 0.99     # 接近最小值
    ], device=device)
    
    # 测试与1接近的数相乘
    small_factors = torch.tensor([0.999, 1.0, 1.001], device=device)
    for num in large_nums:
        for factor in small_factors:
            result = torch.mul(num, factor)
            assert not torch.isnan(result), f"{num} * {factor} 不应该产生NaN"
    print("6.2 数值范围農界测试通过")
    
    # 6.3 精度测试
    precision_test = torch.tensor([
        1e30, 1e-30,         # 非常大和非常小的数
        1e15, 1e-15,         # 中等大小的数
        1e7, 1e-7            # 相对较小的数
    ], device=device)
    
    for i in range(0, len(precision_test), 2):
        large = precision_test[i]
        small = precision_test[i+1]
        product = torch.mul(large, small)
        # 验证结果是否接近1
        assert torch.abs(product - 1.0) < 1e-5, f"{large} * {small} 应该接近1"
    print("6.3 精度测试通过")
    
    # 6.4 矩阵大数测试
    print("开始矩阵大数测试...")
    
    # 6.4.1 有限大数矩阵测试
    matrix_a = torch.tensor([
        [max_float/8, max_float/16],
        [max_float/4, max_float/8]
    ], device=device)
    
    matrix_b = torch.tensor([
        [0.5, 1.0],
        [1.0, 0.5]
    ], device=device)
    
    result = torch.mul(matrix_a, matrix_b)
    assert torch.all(torch.isfinite(result)), "有限大数矩阵乘法应该产生有限的结果"
    print("6.4.1 有限大数矩阵测试通过")
    
    # 6.4.2 溢出测试
    overflow_matrix = torch.tensor([
        [max_float/2, max_float/2],
        [max_float/2, max_float/2]
    ], device=device)
    
    # 预期会溢出的乘法
    result = torch.mul(overflow_matrix, torch.tensor(3.0, device=device))
    assert torch.any(torch.isinf(result)), "大数乘以3应该产生溢出"
    print("6.4.2 溢出测试通过")
    
    # 6.4.3 混合运算测试
    mixed_matrix_a = torch.tensor([
        [max_float/4, 1.0],
        [1.0, max_float/4]
    ], device=device)
    
    mixed_matrix_b = torch.tensor([
        [0.5, 1000.0],
        [1000.0, 0.5]
    ], device=device)
    
    result = torch.mul(mixed_matrix_a, mixed_matrix_b)
    # 检查对角线上的元素是否有限
    assert torch.isfinite(result[0,0]) and torch.isfinite(result[1,1]), "混合运算中的小数乘法应该有限"
    print("6.4.3 混合运算测试通过")
    
    print("6.4 矩阵大数测试通过")
    
    print("大数乘法测试全部通过")
    
    # 7. 就地乘法测试（inplace multiplication）
    a = torch.tensor([1.0, 2.0, 3.0], device=device)
    a_copy = a.clone()
    a.mul_(2.0)  # 就地乘法
    expected = torch.tensor([2.0, 4.0, 6.0], device=device)
    assert torch.allclose(a, expected)
    assert not torch.allclose(a, a_copy)  # 确保原始张量已被修改
    print("就地乘法测试通过")
    
    # 8. 梯度测试
    a = torch.tensor([1.0, 2.0, 3.0], requires_grad=True, device=device)
    b = torch.tensor([2.0, 2.0, 2.0], device=device)
    result = torch.mul(a, b)
    loss = result.sum()
    loss.backward()
    assert torch.allclose(a.grad, b)  # 验证梯度是否正确
    print("梯度测试通过")
    a = torch.tensor([1.0, 2.0, 3.0], device=device)
    b = torch.tensor([2.0, 3.0, 4.0], device=device)
    result = torch.mul(a, b)
    expected = torch.tensor([2.0, 6.0, 12.0], device=device)
    assert torch.allclose(result, expected)
    
    # 矩阵元素乘法
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=device)
    b = torch.tensor([[2.0, 3.0], [4.0, 5.0]], device=device)
    result = torch.mul(a, b)
    expected = torch.tensor([[2.0, 6.0], [12.0, 20.0]], device=device)
    assert torch.allclose(result, expected)
    
    # 广播乘法
    a = torch.tensor([[1.0], [2.0]], device=device)  # shape: [2, 1]
    b = torch.tensor([2.0, 3.0], device=device)      # shape: [2]
    result = torch.mul(a, b)                         # shape: [2, 2]
    expected = torch.tensor([[2.0, 3.0], [4.0, 6.0]], device=device)
    assert torch.allclose(result, expected)
    
    # 测试大规模矩阵乘法的性能
    size = 1000
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)
    
    # 记录计算时间
    start_time = time.time()
    result = torch.mul(a, b)
    torch.cuda.synchronize()  # 确保 GPU 计算完成
    end_time = time.time()
    
    # 验证结果正确性
    assert result.shape == (size, size)
    assert torch.all(torch.isfinite(result))
    
    # 打印性能信息
    print(f"""大规模矩阵乘法性能测试:
    矩阵大小: {size} x {size}
    计算时间: {end_time - start_time:.4f} 秒
    计算设备: {device}
    """)

@allure.epic('llama算子正反向测试')
@allure.story('规范化层测试')
@allure.title('RMSNorm测试')
def test_rmsnorm(device):
    """测试 RMSNorm 层：验证输入经过RMS归一化后的计算结果"""
    # 设置输入数据
    x = torch.tensor([[1.0, 2.0, 3.0],
                      [2.0, 4.0, 6.0]], device=device, dtype=torch.float32)
    
    # 设置权重参数
    weight = torch.ones(3, device=device)  # 初始化为1，这样不会影响归一化结果
    eps = 1e-6
    
    # 手动计算RMSNorm
    # RMSNorm只使用均方根进行归一化，不使用均值
    rms = torch.sqrt(torch.mean(x * x, dim=1, keepdim=True) + eps)
    expected = x / rms * weight
    
    # 使用PyTorch实现RMSNorm
    def rms_norm(x, weight, eps):
        rms = torch.sqrt(torch.mean(x * x, dim=1, keepdim=True) + eps)
        return x / rms * weight
    
    output = rms_norm(x, weight, eps)
    
    # 验证结果
    assert torch.allclose(output, expected, atol=1e-4)
    
    # 验证输出的scale不变性
    scale = 2.0
    scaled_output = rms_norm(x * scale, weight, eps)
    assert torch.allclose(scaled_output, output, atol=1e-4)

@allure.epic('llama算子正反向测试')
@allure.story('损失函数测试')
@allure.title('CrossEntropy测试')
def test_cross_entropy(device):
    """测试 CrossEntropy 损失函数的各种场景，包括：
    1. 基本功能测试
    2. 批处理数据测试
    3. 权重测试
    4. 忽略索引测试
    5. 数值稳定性测试
    """
    # 1. 基本功能测试
    # 创建一个简单的二分类问题
    logits = torch.tensor([[0.6, 0.4], [0.4, 0.6]], device=device)
    targets = torch.tensor([0, 1], device=device)
    loss = F.cross_entropy(logits, targets)
    # 验证损失值是正数
    assert loss > 0
    # 验证损失值是标量
    assert loss.dim() == 0
    
    # 2. 批处理数据测试
    batch_size = 3
    num_classes = 4
    # 创建一个多分类问题
    logits = torch.randn(batch_size, num_classes, device=device)
    targets = torch.randint(0, num_classes, (batch_size,), device=device)
    loss = F.cross_entropy(logits, targets, reduction='none')
    # 验证输出形状
    assert loss.shape == (batch_size,)
    
    # 3. 权重测试
    weights = torch.tensor([0.2, 0.3, 0.4, 0.1], device=device)
    weighted_loss = F.cross_entropy(logits, targets, weight=weights)
    # 验证加权后的损失仍然是标量
    assert weighted_loss.dim() == 0
    
    # 4. 忽略索引测试
    # 创建包含忽略索引的数据
    logits = torch.randn(4, 5, device=device)
    # 使用-100作为忽略索引
    targets = torch.tensor([1, -100, 3, 2], device=device)
    ignore_loss = F.cross_entropy(logits, targets, ignore_index=-100)
    # 验证损失值仍然有效
    assert not torch.isnan(ignore_loss)
    
    # 5. 数值稳定性测试
    # 测试非常大的logits
    large_logits = torch.tensor([[1000.0, -1000.0], [-1000.0, 1000.0]], device=device)
    targets = torch.tensor([0, 1], device=device)
    large_loss = F.cross_entropy(large_logits, targets)
    assert not torch.isnan(large_loss)
    assert not torch.isinf(large_loss)
    
    # 测试不同的reduction模式
    logits = torch.randn(3, 4, device=device)
    targets = torch.randint(0, 4, (3,), device=device)
    # mean reduction
    mean_loss = F.cross_entropy(logits, targets, reduction='mean')
    assert mean_loss.dim() == 0
    # sum reduction
    sum_loss = F.cross_entropy(logits, targets, reduction='sum')
    assert sum_loss.dim() == 0
    # none reduction
    none_loss = F.cross_entropy(logits, targets, reduction='none')
    assert none_loss.shape == (3,)

@allure.epic('llama算子正反向测试')
@allure.story('神经网络层测试')
@allure.title('FFN测试')
def test_ffn(device):
    """测试 Feed-Forward Network 的各种场景，包括：
    1. 基本功能测试
    2. 批处理和序列长度测试
    3. 激活函数测试
    4. 残差连接测试
    5. Dropout测试
    """
    
    class FFN(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
            super().__init__()
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, output_dim)
            self.dropout = nn.Dropout(dropout)
            self.act = nn.SiLU()  # SiLU/Swish activation used in LLaMA
        
        def forward(self, x, add_residual=True):
            residual = x
            x = self.fc1(x)
            x = self.act(x)
            x = self.dropout(x)
            x = self.fc2(x)
            if add_residual:
                x = x + residual
            return x
    
    # 1. 基本功能测试
    input_dim = 64
    hidden_dim = 256  # 通常是输入维度的4倍
    batch_size = 2
    seq_len = 3
    
    ffn = FFN(input_dim, hidden_dim, input_dim).to(device)
    x = torch.randn(batch_size, seq_len, input_dim, device=device)
    output = ffn(x)
    
    # 验证输出形状
    assert output.shape == (batch_size, seq_len, input_dim)
    # 验证输出不全为零
    assert not torch.all(output == 0)
    
    # 2. 批处理和序列长度测试
    # 测试不同的批大小和序列长度
    test_sizes = [(1, 1), (4, 10), (8, 5)]
    for b, s in test_sizes:
        x = torch.randn(b, s, input_dim, device=device)
        output = ffn(x)
        assert output.shape == (b, s, input_dim)
    
    # 3. 激活函数测试
    # 测试SiLU激活函数的属性
    x = torch.randn(batch_size, seq_len, input_dim, device=device)
    output_with_act = ffn(x)
    # 验证输出不全为正或全为负
    assert torch.any(output_with_act > 0) and torch.any(output_with_act < 0)
    
    # 4. 残差连接测试
    x = torch.randn(batch_size, seq_len, input_dim, device=device)
    # 测试有残差和无残差的情况
    output_with_residual = ffn(x, add_residual=True)
    output_without_residual = ffn(x, add_residual=False)
    # 验证有残差和无残差的输出不同
    assert not torch.allclose(output_with_residual, output_without_residual)
    # 验证有残差的输出与输入有关联
    assert torch.any(torch.abs(output_with_residual - x) < torch.abs(output_without_residual - x))
    
    # 5. Dropout测试
    ffn.train()  # 设置为训练模式
    x = torch.randn(100, 1, input_dim, device=device)  # 使用大一点的batch size
    # 进行多次前向传播，由于dropout的随机性，输出应该不同
    output_train1 = ffn(x)
    output_train2 = ffn(x)
    # 验证在训练模式下两次输出不同（由于dropout的随机性）
    assert not torch.allclose(output_train1, output_train2)
    
    ffn.eval()  # 设置为评估模式
    output_eval1 = ffn(x)
    output_eval2 = ffn(x)
    # 验证在评估模式下两次输出相同（因为dropout被禁用）
    assert torch.allclose(output_eval1, output_eval2)
    
    # 6. 模型参数测试
    # 验证模型参数的形状
    assert ffn.fc1.weight.shape == (hidden_dim, input_dim)
    assert ffn.fc2.weight.shape == (input_dim, hidden_dim)
    assert ffn.fc1.bias.shape == (hidden_dim,)
    assert ffn.fc2.bias.shape == (input_dim,)

@allure.epic('llama算子正反向测试')
@allure.story('注意力机制测试')
@allure.title('LLaMA Attention测试')
def test_llama_attention(device):
    """测试 LLaMA 注意力机制的各种场景，包括：
    1. 基本功能测试
    2. RoPE位置编码测试
    3. KV缓存测试
    4. 注意力模式测试
    5. 批处理和多头测试
    """
    
    class LLaMAAttention(nn.Module):
        def __init__(self, hidden_size, num_heads, max_position_embeddings=2048, rope_theta=10000):
            super().__init__()
            self.hidden_size = hidden_size
            self.num_heads = num_heads
            self.head_dim = hidden_size // num_heads
            assert self.head_dim * num_heads == hidden_size, "hidden_size must be divisible by num_heads"
            
            self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
            self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
            self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
            self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
            
            # RoPE相关参数
            self.max_position_embeddings = max_position_embeddings
            self.rope_theta = rope_theta
            
            # 初始化RoPE位置编码
            inv_freq = 1.0 / (rope_theta ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
            self.register_buffer("inv_freq", inv_freq)
            
            # KV缓存
            self.kv_cache = {}
        
        def _rotate_half(self, x):
            x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
            return torch.cat((-x2, x1), dim=-1)
        
        def apply_rotary_embeddings(self, q, k, positions):
            # 计算RoPE的余弦和正弦值
            t = positions.float().unsqueeze(-1) * self.inv_freq
            freqs = torch.cat([t, t], dim=-1)
            emb = torch.cat([freqs.cos(), freqs.sin()], dim=-1)
            
            # 应用RoPE
            q_embed = (q * emb.cos()) + (self._rotate_half(q) * emb.sin())
            k_embed = (k * emb.cos()) + (self._rotate_half(k) * emb.sin())
            return q_embed, k_embed
        
        def forward(self, hidden_states, positions=None, use_cache=False, layer_id=None):
            batch_size, seq_length, _ = hidden_states.shape
            
            # 如果没有提供位置信息，则使用默认的顺序位置
            if positions is None:
                positions = torch.arange(seq_length, device=hidden_states.device)
            
            # 投影Q、K、V
            q = self.q_proj(hidden_states)
            k = self.k_proj(hidden_states)
            v = self.v_proj(hidden_states)
            
            # 重排形状为多头格式
            q = q.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
            k = k.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
            v = v.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
            
            # 应用RoPE
            q, k = self.apply_rotary_embeddings(q, k, positions)
            
            # 处理KV缓存
            if use_cache and layer_id is not None:
                if layer_id in self.kv_cache:
                    cached_k, cached_v = self.kv_cache[layer_id]
                    k = torch.cat([cached_k, k], dim=2)
                    v = torch.cat([cached_v, v], dim=2)
                self.kv_cache[layer_id] = (k, v)
            
            # 计算注意力分数
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            attn_weights = F.softmax(attn_weights, dim=-1)
            
            # 计算输出
            attn_output = torch.matmul(attn_weights, v)
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.view(batch_size, seq_length, self.hidden_size)
            
            # 输出投影
            return self.o_proj(attn_output)
    
    # 1. 基本功能测试
    hidden_size = 64
    num_heads = 4
    batch_size = 2
    seq_length = 8
    
    attention = LLaMAAttention(hidden_size, num_heads).to(device)
    hidden_states = torch.randn(batch_size, seq_length, hidden_size, device=device)
    output = attention(hidden_states)
    
    # 验证输出形状
    assert output.shape == (batch_size, seq_length, hidden_size)
    
    # 2. RoPE位置编码测试
    # 测试不同位置的编码
    positions = torch.tensor([0, 1, 4, 8, 16, 32, 64, 128], device=device)
    output_with_pos = attention(hidden_states, positions)
    # 验证位置编码影响了输出
    assert not torch.allclose(output, output_with_pos)
    
    # 3. KV缓存测试
    # 测试启用缓存的情况
    layer_id = 0
    output1 = attention(hidden_states, use_cache=True, layer_id=layer_id)
    # 使用新的输入，应该会使用之前的缓存
    new_input = torch.randn(batch_size, 2, hidden_size, device=device)
    output2 = attention(new_input, use_cache=True, layer_id=layer_id)
    assert output2.shape == (batch_size, 2, hidden_size)
    
    # 4. 注意力模式测试
    # 创建一个简单的注意力模式
    simple_input = torch.ones(1, 3, hidden_size, device=device)
    simple_output = attention(simple_input)
    # 验证所有头的输出都被正确处理
    assert simple_output.shape == (1, 3, hidden_size)
    
    # 5. 批处理和多头测试
    # 测试不同的批大小
    batch_sizes = [1, 4, 8]
    for bs in batch_sizes:
        test_input = torch.randn(bs, seq_length, hidden_size, device=device)
        test_output = attention(test_input)
        assert test_output.shape == (bs, seq_length, hidden_size)
    
    # 测试不同的头数
    head_sizes = [2, 4, 8]
    for num_head in head_sizes:
        if hidden_size % num_head == 0:
            test_attention = LLaMAAttention(hidden_size, num_head).to(device)
            test_output = test_attention(hidden_states)
            assert test_output.shape == (batch_size, seq_length, hidden_size)

@allure.epic('llama算子正反向测试')
@allure.story('注意力机制测试')
@allure.title('LLaMA Attention测试')
def test_llama_attention(device):
    """测试 LLaMA 注意力机制的各种场景，包括：
    1. 基本功能测试
    2. RoPE位置编码测试
    3. KV缓存测试
    4. 注意力模式测试
    5. 批处理和多头测试
    """
    
    class LLaMAAttention(nn.Module):
        def __init__(self, hidden_size, num_heads, max_position_embeddings=2048, rope_theta=10000):
            super().__init__()
            self.hidden_size = hidden_size
            self.num_heads = num_heads
            self.head_dim = hidden_size // num_heads
            assert self.head_dim * num_heads == hidden_size, "hidden_size must be divisible by num_heads"
            
            self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
            self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
            self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
            self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
            
            # RoPE相关参数
            self.max_position_embeddings = max_position_embeddings
            self.rope_theta = rope_theta
            
            # 初始化RoPE位置编码
            inv_freq = 1.0 / (rope_theta ** (torch.arange(0, self.head_dim // 2).float() / (self.head_dim // 2)))
            self.register_buffer("inv_freq", inv_freq)
            
            # KV缓存
            self.kv_cache = {}
        
        def _rotate_half(self, x):
            """将输入张量的后半部分旋转"""
            x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
            return torch.cat((-x2, x1), dim=-1)
        
        def apply_rotary_embeddings(self, q, k, positions):
            """应用RoPE位置编码"""
            # 将位置信息转换为适当的形状
            positions = positions.unsqueeze(1).unsqueeze(1)  # [seq_len, 1, 1]
            
            # 计算时间混合
            t = positions.float() * self.inv_freq  # [seq_len, 1, dim//2]
            
            # 计算余弦和正弦
            freqs_cos = torch.cos(t)  # [seq_len, 1, dim//2]
            freqs_sin = torch.sin(t)  # [seq_len, 1, dim//2]
            
            # 将余弦和正弦扩展到完整的维度
            freqs_cos = torch.cat([freqs_cos, freqs_cos], dim=-1)  # [seq_len, 1, dim]
            freqs_sin = torch.cat([freqs_sin, freqs_sin], dim=-1)  # [seq_len, 1, dim]
            
            # 将形状调整为与输入匹配
            freqs_cos = freqs_cos.unsqueeze(0)  # [1, seq_len, 1, dim]
            freqs_sin = freqs_sin.unsqueeze(0)  # [1, seq_len, 1, dim]
            
            # 应用旋转
            q_embed = (q * freqs_cos) + (self._rotate_half(q) * freqs_sin)
            k_embed = (k * freqs_cos) + (self._rotate_half(k) * freqs_sin)
            
            return q_embed, k_embed
        
        def forward(self, hidden_states, positions=None, use_cache=False, layer_id=None):
            batch_size, seq_length, _ = hidden_states.shape
            
            # 如果没有提供位置信息，则使用默认的顺序位置
            if positions is None:
                positions = torch.arange(seq_length, device=hidden_states.device)
            
            # 投影Q、K、V
            q = self.q_proj(hidden_states)
            k = self.k_proj(hidden_states)
            v = self.v_proj(hidden_states)
            
            # 重排形状为多头格式
            q = q.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
            k = k.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
            v = v.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
            
            # 应用RoPE
            q, k = self.apply_rotary_embeddings(q, k, positions)
            
            # 处理KV缓存
            if use_cache and layer_id is not None:
                if layer_id in self.kv_cache:
                    cached_k, cached_v = self.kv_cache[layer_id]
                    k = torch.cat([cached_k, k], dim=2)
                    v = torch.cat([cached_v, v], dim=2)
                self.kv_cache[layer_id] = (k, v)
            
            # 计算注意力分数
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            attn_weights = F.softmax(attn_weights, dim=-1)
            
            # 计算输出
            attn_output = torch.matmul(attn_weights, v)
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.view(batch_size, seq_length, self.hidden_size)
            
            # 输出投影
            return self.o_proj(attn_output)
    
    # 1. 基本功能测试
    hidden_size = 64
    num_heads = 4
    batch_size = 2
    seq_length = 8
    
    attention = LLaMAAttention(hidden_size, num_heads).to(device)
    hidden_states = torch.randn(batch_size, seq_length, hidden_size, device=device)
    output = attention(hidden_states)
    
    # 验证输出形状
    assert output.shape == (batch_size, seq_length, hidden_size)
    
    # 2. RoPE位置编码测试
    # 测试不同位置的编码
    positions = torch.tensor([0, 1, 4, 8, 16, 32, 64, 128], device=device)
    output_with_pos = attention(hidden_states, positions)
    # 验证位置编码影响了输出
    assert not torch.allclose(output, output_with_pos)
    
    # 3. KV缓存测试
    # 测试启用缓存的情况
    layer_id = 0
    output1 = attention(hidden_states, use_cache=True, layer_id=layer_id)
    # 使用新的输入，应该会使用之前的缓存
    new_input = torch.randn(batch_size, 2, hidden_size, device=device)
    output2 = attention(new_input, use_cache=True, layer_id=layer_id)
    assert output2.shape == (batch_size, 2, hidden_size)
    
    # 4. 注意力模式测试
    # 创建一个简单的注意力模式
    simple_input = torch.ones(1, 3, hidden_size, device=device)
    simple_output = attention(simple_input)
    # 验证所有头的输出都被正确处理
    assert simple_output.shape == (1, 3, hidden_size)
    
    # 5. 批处理和多头测试
    # 测试不同的批大小
    batch_sizes = [1, 4, 8]
    for bs in batch_sizes:
        test_input = torch.randn(bs, seq_length, hidden_size, device=device)
        test_output = attention(test_input)
        assert test_output.shape == (bs, seq_length, hidden_size)
    
    # 测试不同的头数
    head_sizes = [2, 4, 8]
    for num_head in head_sizes:
        if hidden_size % num_head == 0:
            test_attention = LLaMAAttention(hidden_size, num_head).to(device)
            test_output = test_attention(hidden_states)
            assert test_output.shape == (batch_size, seq_length, hidden_size)

@allure.epic('llama算子正反向测试')
@allure.story('注意力机制测试')
@allure.title('LLaMA Attention测试')
def test_llama_attention(device):
    """测试 LLaMA 注意力机制的各种场景，包括：
    1. 基本功能测试
    2. RoPE位置编码测试
    3. KV缓存测试
    4. 注意力模式测试
    5. 批处理和多头测试
    """
    
    class LLaMAAttention(nn.Module):
        def __init__(self, hidden_size, num_heads, max_position_embeddings=2048, rope_theta=10000):
            super().__init__()
            self.hidden_size = hidden_size
            self.num_heads = num_heads
            self.head_dim = hidden_size // num_heads
            assert self.head_dim * num_heads == hidden_size, "hidden_size must be divisible by num_heads"
            
            self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
            self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
            self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
            self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
            
            # RoPE相关参数
            self.max_position_embeddings = max_position_embeddings
            self.rope_theta = rope_theta
            
            # 初始化RoPE位置编码
            inv_freq = 1.0 / (rope_theta ** (torch.arange(0, self.head_dim // 2).float() / (self.head_dim // 2)))
            self.register_buffer("inv_freq", inv_freq)
            
            # KV缓存
            self.kv_cache = {}
        
        def _rotate_half(self, x):
            """将输入张量的后半部分旋转"""
            x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
            return torch.cat((-x2, x1), dim=-1)
        
        def apply_rotary_embeddings(self, q, k, positions):
            """应用RoPE位置编码
            输入:
                q: [batch_size, num_heads, seq_len, head_dim]
                k: [batch_size, num_heads, seq_len, head_dim]
                positions: [seq_len]
            """
            seq_length = positions.shape[0]
            
            # 计算时间混合
            t = positions.float().unsqueeze(-1) * self.inv_freq  # [seq_len, dim//2]
            
            # 计算余弦和正弦
            freqs_cos = torch.cos(t)  # [seq_len, dim//2]
            freqs_sin = torch.sin(t)  # [seq_len, dim//2]
            
            # 将余弦和正弦扩展到完整的维度
            freqs_cos = torch.cat([freqs_cos, freqs_cos], dim=-1)  # [seq_len, dim]
            freqs_sin = torch.cat([freqs_sin, freqs_sin], dim=-1)  # [seq_len, dim]
            
            # 将形状调整为与输入匹配
            freqs_cos = freqs_cos.view(seq_length, self.head_dim).unsqueeze(0).unsqueeze(0)
            freqs_sin = freqs_sin.view(seq_length, self.head_dim).unsqueeze(0).unsqueeze(0)
            
            # 应用旋转
            q_embed = (q * freqs_cos) + (self._rotate_half(q) * freqs_sin)
            k_embed = (k * freqs_cos) + (self._rotate_half(k) * freqs_sin)
            
            return q_embed, k_embed
        
        def forward(self, hidden_states, positions=None, use_cache=False, layer_id=None):
            batch_size, seq_length, _ = hidden_states.shape
            
            # 如果没有提供位置信息，则使用默认的顺序位置
            if positions is None:
                positions = torch.arange(seq_length, device=hidden_states.device)
            
            # 投影Q、K、V
            q = self.q_proj(hidden_states)
            k = self.k_proj(hidden_states)
            v = self.v_proj(hidden_states)
            
            # 重排形状为多头格式
            q = q.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
            k = k.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
            v = v.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
            
            # 应用RoPE
            q, k = self.apply_rotary_embeddings(q, k, positions)
            
            # 处理KV缓存
            if use_cache and layer_id is not None:
                if layer_id in self.kv_cache:
                    cached_k, cached_v = self.kv_cache[layer_id]
                    k = torch.cat([cached_k, k], dim=2)
                    v = torch.cat([cached_v, v], dim=2)
                self.kv_cache[layer_id] = (k, v)
            
            # 计算注意力分数
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            attn_weights = F.softmax(attn_weights, dim=-1)
            
            # 计算输出
            attn_output = torch.matmul(attn_weights, v)
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.view(batch_size, seq_length, self.hidden_size)
            
            # 输出投影
            return self.o_proj(attn_output)
    
    # 1. 基本功能测试
    hidden_size = 64
    num_heads = 4
    batch_size = 2
    seq_length = 8
    
    attention = LLaMAAttention(hidden_size, num_heads).to(device)
    hidden_states = torch.randn(batch_size, seq_length, hidden_size, device=device)
    output = attention(hidden_states)
    
    # 验证输出形状
    assert output.shape == (batch_size, seq_length, hidden_size)
    
    # 2. RoPE位置编码测试
    # 测试不同位置的编码
    positions = torch.tensor([0, 1, 4, 8, 16, 32, 64, 128], device=device)
    output_with_pos = attention(hidden_states, positions)
    # 验证位置编码影响了输出
    assert not torch.allclose(output, output_with_pos)
    
    # 3. KV缓存测试
    # 测试启用缓存的情况
    layer_id = 0
    output1 = attention(hidden_states, use_cache=True, layer_id=layer_id)
    # 使用新的输入，应该会使用之前的缓存
    new_input = torch.randn(batch_size, 2, hidden_size, device=device)
    output2 = attention(new_input, use_cache=True, layer_id=layer_id)
    assert output2.shape == (batch_size, 2, hidden_size)
    
    # 4. 注意力模式测试
    # 创建一个简单的注意力模式
    simple_input = torch.ones(1, 3, hidden_size, device=device)
    simple_output = attention(simple_input)
    # 验证所有头的输出都被正确处理
    assert simple_output.shape == (1, 3, hidden_size)
    
    # 5. 批处理和多头测试
    # 测试不同的批大小
    batch_sizes = [1, 4, 8]
    for bs in batch_sizes:
        test_input = torch.randn(bs, seq_length, hidden_size, device=device)
        test_output = attention(test_input)
        assert test_output.shape == (bs, seq_length, hidden_size)
    
    # 测试不同的头数
    head_sizes = [2, 4, 8]
    for num_head in head_sizes:
        if hidden_size % num_head == 0:
            test_attention = LLaMAAttention(hidden_size, num_head).to(device)
            test_output = test_attention(hidden_states)
            assert test_output.shape == (batch_size, seq_length, hidden_size)

@allure.epic('基础算子测试')
@allure.story('repeat算子测试')
@allure.title('repeat算子测试')
def test_repeat_operator(device):
    """
    测试PyTorch的repeat算子在不同场景下的表现，包括：
    1. 基本功能测试
    2. 多维张量测试
    3. 零维和标量测试
    4. 边界情况测试
    5. 内存连续性测试
    """
    
    # 1. 基本功能测试
    # 测试1D张量
    x = torch.tensor([1, 2, 3], device=device)
    repeated = x.repeat(2)
    assert torch.equal(repeated, torch.tensor([1, 2, 3, 1, 2, 3], device=device))
    assert repeated.shape == (6,)
    
    # 测试2D张量
    x = torch.tensor([[1, 2], [3, 4]], device=device)
    repeated = x.repeat(2, 3)
    expected = torch.tensor([[1, 2, 1, 2, 1, 2],
                           [3, 4, 3, 4, 3, 4],
                           [1, 2, 1, 2, 1, 2],
                           [3, 4, 3, 4, 3, 4]], device=device)
    assert torch.equal(repeated, expected)
    assert repeated.shape == (4, 6)
    
    # 2. 多维张量测试
    x = torch.tensor([[[1, 2], [3, 4]]], device=device)  # 3D张量 [1, 2, 2]
    repeated = x.repeat(2, 1, 2)
    assert repeated.shape == (2, 2, 4)
    
    # 3. 零维和标量测试
    # 测试标量
    x = torch.tensor(5, device=device)
    repeated = x.repeat(3)
    assert torch.equal(repeated, torch.tensor([5, 5, 5], device=device))
    
    # 测试零维张量
    x = torch.zeros(0, device=device)
    repeated = x.repeat(2)
    assert repeated.shape == (0,)
    
    # 4. 边界情况测试
    # 测试重复次数为1
    x = torch.tensor([1, 2, 3], device=device)
    repeated = x.repeat(1)
    assert torch.equal(repeated, x)
    
    # 测试重复次数为0
    repeated = x.repeat(0)
    assert repeated.shape == (0,)
    
    # 5. 内存连续性测试
    x = torch.tensor([[1, 2], [3, 4]], device=device)
    # 创建非连续张量
    x_transpose = x.t()
    assert not x_transpose.is_contiguous()
    
    # 测试非连续张量的repeat操作
    repeated = x_transpose.repeat(2, 1)
    assert repeated.is_contiguous()  # repeat操作应该返回连续的张量
    assert repeated.shape == (4, 2)
    
    # 测试带有复杂重复模式的情况
    x = torch.tensor([1, 2], device=device)
    repeated = x.repeat(2, 3, 2)
    assert repeated.shape == (2, 3, 4)
    
    # 测试不同数据类型
    x = torch.tensor([1.5, 2.5], dtype=torch.float32, device=device)
    repeated = x.repeat(2)
    assert repeated.dtype == torch.float32
    
    # 测试大规模重复
    x = torch.tensor([1], device=device)
    repeated = x.repeat(1000)
    assert repeated.shape == (1000,)
    assert torch.all(repeated == 1)

@allure.epic('基础算子测试')
@allure.story('transpose算子测试')
@allure.title('transpose算子测试')
def test_transpose_operator(device):
    """
    测试PyTorch的transpose算子在不同场景下的表现，包括：
    1. 基本功能测试
    2. 多维张量测试
    3. 内存布局测试
    4. 边界情况测试
    5. 性能相关测试
    """
    
    # 1. 基本功能测试
    # 测试2D张量
    x = torch.tensor([[1, 2, 3],
                     [4, 5, 6]], device=device)
    transposed = x.transpose(0, 1)
    expected = torch.tensor([[1, 4],
                           [2, 5],
                           [3, 6]], device=device)
    assert torch.equal(transposed, expected)
    assert transposed.shape == (3, 2)
    
    # 测试连续转置
    x = torch.tensor([[1, 2], [3, 4]], device=device)
    double_transposed = x.transpose(0, 1).transpose(0, 1)
    assert torch.equal(double_transposed, x)
    
    # 2. 多维张量测试
    # 测试3D张量
    x = torch.arange(24, device=device).reshape(2, 3, 4)
    transposed = x.transpose(0, 2)
    assert transposed.shape == (4, 3, 2)
    
    # 测试4D张量
    x = torch.arange(120, device=device).reshape(2, 3, 4, 5)
    transposed = x.transpose(1, 3)
    assert transposed.shape == (2, 5, 4, 3)
    
    # 3. 内存布局测试
    # 测试内存连续性
    x = torch.tensor([[1, 2], [3, 4]], device=device)
    transposed = x.transpose(0, 1)
    assert not transposed.is_contiguous()
    
    # 测试contiguous()操作
    contiguous = transposed.contiguous()
    assert contiguous.is_contiguous()
    assert torch.equal(contiguous, transposed)
    
    # 4. 边界情况测试
    # 测试相同维度的转置
    x = torch.tensor([[1, 2], [3, 4]], device=device)
    same_dim_transposed = x.transpose(1, 1)
    assert torch.equal(same_dim_transposed, x)
    
    # 测试特殊形状
    x = torch.tensor([[[]]], device=device)  # 空3D张量
    transposed = x.transpose(0, 2)
    assert transposed.shape == (0, 1, 1)
    
    # 5. 性能相关测试
    # 测试大张量
    large_tensor = torch.randn(100, 200, device=device)
    transposed = large_tensor.transpose(0, 1)
    assert transposed.shape == (200, 100)
    
    # 测试不同数据类型
    dtypes = [torch.float32, torch.float64, torch.int32, torch.int64]
    for dtype in dtypes:
        x = torch.tensor([[1, 2], [3, 4]], dtype=dtype, device=device)
        transposed = x.transpose(0, 1)
        assert transposed.dtype == dtype
    
    # 测试复杂的转置序列
    x = torch.randn(2, 3, 4, 5, device=device)
    # 多次转置
    result = x.transpose(0, 1).transpose(2, 3).transpose(1, 2)
    assert result.shape == (3, 5, 2, 4)  # 正确的转置后形状
    
    # 测试负数索引
    x = torch.tensor([[1, 2], [3, 4]], device=device)
    transposed = x.transpose(-2, -1)
    assert torch.equal(transposed, x.transpose(0, 1))
    
    # 测试视图操作
    x = torch.randn(10, 20, device=device)
    transposed = x.transpose(0, 1)
    view1 = x.view(-1)
    view2 = transposed.contiguous().view(-1)
    assert torch.equal(torch.sort(view1.float())[0], torch.sort(view2.float())[0])

@allure.epic('基础算子测试')
@allure.story('matmul算子测试')
@allure.title('matmul算子测试')
def test_matmul_operator(device):
    """
    测试PyTorch的matmul算子在不同场景下的表现，包括：
    1. 基本矩阵乘法测试
    2. 广播机制测试
    3. 批处理测试
    4. 特殊情况测试
    5. 数值精度测试
    """
    
    # 1. 基本矩阵乘法测试
    # 测试2D矩阵乘法
    a = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32, device=device)
    b = torch.tensor([[5, 6], [7, 8]], dtype=torch.float32, device=device)
    c = torch.matmul(a, b)
    expected = torch.tensor([[19, 22], [43, 50]], dtype=torch.float32, device=device)
    assert torch.allclose(c, expected)
    
    # 测试向量点积
    v1 = torch.tensor([1, 2, 3], dtype=torch.float32, device=device)
    v2 = torch.tensor([4, 5, 6], dtype=torch.float32, device=device)
    dot_product = torch.matmul(v1, v2)
    assert torch.allclose(dot_product, torch.tensor(32.0, device=device))
    
    # 2. 广播机制测试
    # 测试广播矩阵乘法
    a = torch.randn(3, 4, 5, device=device)
    b = torch.randn(5, 2, device=device)
    c = torch.matmul(a, b)
    assert c.shape == (3, 4, 2)
    
    # 测试复杂的广播
    a = torch.randn(2, 3, 4, 5, device=device)
    b = torch.randn(2, 1, 5, 6, device=device)
    c = torch.matmul(a, b)
    assert c.shape == (2, 3, 4, 6)
    
    # 3. 批处理测试
    # 测试批量矩阵乘法
    batch_size = 10
    a = torch.randn(batch_size, 3, 4, device=device)
    b = torch.randn(batch_size, 4, 5, device=device)
    c = torch.matmul(a, b)
    assert c.shape == (batch_size, 3, 5)
    
    # 测试不同批大小的广播
    a = torch.randn(10, 1, 3, 4, device=device)
    b = torch.randn(1, 5, 4, 2, device=device)
    c = torch.matmul(a, b)
    assert c.shape == (10, 5, 3, 2)
    
    # 4. 特殊情况测试
    # 测试空维度
    a = torch.randn(0, 2, 3, device=device)
    b = torch.randn(3, 4, device=device)
    c = torch.matmul(a, b)
    assert c.shape == (0, 2, 4)
    
    # 测试1x1矩阵
    a = torch.tensor([[2.0]], device=device)
    b = torch.tensor([[3.0]], device=device)
    c = torch.matmul(a, b)
    assert torch.allclose(c, torch.tensor([[6.0]], device=device))
    
    # 5. 数值精度测试
    # 测试不同数据类型
    dtypes = [torch.float32, torch.float64]
    for dtype in dtypes:
        a = torch.tensor([[1, 2], [3, 4]], dtype=dtype, device=device)
        b = torch.tensor([[5, 6], [7, 8]], dtype=dtype, device=device)
        c = torch.matmul(a, b)
        assert c.dtype == dtype
    
    # 测试大数值
    a = torch.tensor([[1e10, 1e-10], [1e-10, 1e10]], device=device)
    b = torch.tensor([[1e10, 1e-10], [1e-10, 1e10]], device=device)
    c = torch.matmul(a, b)
    assert not torch.any(torch.isinf(c))
    
    # 测试数值稳定性
    a = torch.randn(10, 10, device=device) * 1e-5
    b = torch.randn(10, 10, device=device) * 1e5
    c = torch.matmul(a, b)
    assert not torch.any(torch.isnan(c))
    assert not torch.any(torch.isinf(c))
    
    # 测试矩阵乘法的结合律
    a = torch.randn(2, 3, device=device)
    b = torch.randn(3, 4, device=device)
    c = torch.randn(4, 5, device=device)
    
    # (AB)C = A(BC)
    left = torch.matmul(torch.matmul(a, b), c)
    right = torch.matmul(a, torch.matmul(b, c))
    assert torch.allclose(left, right, rtol=1e-4)

@allure.epic('基础算子测试')
@allure.story('dropout算子测试')
@allure.title('dropout算子测试')
def test_dropout_operator(device):
    """
    测试PyTorch的dropout算子在不同场景下的表现，包括：
    1. 基本功能测试
    2. 训练和评估模式测试
    3. 不同概率和形状测试
    4. 边界情况测试
    5. 随机性和可重复性测试
    """
    
    # 1. 基本功能测试
    dropout = nn.Dropout(p=0.5)
    x = torch.ones(1000, device=device)
    
    # 训练模式下的dropout
    dropout.train()
    y_train = dropout(x)
    # 验证输出中的非零元素应该是输入的2倍
    non_zero_elements = y_train[y_train > 0]
    assert torch.allclose(non_zero_elements, torch.tensor(2.0, device=device))
    
    # 验证dropout比例
    zero_ratio = (y_train == 0).float().mean()
    assert 0.4 <= zero_ratio <= 0.6  # 允许一定的随机性
    
    # 2. 训练和评估模式测试
    # 评估模式
    dropout.eval()
    y_eval = dropout(x)
    assert torch.equal(y_eval, x)  # 评估模式下应该不发生任何改变
    
    # 测试模式切换
    dropout.train()
    y_train_again = dropout(x)
    assert not torch.equal(y_train_again, x)  # 训练模式下应该发生改变
    
    # 3. 不同概率和形状测试
    # 测试不同的dropout概率
    probabilities = [0.0, 0.3, 0.7, 1.0]
    for p in probabilities:
        dropout = nn.Dropout(p=p)
        dropout.train()
        y = dropout(x)
        zero_ratio = (y == 0).float().mean()
        assert p - 0.1 <= zero_ratio <= p + 0.1  # 允许一定的误差
    
    # 测试不同的输入形状
    shapes = [(10,), (10, 20), (10, 20, 30), (10, 20, 30, 40)]
    dropout = nn.Dropout(p=0.5)
    dropout.train()
    for shape in shapes:
        x = torch.ones(shape, device=device)
        y = dropout(x)
        assert y.shape == shape
        zero_ratio = (y == 0).float().mean()
        assert 0.3 <= zero_ratio <= 0.7  # 增加容错范围，考虑到小批量数据的随机性
    
    # 4. 边界情况测试
    # 测试空张量
    x_empty = torch.ones(0, device=device)
    y_empty = dropout(x_empty)
    assert y_empty.shape == (0,)
    
    # 测试单个元素
    x_single = torch.ones(1, device=device)
    y_single = dropout(x_single)
    assert y_single.shape == (1,)
    
    # 测试负值输入
    x_neg = -torch.ones(1000, device=device)
    y_neg = dropout(x_neg)
    non_zero_elements = y_neg[y_neg < 0]
    assert torch.allclose(non_zero_elements, torch.tensor(-2.0, device=device))
    
    # 5. 随机性和可重复性测试
    # 测试随机种子
    torch.manual_seed(42)
    dropout = nn.Dropout(p=0.5)
    dropout.train()
    y1 = dropout(x)
    
    torch.manual_seed(42)
    y2 = dropout(x)
    assert torch.equal(y1, y2)  # 相同的随机种子应该产生相同的结果
    
    # 测试不同的随机种子
    torch.manual_seed(43)
    y3 = dropout(x)
    assert not torch.equal(y1, y3)  # 不同的随机种子应该产生不同的结果
    
    # 测试多次调用的独立性
    y4 = dropout(x)
    y5 = dropout(x)
    assert not torch.equal(y4, y5)  # 每次调用应该产生不同的结果
    
    # 测试批处理中的独立性
    batch = torch.ones(10, 1000, device=device)
    y_batch = dropout(batch)
    # 验证每个样本都有不同的dropout模式
    for i in range(9):
        assert not torch.equal(y_batch[i], y_batch[i+1])

if __name__ == "__main__":
    pytest.main([__file__])
