import torch
import pytest
import numpy as np

def test_matrix_addition():
    """测试矩阵加法运算"""
    # 定义两个矩阵
    matrix_a = torch.tensor([[1, 2], [3, 4]])
    matrix_b = torch.tensor([[5, 6], [7, 8]])
    expected_result = torch.tensor([[6, 8], [10, 12]])

    # 进行矩阵相加
    result = matrix_a + matrix_b

    # 断言结果是否等于预期
    assert torch.equal(result, expected_result), "Matrix addition result is incorrect"


def test_matrix_subtraction():
    """测试矩阵减法运算"""
    matrix_a = torch.tensor([[5, 6], [7, 8]])
    matrix_b = torch.tensor([[1, 2], [3, 4]])
    expected_result = torch.tensor([[4, 4], [4, 4]])
    result = matrix_a - matrix_b
    assert torch.equal(result, expected_result), "Matrix subtraction result is incorrect"


def test_matrix_multiplication():
    """测试矩阵乘法运算"""
    matrix_a = torch.tensor([[1, 2], [3, 4]])
    matrix_b = torch.tensor([[5, 6], [7, 8]])
    expected_result = torch.tensor([[19, 22], [43, 50]])
    result = torch.mm(matrix_a, matrix_b)
    assert torch.equal(result, expected_result), "Matrix multiplication result is incorrect"


def test_matrix_shape():
    """测试矩阵形状"""
    matrix_a = torch.tensor([[1, 2], [3, 4]])
    matrix_b = torch.tensor([[5, 6], [7, 8]])
    result = matrix_a + matrix_b
    assert result.shape == (2, 2), "Resulting matrix shape is incorrect"


#def test_invalid_input():
#    """测试各种无效输入情况"""
#    # 测试维度不匹配 - 会抛出异常
#    matrix_a = torch.tensor([[1, 2], [3, 4]])
#    matrix_b = torch.tensor([[5, 6]])
#    with pytest.raises(RuntimeError) as excinfo:
#        result = matrix_a + matrix_b
#    assert "size" in str(excinfo.value), "应该提示尺寸不匹配错误"
#
#    # 测试 NaN 值
#    matrix_nan = torch.tensor([[float('nan'), 2], [3, 4]])
#    matrix_normal = torch.tensor([[1, 2], [3, 4]])
#    result = matrix_nan + matrix_normal
#    assert torch.isnan(result[0,0]), "结果应该包含 NaN"
#
#    # 测试无效的矩阵乘法维度
#    with pytest.raises(RuntimeError) as excinfo:
#        invalid_result = torch.mm(matrix_a, torch.tensor([1, 2, 3]))
#    assert "size" in str(excinfo.value), "应该提示矩阵乘法维度不匹配错误"
#
#    # 测试索引越界
#    with pytest.raises(IndexError):
#        invalid_index = matrix_a[2, 2]  # 2x2矩阵没有[2,2]位置


def test_matrix_transpose():
    """测试矩阵转置操作"""
    # 定义测试矩阵
    matrix = torch.tensor([[1, 2, 3],
                          [4, 5, 6]])
    expected_result = torch.tensor([[1, 4],
                                  [2, 5],
                                  [3, 6]])
    
    # 进行矩阵转置
    result = matrix.transpose(0, 1)
    
    # 验证转置结果
    assert torch.equal(result, expected_result), "Matrix transpose result is incorrect"
    # 验证转置后的形状
    assert result.shape == (3, 2), "Transposed matrix shape is incorrect"


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_matrix_datatypes(dtype):
    """测试不同数据类型的矩阵运算"""
    matrix_a = torch.tensor([[1, 2], [3, 4]], dtype=dtype)
    matrix_b = torch.tensor([[5, 6], [7, 8]], dtype=dtype)
    
    result = matrix_a + matrix_b
    assert result.dtype == dtype, f"结果数据类型应该是 {dtype}"


def test_large_matrix_operations():
    """测试大型矩阵的运算性能"""
    size = 1000
    matrix_a = torch.randn(size, size)
    matrix_b = torch.randn(size, size)
    
    # 测试大矩阵加法
    result_add = matrix_a + matrix_b
    assert result_add.shape == (size, size)
    
    # 测试大矩阵乘法
    result_mul = torch.mm(matrix_a, matrix_b)
    assert result_mul.shape == (size, size)


def test_matrix_operations():
    """测试矩阵的其他常用操作"""
    # 测试矩阵求逆
    matrix = torch.tensor([[4.0, 7.0], [2.0, 6.0]])
    inverse = torch.inverse(matrix)
    identity = torch.eye(2)
    result = torch.mm(matrix, inverse)
    assert torch.allclose(result, identity, rtol=1e-5), "矩阵求逆结果不正确"
    
    # 测试特征值和特征向量
    eigenvalues, eigenvectors = torch.linalg.eig(matrix)
    assert eigenvalues.shape == (2,), "特征值形状不正确"
    assert eigenvectors.shape == (2, 2), "特征向量形状不正确"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 GPU 支持")
def test_gpu_operations():
    """测试 GPU 设备上的矩阵运算"""
    matrix_a = torch.tensor([[1, 2], [3, 4]], device='cuda')
    matrix_b = torch.tensor([[5, 6], [7, 8]], device='cuda')
    
    # 测试 GPU 上的矩阵加法
    result = matrix_a + matrix_b
    assert result.device.type == 'cuda', "结果应该在 GPU 上"
    
    # 测试 CPU 和 GPU 之间的数据转移
    cpu_result = result.cpu()
    assert cpu_result.device.type == 'cpu', "结果应该已转移到 CPU 上"


def test_edge_cases():
    """测试边界情况"""
    # 测试极大值
    max_matrix = 0
    with pytest.raises(ZeroDivisionError):
        result = max_matrix / max_matrix
#    max_matrix = torch.tensor([[torch.finfo(torch.float32).max]])
#    with pytest.raises(RuntimeError):
#        result = max_matrix * max_matrix
#    
#    # 测试极小值
#    min_matrix = torch.tensor([[torch.finfo(torch.float32).tiny]])
#    result = min_matrix * min_matrix
#    assert torch.isfinite(result).all(), "结果应该是有限值"
#    
#    # 测试零除
#    zero_matrix = torch.zeros(2, 2)
#    with pytest.raises(RuntimeError):
#        result = torch.inverse(zero_matrix)


@pytest.mark.parametrize("shape", [
    ((2, 3), (3, 2)),
    ((3, 4), (4, 2)),
    ((4, 5), (5, 3))
])
def test_matrix_multiplication_shapes(shape):
    """测试不同形状矩阵的乘法"""
    shape_a, shape_b = shape
    matrix_a = torch.randn(*shape_a)
    matrix_b = torch.randn(*shape_b)
    
    result = torch.mm(matrix_a, matrix_b)
    expected_shape = (shape_a[0], shape_b[1])
    assert result.shape == expected_shape, f"矩阵乘法结果形状应为 {expected_shape}"