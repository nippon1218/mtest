import torch
import pytest
import allure
import numpy as np
from utils.device_utils import get_device_object, get_device_info

# 支持的数据类型
test_dtypes = [
    torch.float32,
    torch.bfloat16
]

# 测试的填充值
test_values = [
    0.0,        # 零
    1.0,        # 正数
    -1.0,       # 负数
    3.14,       # 小数
    1e3,        # 较大的数
    -1e3,       # 较小的数
]

@allure.epic("PyTorch算子测试")
@allure.feature("Fill_算子")
@allure.description("""
该测试模块验证PyTorch中fill_算子的功能正确性，包括：
1. 基本填充功能：验证不同数据类型的填充
2. 边界值测试：验证各种特殊值的填充
3. 大张量测试：验证大规模数据的填充
""")
class TestFill:
    def teardown_method(self, method):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    @allure.story("基础功能测试")
    @allure.title("测试基本填充功能 - {dtype}")
    @allure.description("""
    验证基本的填充功能，测试要点：
    1. 验证不同数据类型的填充
    2. 验证不同填充值
    3. 验证填充后的值是否正确
    """)
    @pytest.mark.parametrize("dtype", test_dtypes)
    @pytest.mark.parametrize("value", test_values)
    def test_fill_basic(self, device, dtype, value):
        # 准备测试数据
        x = torch.randn(2, 3).to(dtype)
        if device == "cuda":
            x = x.cuda()
        
        # 执行fill_操作
        x.fill_(value)
        
        # 验证填充结果
        expected_value = torch.full_like(x, value)

        if device == "cuda":
            expected_value = expected_value.cuda()
            
        # 考虑到bfloat16的精度限制，使用适当的误差范围
        rtol = 1e-2 if dtype == torch.bfloat16 else 1e-5
        atol = 1e-2 if dtype == torch.bfloat16 else 1e-5
        
        assert torch.allclose(x, expected_value, rtol=rtol, atol=atol), \
            f"填充值{value}的结果与预期不符"

    @allure.story("边界测试 - 大张量")
    @allure.title("测试大张量的填充功能 - {dtype}")
    @allure.description("""
    验证大张量的填充功能，测试要点：
    1. 验证不同维度的大张量填充
    2. 验证填充的一致性
    3. 验证内存使用情况
    """)
    @pytest.mark.parametrize("dtype", test_dtypes)
    def test_fill_large_tensor(self, device, dtype):
        # 测试不同维度的大张量
        shapes = [
            (1024, 1024),          # 2D大矩阵
            (32, 64, 128),         # 3D张量
            (16, 32, 64, 32)       # 4D张量
        ]
        
        for shape in shapes:
            # 准备大张量数据
            x = torch.randn(*shape).to(dtype)
            if device == "cuda":
                x = x.cuda()
            
            # 使用不同的填充值进行测试
            for value in [0.0, 1.0, -1.0]:
                # 执行fill_操作
                x.fill_(value)
                
                # 验证填充结果
                expected_value = torch.full_like(x, value)
                #if device == "cuda":
                #    expected_value = expected_value.cuda()
                
                # 考虑到bfloat16的精度限制
                rtol = 1e-2 if dtype == torch.bfloat16 else 1e-5
                atol = 1e-2 if dtype == torch.bfloat16 else 1e-5
                
                assert torch.allclose(x, expected_value, rtol=rtol, atol=atol), \
                    f"大张量{shape}使用{value}填充的结果与预期不符"
            
            # 清理内存
            del x
            if device == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

    @allure.story("边界测试 - 特殊形状")
    @allure.title("测试特殊形状张量的填充功能 - {dtype}")
    @allure.description("""
    验证特殊形状张量的填充功能，测试要点：
    1. 验证空张量的填充
    2. 验证标量张量的填充
    3. 验证单元素张量的填充
    """)
    @pytest.mark.parametrize("dtype", test_dtypes)
    def test_fill_special_shapes(self, device, dtype):
        special_shapes = [
            (),             # 标量
            (1,),          # 一维单元素
            (1, 1),        # 二维单元素
            (0,),          # 空张量
            (0, 2),        # 空张量（第一维为0）
            (2, 0),        # 空张量（第二维为0）
        ]
        
        for shape in special_shapes:
            # 准备测试数据
            if shape == ():
                # 对于标量张量，使用tensor创建
                x = torch.tensor(0.0, dtype=dtype)
            else:
                # 对于其他形状，使用zeros创建
                x = torch.zeros(shape, dtype=dtype)
                
            if device == "cuda":
                x = x.cuda()
            
            # 使用1.0作为填充值进行测试
            value = 1.0
            x.fill_(value)
            
            # 验证填充结果
            expected_value = torch.full_like(x, value)
            if device == "cuda":
                expected_value = expected_value.cuda()
            
            # 考虑到bfloat16的精度限制
            rtol = 1e-2 if dtype == torch.bfloat16 else 1e-5
            atol = 1e-2 if dtype == torch.bfloat16 else 1e-5
            
            assert torch.allclose(x, expected_value, rtol=rtol, atol=atol), \
                f"特殊形状{shape}的填充结果与预期不符"

    @pytest.mark.gpu
    @pytest.mark.torch
    def test_fill_gpu_tensor(self):
        # 准备测试数据
        x = torch.randn(2, 3).to("cuda")
        
        # 执行fill_操作
        x.fill_(1.0)
        
        # 验证填充结果
        expected_value = torch.full_like(x, 1.0)
        
        # 考虑到bfloat16的精度限制
        rtol = 1e-5
        atol = 1e-5
        
        assert torch.allclose(x, expected_value, rtol=rtol, atol=atol), \
            f"GPU上填充值1.0的结果与预期不符"
