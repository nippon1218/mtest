import torch
import pytest
import allure
import numpy as np
from utils.device_utils import get_device_object, get_device_info

# 支持的数据类型组合
test_dtypes = [
    (torch.float32, torch.bfloat16),
    (torch.bfloat16, torch.float32)
]

@allure.epic("PyTorch算子测试")
@allure.feature("Copy_算子")
@allure.description("""
该测试模块验证PyTorch中copy_算子的功能正确性，包括：
1. Device to Device的拷贝
2. 不同数据类型之间的转换：
   - float32 -> bfloat16
   - bfloat16 -> float32
""")
class TestCopy:
    def teardown_method(self, method):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def _check_special_values(self, x, y, rtol=1e-2, atol=1e-2):
        """辅助函数：检查特殊值是否正确拷贝"""
        # 检查无穷值
        x_inf_mask = torch.isinf(x)
        y_inf_mask = torch.isinf(y)
        assert torch.all(x_inf_mask == y_inf_mask), "无穷值未正确拷贝"
        
        # 检查NaN值
        x_nan_mask = torch.isnan(x)
        y_nan_mask = torch.isnan(y)
        assert torch.all(x_nan_mask == y_nan_mask), "NaN值未正确拷贝"
        
        # 检查正常值
        x_normal = x[~(x_inf_mask | x_nan_mask)]
        y_normal = y[~(y_inf_mask | y_nan_mask)]
        if len(x_normal) > 0:
            assert torch.allclose(x_normal, y_normal, rtol=rtol, atol=atol), "正常值未正确拷贝"
    
    @allure.story("基础功能测试")
    @allure.title("测试Device to Device的数据拷贝和类型转换 - {src_dtype} -> {dst_dtype}")
    @allure.description("""
    验证Device to Device的数据拷贝和类型转换，测试要点：
    1. 验证数据拷贝的正确性
    2. 验证类型转换的准确性
    3. 保持张量的原有形状
    """)
    @pytest.mark.parametrize("src_dtype,dst_dtype", test_dtypes)
    def test_copy_device_to_device(self, device, src_dtype, dst_dtype):
        if not torch.cuda.is_available():
            pytest.skip("CUDA设备不可用，跳过测试")
            
        # 准备测试数据
        #x = torch.randn(2, 3).to(src_dtype).cuda()
        #y = torch.empty_like(x, dtype=dst_dtype).cuda()
        x = torch.randn(2, 3).to(src_dtype)
        y = torch.empty_like(x, dtype=dst_dtype)
        if device == "cuda":
            x =x.cuda()
            y =y.cuda()
        
        # 执行copy_操作
        y.copy_(x)
        
        # 验证形状保持不变
        assert x.shape == y.shape, f"拷贝后形状不一致: 输入{x.shape} vs 输出{y.shape}"
        
        # 将数据转换为相同类型进行比较
        if src_dtype == torch.float32 and dst_dtype == torch.bfloat16:
            x_compare = x.bfloat16()
            assert torch.allclose(x_compare, y, rtol=1e-2, atol=1e-2), "float32到bfloat16的转换结果不匹配"
        elif src_dtype == torch.bfloat16 and dst_dtype == torch.float32:
            y_compare = y.bfloat16()
            assert torch.allclose(x, y_compare, rtol=1e-2, atol=1e-2), "bfloat16到float32的转换结果不匹配"

    @allure.story("边界测试 - 大张量")
    @allure.title("测试大张量的Device to Device拷贝 - {src_dtype} -> {dst_dtype}")
    @allure.description("""
    验证大张量的Device to Device拷贝，测试要点：
    1. 验证大规模数据的拷贝正确性
    2. 测试不同维度的大张量
    3. 验证内存使用情况
    """)
    @pytest.mark.parametrize("src_dtype,dst_dtype", test_dtypes)
    def test_copy_large_tensor(self, device, src_dtype, dst_dtype):
        if not torch.cuda.is_available():
            pytest.skip("CUDA设备不可用，跳过测试")

        # 测试不同维度的大张量
        shapes = [
            (1024, 1024),          # 2D大矩阵
            (32, 64, 128),         # 3D张量
            (16, 32, 64, 32)       # 4D张量
        ]

        for shape in shapes:
            # 准备大张量数据
            x = torch.randn(*shape).to(src_dtype)
            y = torch.empty_like(x, dtype=dst_dtype)

            if device == "cuda":
                x = x.cuda()
                y = y.cuda()

            # 执行copy_操作
            y.copy_(x)

            # 验证形状保持不变
            assert x.shape == y.shape, f"大张量拷贝后形状不一致: 输入{x.shape} vs 输出{y.shape}"

            # 将数据转换为相同类型进行比较
            if src_dtype == torch.float32 and dst_dtype == torch.bfloat16:
                x_compare = x.bfloat16()
                assert torch.allclose(x_compare, y, rtol=1e-2, atol=1e-2), f"大张量{shape}的float32到bfloat16的转换结果不匹配"
            elif src_dtype == torch.bfloat16 and dst_dtype == torch.float32:
                y_compare = y.bfloat16()
                assert torch.allclose(x, y_compare, rtol=1e-2, atol=1e-2), f"大张量{shape}的bfloat16到float32的转换结果不匹配"

            # 清理内存
            del x, y
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    @allure.story("边界测试 - 特殊值")
    @allure.title("测试特殊值的Device to Device拷贝 - {src_dtype} -> {dst_dtype}")
    @allure.description("""
    验证包含特殊值的张量的Device to Device拷贝，测试要点：
    1. 验证无穷大值的拷贝
    2. 验证NaN值的拷贝
    3. 验证极大值和极小值的拷贝
    """)
    @pytest.mark.parametrize("src_dtype,dst_dtype", test_dtypes)
    def test_copy_special_values(self, device, src_dtype, dst_dtype):
        if not torch.cuda.is_available():
            pytest.skip("CUDA设备不可用，跳过测试")

        # 创建包含特殊值的张量
        special_values = torch.tensor([
            float('inf'),          # 正无穷
            float('-inf'),         # 负无穷
            float('nan'),          # NaN
            1e38,                  # 较大的值
            -1e38,                 # 较小的值
            1.0,                   # 普通值
            0.0,                   # 零
            -1.0                   # 负数
        ]).reshape(-1, 2)

        # 转换为目标类型并移动到GPU
        x = special_values.to(src_dtype).cuda()
        y = torch.empty_like(x, dtype=dst_dtype).cuda()

        # 执行copy_操作
        y.copy_(x)

        # 验证形状保持不变
        assert x.shape == y.shape, f"特殊值拷贝后形状不一致: 输入{x.shape} vs 输出{y.shape}"

        # 根据不同的类型组合进行验证
        if src_dtype == torch.float32 and dst_dtype == torch.bfloat16:
            x_compare = x.bfloat16()
            self._check_special_values(x_compare, y)
        elif src_dtype == torch.bfloat16 and dst_dtype == torch.float32:
            y_compare = y.bfloat16()
            self._check_special_values(x, y_compare)
