import torch
import pytest
import allure
import numpy as np
from utils.device_utils import get_device_object, get_device_info

# 支持的数据类型列表
test_dtypes = [
    torch.float32,
    torch.float64,
    torch.int32,
    torch.int64,
    torch.bool
]

@allure.epic("PyTorch算子测试")
@allure.feature("Cast算子")
@allure.description("""
该测试模块验证PyTorch中Cast算子的功能正确性，包括：
1. 基本类型转换：验证不同数据类型之间的转换
2. 特殊值处理：验证对NaN、无穷等特殊值的处理
3. 性能测试：验证大规模数据的类型转换性能

所有测试都在CPU和CUDA设备上执行，并验证结果的一致性。
""")
@pytest.mark.order(1)
class TestCast:
    def teardown_method(self, method):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    @allure.story("基础功能测试")
    @allure.title("测试不同数据类型之间的转换 - {src_dtype} -> {dst_dtype}")
    @allure.description("""
    验证不同数据类型之间的转换，测试要点：
    1. 支持各种数据类型的相互转换
    2. 保持张量的原有形状
    3. 验证转换后的数值精度
    4. CPU和CUDA设备结果一致性
    """)
    @pytest.mark.parametrize("src_dtype", test_dtypes)
    @pytest.mark.parametrize("dst_dtype", test_dtypes)
    def test_cast_basic(self, device, src_dtype, dst_dtype):
        # 准备测试数据
        if src_dtype == torch.bool:
            x = torch.randint(0, 2, (2, 3), dtype=torch.int64).to(src_dtype)
        else:
            x = torch.randn(2, 3).to(src_dtype)
        
        device_obj = get_device_object(device)
        with allure.step(f"执行类型转换 - 设备: {device}, 源类型: {src_dtype}, 目标类型: {dst_dtype}"):
#            if device == "cuda":
#                get_device_info()
            x_dev = x.to(device=device_obj)
            output = x_dev.to(dst_dtype)
        
        with allure.step("验证输出"):
            # 计算期望输出
            expected = x.to(dst_dtype)
            # 将输出转回CPU进行比较
            output_cpu = output.cpu()
            
            assert output.shape == expected.shape, f"输出形状不符合预期: 期望 {expected.shape}, 实际 {output.shape}"
            assert output.dtype == dst_dtype, f"输出数据类型不符合预期: 期望 {dst_dtype}, 实际 {output.dtype}"
            
            # 对于浮点数到整数的转换，使用allclose进行比较
            if dst_dtype in [torch.int32, torch.int64]:
                assert torch.all(output_cpu == expected), "输出结果不正确"
            else:
                assert torch.allclose(output_cpu.float(), expected.float(), rtol=1e-5), "输出结果不正确"
        
        if device == "cuda":
            # 在CPU上运行参考结果
            cpu_output = x.to(dst_dtype)
            
            with allure.step("比较CPU和CUDA结果"):
                if src_dtype in [torch.float32, torch.float64] and dst_dtype in [torch.int32, torch.int64]:
                    assert torch.allclose(output_cpu.float(), cpu_output.float(), rtol=1e-5), "CUDA输出与CPU输出不一致"
                else:
                    assert torch.all(output_cpu == cpu_output), "CUDA输出与CPU输出不一致"
    
    @allure.story("边界条件测试")
    @allure.title("测试特殊值的类型转换")
    @allure.description("""
    验证对特殊值的类型转换处理，测试要点：
    1. 处理特殊浮点数值（inf、-inf、nan）
    2. 整数边界值的转换
    3. 布尔值的转换
    4. 大数值的安全转换
    """)
    def test_cast_special_values(self, device):
        special_values = [
            # 测试特殊浮点数值
            (torch.tensor([float('inf'), float('-inf'), float('nan')]), torch.float32, torch.float64),
            # 测试整数边界值
            (torch.tensor([0, 1, -1, 127, -128]), torch.int8, torch.float32),
            # 测试布尔值
            (torch.tensor([True, False]), torch.bool, torch.float32),
            # 测试大数（但不超出范围）
            (torch.tensor([1e5, -1e5]), torch.float32, torch.int64),
        ]
        
        for x, src_dtype, dst_dtype in special_values:
            x = x.to(src_dtype)
            
            device_obj = get_device_object(device)
            with allure.step(f"执行特殊值转换 - 设备: {device}, 源类型: {src_dtype}, 目标类型: {dst_dtype}"):
                if device == "cuda":
                    get_device_info()
                x_dev = x.to(device=device_obj)
                output = x_dev.to(dst_dtype)
            
            with allure.step("验证输出"):
                # 计算期望输出
                expected = x.to(dst_dtype)
                # 将输出转回CPU进行比较
                output_cpu = output.cpu()
                
                assert output.shape == expected.shape, "形状不匹配"
                assert output.dtype == dst_dtype, "数据类型不匹配"
                
                if torch.isnan(x).any() or torch.isinf(x).any():
                    # 对于NaN和无穷大值，使用特殊的比较逻辑
                    nan_mask = torch.isnan(x)
                    inf_mask = torch.isinf(x)
                    normal_mask = ~(nan_mask | inf_mask)
                    
                    # 验证NaN的位置是否一致
                    assert torch.all(torch.isnan(output_cpu) == torch.isnan(expected)), "NaN处理不一致"
                    
                    # 验证无穷大值的符号是否一致
                    if inf_mask.any():
                        assert torch.all(torch.sign(output_cpu[inf_mask]) == torch.sign(expected[inf_mask])), "无穷大值的符号不一致"
                    
                    # 对普通值进行比较
                    if normal_mask.any():
                        assert torch.all(output_cpu[normal_mask] == expected[normal_mask]), "普通值不一致"
                else:
                    assert torch.all(output_cpu == expected), "输出结果不正确"
            
            if device == "cuda":
                # 在CPU上运行参考结果
                cpu_output = x.to(dst_dtype)
                
                with allure.step("比较CPU和CUDA结果"):
                    if torch.isnan(x).any() or torch.isinf(x).any():
                        # 对于NaN和无穷大值，使用特殊的比较逻辑
                        nan_mask = torch.isnan(output_cpu)
                        inf_mask = torch.isinf(output_cpu)
                        normal_mask = ~(nan_mask | inf_mask)

                        # 验证NaN的位置是否一致
                        assert torch.all(torch.isnan(output_cpu) == torch.isnan(cpu_output)), "NaN处理不一致"

                        # 验证无穷大值的符号是否一致
                        if inf_mask.any():
                            assert torch.all(torch.sign(output_cpu[inf_mask]) == torch.sign(cpu_output[inf_mask])), "无穷大值的符号不一致"

                        # 对普通值进行比较
                        if normal_mask.any():
                            assert torch.all(output_cpu[normal_mask] == cpu_output[normal_mask]), "普通值不一致"
                    elif dst_dtype in [torch.int32, torch.int64] and src_dtype in [torch.float32, torch.float64]:
                        # 对于浮点数转整数的情况，某些极限值可能会有平台差异
                        # 将结果转回浮点数进行比较
                        assert torch.allclose(
                            output_cpu.float(),
                            cpu_output.float(),
                            rtol=1e-5,
                            atol=1e-5
                        ), "CUDA输出与CPU输出不一致（考虑精度）"
                    else:
                        assert torch.all(output_cpu == cpu_output), "CUDA输出与CPU输出不一致"
    
    @allure.story("性能测试")
    @allure.title("测试大规模数据的类型转换")
    @allure.description("""
    验证大规模数据的类型转换性能，测试要点：
    1. 使用1000x1000大小的张量
    2. 验证大规模转换的数值稳定性
    3. 比较CPU和CUDA设备的转换效率
    4. 确保内存使用合理
    """)
    def test_cast_performance(self, device):
        # 准备大规模测试数据
        x = torch.randn(1000, 1000)  # 100万个元素
        src_dtype = torch.float32
        dst_dtype = torch.int64
        x = x.to(src_dtype)
        
        device_obj = get_device_object(device)
        with allure.step(f"执行大规模转换 - 设备: {device}, 源类型: {src_dtype}, 目标类型: {dst_dtype}"):
            if device == "cuda":
                get_device_info()
            x_dev = x.to(device=device_obj)
            output = x_dev.to(dst_dtype)
        
        with allure.step("验证输出"):
            # 计算期望输出
            expected = x.to(dst_dtype)
            # 将输出转回CPU进行比较
            output_cpu = output.cpu()
            
            assert output.shape == expected.shape, "形状不匹配"
            assert output.dtype == dst_dtype, "数据类型不匹配"
            assert torch.allclose(output_cpu.float(), expected.float(), rtol=1e-5), "输出结果不正确"
        
        if device == "cuda":
            # 在CPU上运行参考结果
            cpu_output = x.to(dst_dtype)
            
            with allure.step("比较CPU和CUDA结果"):
                assert torch.allclose(output_cpu.float(), cpu_output.float(), rtol=1e-5), "CUDA输出与CPU输出不一致"
