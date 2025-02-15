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
@pytest.mark.order(1)
class TestCast:
    def teardown_method(self, method):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    @allure.story("基础功能测试")
    @allure.title("测试不同数据类型之间的转换 - {src_dtype} -> {dst_dtype}")
    @pytest.mark.parametrize("src_dtype", test_dtypes)
    @pytest.mark.parametrize("dst_dtype", test_dtypes)
    def test_cast_basic(self, device, src_dtype, dst_dtype):
        # 准备测试数据
        if src_dtype == torch.bool:
            x = torch.randint(0, 2, (2, 3), dtype=torch.int64).to(src_dtype)
        else:
            x = torch.randn(2, 3).to(src_dtype)
        
        if device == "cpu":
            dev_obj = get_device_object("cpu")
            with allure.step(f"执行类型转换 - 设备: cpu, 源类型: {src_dtype}, 目标类型: {dst_dtype}"):
                x_dev = x.to(device=dev_obj)
                output = x_dev.to(dst_dtype)
                
            with allure.step("验证输出"):
                expected = x.to(dst_dtype)
                assert output.shape == expected.shape, f"输出形状不符合预期: 期望 {expected.shape}, 实际 {output.shape}"
                assert output.dtype == dst_dtype, f"输出数据类型不符合预期: 期望 {dst_dtype}, 实际 {output.dtype}"
                
                # 对于浮点数到整数的转换，使用allclose进行比较
                if src_dtype in [torch.float32, torch.float64] and dst_dtype in [torch.int32, torch.int64]:
                    assert torch.allclose(output.float(), expected.float(), rtol=1e-5), "输出结果不正确"
                else:
                    assert torch.all(output == expected), "输出结果不正确"
        
        elif device == "cuda":
            # 在CPU上运行
            cpu_output = x.to(dst_dtype)
            
            # 在CUDA上运行
            dev_obj = get_device_object("cuda")
            with allure.step(f"执行类型转换 - 设备: cuda, 源类型: {src_dtype}, 目标类型: {dst_dtype}"):
                get_device_info()
                x_dev = x.to(device=dev_obj)
                cuda_output = x_dev.to(dst_dtype)
                
            with allure.step("验证输出"):
                assert cuda_output.shape == cpu_output.shape, f"输出形状不符合预期: 期望 {cpu_output.shape}, 实际 {cuda_output.shape}"
                assert cuda_output.dtype == dst_dtype, f"输出数据类型不符合预期: 期望 {dst_dtype}, 实际 {cuda_output.dtype}"
                
                # 将CUDA输出转移到CPU进行比较
                cuda_output_cpu = cuda_output.cpu()
                
                # 对于浮点数到整数的转换，使用allclose进行比较
                if src_dtype in [torch.float32, torch.float64] and dst_dtype in [torch.int32, torch.int64]:
                    assert torch.allclose(cuda_output_cpu.float(), cpu_output.float(), rtol=1e-5), "CUDA输出与CPU输出不一致"
                else:
                    assert torch.all(cuda_output_cpu == cpu_output), "CUDA输出与CPU输出不一致"
    
    @allure.story("边界条件测试")
    @allure.title("测试特殊值的类型转换")
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
            
            if device == "cpu":
                dev_obj = get_device_object("cpu")
                with allure.step(f"执行特殊值转换 - 设备: cpu, 源类型: {src_dtype}, 目标类型: {dst_dtype}"):
                    x_dev = x.to(device=dev_obj)
                    output = x_dev.to(dst_dtype)
                    
                with allure.step("验证输出"):
                    expected = x.to(dst_dtype)
                    assert output.shape == expected.shape, "形状不匹配"
                    assert output.dtype == dst_dtype, "数据类型不匹配"
                    
                    if torch.isnan(x).any():
                        assert torch.isnan(output).any() == torch.isnan(expected).any(), "NaN处理不一致"
                        # 对非NaN值进行比较
                        mask = ~torch.isnan(x)
                        assert torch.all(output[mask] == expected[mask]), "非NaN值不一致"
                    else:
                        assert torch.all(output == expected), "输出结果不正确"
            
            elif device == "cuda":
                # 在CPU上运行
                cpu_output = x.to(dst_dtype)
                
                # 在CUDA上运行
                dev_obj = get_device_object("cuda")
                with allure.step(f"执行特殊值转换 - 设备: cuda, 源类型: {src_dtype}, 目标类型: {dst_dtype}"):
                    get_device_info()
                    x_dev = x.to(device=dev_obj)
                    cuda_output = x_dev.to(dst_dtype)
                
                with allure.step("验证输出"):
                    cuda_output_cpu = cuda_output.cpu()
                    assert cuda_output.shape == cpu_output.shape, "形状不匹配"
                    assert cuda_output.dtype == dst_dtype, "数据类型不匹配"
                    
                    if torch.isnan(x).any():
                        assert torch.isnan(cuda_output_cpu).any() == torch.isnan(cpu_output).any(), "NaN处理不一致"
                        # 对非NaN值进行比较
                        mask = ~torch.isnan(x)
                        assert torch.all(cuda_output_cpu[mask] == cpu_output[mask]), "非NaN值不一致"
                    elif dst_dtype in [torch.int32, torch.int64] and src_dtype in [torch.float32, torch.float64]:
                        # 对于浮点数转整数的情况，某些极限值可能会有平台差异
                        # 将结果转回浮点数进行比较
                        assert torch.allclose(
                            cuda_output_cpu.float(),
                            cpu_output.float(),
                            rtol=1e-5,
                            atol=1e-5
                        ), "CUDA输出与CPU输出不一致（考虑精度）"
                    else:
                        assert torch.all(cuda_output_cpu == cpu_output), "CUDA输出与CPU输出不一致"
    
    @allure.story("性能测试")
    @allure.title("测试大规模数据的类型转换")
    def test_cast_performance(self, device):
        # 准备大规模测试数据
        x = torch.randn(1000, 1000)  # 100万个元素
        src_dtype = torch.float32
        dst_dtype = torch.int64
        x = x.to(src_dtype)
        
        if device == "cpu":
            dev_obj = get_device_object("cpu")
            with allure.step(f"执行大规模转换 - 设备: cpu, 源类型: {src_dtype}, 目标类型: {dst_dtype}"):
                x_dev = x.to(device=dev_obj)
                output = x_dev.to(dst_dtype)
                
            with allure.step("验证输出"):
                expected = x.to(dst_dtype)
                assert output.shape == expected.shape, "形状不匹配"
                assert output.dtype == dst_dtype, "数据类型不匹配"
                assert torch.allclose(output.float(), expected.float(), rtol=1e-5), "输出结果不正确"
        
        elif device == "cuda":
            # 在CPU上运行
            cpu_output = x.to(dst_dtype)
            
            # 在CUDA上运行
            dev_obj = get_device_object("cuda")
            with allure.step(f"执行大规模转换 - 设备: cuda, 源类型: {src_dtype}, 目标类型: {dst_dtype}"):
                get_device_info()
                x_dev = x.to(device=dev_obj)
                cuda_output = x_dev.to(dst_dtype)
                
            with allure.step("验证输出"):
                cuda_output_cpu = cuda_output.cpu()
                assert cuda_output.shape == cpu_output.shape, "形状不匹配"
                assert cuda_output.dtype == dst_dtype, "数据类型不匹配"
                assert torch.allclose(cuda_output_cpu.float(), cpu_output.float(), rtol=1e-5), "CUDA输出与CPU输出不一致"
