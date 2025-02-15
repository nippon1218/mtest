#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import torch
import allure
import torch.nn.functional as F
from .utils import get_device_object, test_dtypes

@allure.epic("PyTorch算子测试")
@allure.feature("SiLU算子")
@pytest.mark.order(6)
class TestSiLU:
    def setup_method(self, method):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
    def teardown_method(self, method):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    @allure.story("基础功能测试")
    @allure.title("测试不同数据类型的SiLU")
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_silu_basic(self, device, dtype):
        # 准备测试数据
        x = torch.tensor([[-2.0, -1.0, 0.0, 1.0, 2.0]], dtype=dtype)
        
        if device == "cpu":
            dev_obj = get_device_object("cpu")
            with allure.step(f"执行SiLU - 设备: cpu, 数据类型: {dtype}"):
                x_dev = x.to(device=dev_obj)
                output = F.silu(x_dev)
                
            with allure.step("验证输出"):
                # 手动计算期望值：x * sigmoid(x)
                sigmoid = torch.sigmoid(x)
                expected = x * sigmoid
                assert output.shape == expected.shape, f"输出形状不符合预期: 期望 {expected.shape}, 实际 {output.shape}"
                assert output.dtype == dtype, f"输出数据类型不符合预期: 期望 {dtype}, 实际 {output.dtype}"
                assert torch.allclose(output, expected, rtol=1e-5), "输出结果不正确"
        
        elif device == "cuda":
            # 在CPU上运行
            cpu_output = F.silu(x)
            
            # 在CUDA上运行
            with allure.step(f"执行SiLU - 设备: cuda, 数据类型: {dtype}"):
                cuda_x = x.cuda()
                cuda_output = F.silu(cuda_x)
                
            with allure.step("验证输出"):
                assert cuda_output.shape == cpu_output.shape, "CUDA和CPU的输出形状不一致"
                
            with allure.step("比较CPU和CUDA结果"):
                cuda_output_cpu = cuda_output.cpu()
                max_diff = torch.max(torch.abs(cpu_output - cuda_output_cpu))
                assert max_diff < 1e-5, f"CPU和CUDA结果不一致，最大差异: {max_diff}"

    @allure.story("边界条件测试")
    @allure.title("测试SiLU的边界值")
    def test_silu_edge_cases(self, device):
        # 准备测试数据，包括非常大和非常小的值
        dtype = torch.float32
        # 测试不同量级的输入
        x_normal = torch.tensor([-1e3, -1e2, -10, -1, 0, 1, 10, 1e2, 1e3], dtype=dtype)
        # 测试特殊值
        x_special = torch.tensor([float('inf'), float('-inf'), float('nan')], dtype=dtype)
        
        if device == "cpu":
            dev_obj = get_device_object("cpu")
            with allure.step("执行SiLU边界值测试 - 设备: cpu"):
                # 测试不同量级的输入
                x_normal_dev = x_normal.to(device=dev_obj)
                output_normal = F.silu(x_normal_dev)
                
                # 测试特殊值
                x_special_dev = x_special.to(device=dev_obj)
                output_special = F.silu(x_special_dev)
                
            with allure.step("验证输出"):
                # 对于非常大的正数，SiLU应该接近输入值
                assert torch.allclose(output_normal[-1], x_normal[-1], rtol=1e-3), "大正数的输出不正确"
                # 对于非常小的负数，SiLU应该接近0
                assert torch.abs(output_normal[0]) < 1e-3, "大负数的输出不接近0"
                # 对于0，输出应该是0
                assert output_normal[4] == 0, "0的输出不是0"
                # 特殊值的处理
                assert torch.isinf(output_special[0]), "正无穷大的输出应为无穷大"
                assert torch.isnan(output_special[1]), "负无穷大的输出应为NaN"
                assert torch.isnan(output_special[2]), "NaN的输出应为NaN"
        
        elif device == "cuda":
            # 在CPU上运行
            cpu_output_normal = F.silu(x_normal)
            cpu_output_special = F.silu(x_special)
            
            # 在CUDA上运行
            with allure.step("执行SiLU边界值测试 - 设备: cuda"):
                cuda_x_normal = x_normal.cuda()
                cuda_x_special = x_special.cuda()
                cuda_output_normal = F.silu(cuda_x_normal)
                cuda_output_special = F.silu(cuda_x_special)
                
            with allure.step("验证输出"):
                assert cuda_output_normal.shape == cpu_output_normal.shape, "CUDA和CPU的输出形状不一致"
                assert cuda_output_special.shape == cpu_output_special.shape, "CUDA和CPU的特殊值输出形状不一致"
                
            with allure.step("比较CPU和CUDA结果"):
                cuda_output_normal_cpu = cuda_output_normal.cpu()
                # 只比较正常值
                max_diff = torch.max(torch.abs(cpu_output_normal - cuda_output_normal_cpu))
                assert max_diff < 1e-5, f"CPU和CUDA结果不一致，最大差异: {max_diff}"

    @allure.story("边界条件测试")
    @allure.title("测试大规模SiLU")
    def test_silu_performance(self, device):
        # 准备大规模测试数据
        dtype = torch.float32
        shape = (1000, 1000)  # 100万个元素
        x = torch.randn(shape, dtype=dtype)
        
        if device == "cpu":
            dev_obj = get_device_object("cpu")
            with allure.step("执行大规模SiLU - 设备: cpu"):
                x_dev = x.to(device=dev_obj)
                output = F.silu(x_dev)
                
            with allure.step("验证输出"):
                assert output.shape == shape, f"输出形状不符合预期: 期望 {shape}, 实际 {output.shape}"
                # 验证输出范围
                # SiLU(x) = x * sigmoid(x)
                # 当x < 0时，输出应该小于0
                # 当x > 0时，输出应该大于0
                neg_mask = x < 0
                pos_mask = x > 0
                assert torch.all(output[neg_mask] < 0), "负输入的SiLU输出应为负数"
                assert torch.all(output[pos_mask] > 0), "正输入的SiLU输出应为正数"
        
        elif device == "cuda":
            # 在CPU上运行
            cpu_output = F.silu(x)
            
            # 在CUDA上运行
            with allure.step("执行大规模SiLU - 设备: cuda"):
                cuda_x = x.cuda()
                cuda_output = F.silu(cuda_x)
                
            with allure.step("验证输出"):
                assert cuda_output.shape == cpu_output.shape, "CUDA和CPU的输出形状不一致"
                
            with allure.step("比较CPU和CUDA结果"):
                cuda_output_cpu = cuda_output.cpu()
                max_diff = torch.max(torch.abs(cpu_output - cuda_output_cpu))
                assert max_diff < 1e-4, f"CPU和CUDA结果不一致，最大差异: {max_diff}"

    @allure.story("基础功能测试")
    @allure.title("测试不同维度的SiLU")
    def test_silu_dimensions(self, device):
        # 准备不同维度的测试数据
        dtype = torch.float32
        test_shapes = [
            (10,),           # 1D
            (5, 10),         # 2D
            (3, 4, 5),       # 3D
            (2, 3, 4, 5)     # 4D
        ]
        
        for shape in test_shapes:
            x = torch.randn(shape, dtype=dtype)
            
            if device == "cpu":
                dev_obj = get_device_object("cpu")
                with allure.step(f"执行{len(shape)}D SiLU - 设备: cpu"):
                    x_dev = x.to(device=dev_obj)
                    output = F.silu(x_dev)
                    
                with allure.step("验证输出"):
                    assert output.shape == shape, f"输出形状不符合预期: 期望 {shape}, 实际 {output.shape}"
            
            elif device == "cuda":
                # 在CPU上运行
                cpu_output = F.silu(x)
                
                # 在CUDA上运行
                with allure.step(f"执行{len(shape)}D SiLU - 设备: cuda"):
                    cuda_x = x.cuda()
                    cuda_output = F.silu(cuda_x)
                    
                with allure.step("验证输出"):
                    assert cuda_output.shape == cpu_output.shape, "CUDA和CPU的输出形状不一致"
                    
                with allure.step("比较CPU和CUDA结果"):
                    cuda_output_cpu = cuda_output.cpu()
                    max_diff = torch.max(torch.abs(cpu_output - cuda_output_cpu))
                    assert max_diff < 1e-5, f"CPU和CUDA结果不一致，最大差异: {max_diff}"
