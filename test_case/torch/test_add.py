#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import torch
import allure
import numpy as np
from .utils import get_device_object, test_dtypes

@allure.epic("PyTorch算子测试")
@allure.feature("Add算子")
@allure.description("""
该测试模块验证PyTorch中Add算子的功能正确性，包括：
1. 标量加法：验证张量与标量的加法运算
2. 张量加法：验证相同形状张量之间的加法
3. 广播加法：验证不同形状张量之间的广播加法
4. 性能测试：验证大规模数据下的加法性能

所有测试都在CPU和CUDA设备上执行，并验证结果的一致性。
""")
@pytest.mark.order(1)
class TestAdd:
    def teardown_method(self, method):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    @allure.story("基础功能测试")
    @allure.title("测试不同数据类型的标量加法")
    @allure.description("""
    验证张量与标量的加法操作，测试要点：
    1. 支持多种数据类型（float32, float64, int32等）
    2. 保持张量的原有形状
    3. CPU和CUDA设备结果一致性
    4. 数值计算精度
    """)
    @pytest.mark.parametrize("dtype", test_dtypes)
    def test_add_scalar(self, device, dtype):
        # 准备测试数据
        a = torch.tensor([[1, 2], [3, 4]], dtype=dtype)
        scalar = 2
        
        if device == "cpu":
            dev_obj = get_device_object("cpu")
            with allure.step(f"执行标量加法 - 设备: cpu, 数据类型: {dtype}"):
                a_dev = a.to(device=dev_obj)
                output = a_dev + scalar
                
            with allure.step("验证输出"):
                expected = a + scalar
                assert output.shape == expected.shape, f"输出形状不符合预期: 期望 {expected.shape}, 实际 {output.shape}"
                assert output.dtype == expected.dtype, f"输出数据类型不符合预期: 期望 {expected.dtype}, 实际 {output.dtype}"
                assert torch.all(output == expected), "输出结果不正确"
        
        elif device == "cuda":
            # 在CPU上运行
            cpu_output = a + scalar
            
            # 在CUDA上运行
            with allure.step(f"执行标量加法 - 设备: cuda, 数据类型: {dtype}"):
                cuda_a = a.cuda()
                cuda_output = cuda_a + scalar
                
            with allure.step("验证输出"):
                assert cuda_output.shape == cpu_output.shape, "CUDA和CPU的输出形状不一致"
                
            with allure.step("比较CPU和CUDA结果"):
                cuda_output_cpu = cuda_output.cpu()
                max_diff = torch.max(torch.abs(cpu_output - cuda_output_cpu))
                assert max_diff < 1e-5, f"CPU和CUDA结果不一致，最大差异: {max_diff}"

    @allure.story("基础功能测试")
    @allure.title("测试不同数据类型的张量加法")
    @allure.description("""
    验证两个相同形状张量的加法操作，测试要点：
    1. 支持多种数据类型（float32, float64, int32等）
    2. 验证输出张量的形状正确性
    3. CPU和CUDA设备结果一致性
    4. 数值计算精度
    """)
    @pytest.mark.parametrize("dtype", test_dtypes)
    def test_add_tensor(self, device, dtype):
        # 准备测试数据
        a = torch.tensor([[1, 2], [3, 4]], dtype=dtype)
        b = torch.tensor([[5, 6], [7, 8]], dtype=dtype)
        
        if device == "cpu":
            dev_obj = get_device_object("cpu")
            with allure.step(f"执行张量加法 - 设备: cpu, 数据类型: {dtype}"):
                a_dev = a.to(device=dev_obj)
                b_dev = b.to(device=dev_obj)
                output = a_dev + b_dev
                
            with allure.step("验证输出"):
                expected = a + b
                assert output.shape == expected.shape, f"输出形状不符合预期: 期望 {expected.shape}, 实际 {output.shape}"
                assert output.dtype == expected.dtype, f"输出数据类型不符合预期: 期望 {expected.dtype}, 实际 {output.dtype}"
                assert torch.all(output == expected), "输出结果不正确"
        
        elif device == "cuda":
            # 在CPU上运行
            cpu_output = a + b
            
            # 在CUDA上运行
            with allure.step(f"执行张量加法 - 设备: cuda, 数据类型: {dtype}"):
                cuda_a = a.cuda()
                cuda_b = b.cuda()
                cuda_output = cuda_a + cuda_b
                
            with allure.step("验证输出"):
                assert cuda_output.shape == cpu_output.shape, "CUDA和CPU的输出形状不一致"
                
            with allure.step("比较CPU和CUDA结果"):
                cuda_output_cpu = cuda_output.cpu()
                max_diff = torch.max(torch.abs(cpu_output - cuda_output_cpu))
                assert max_diff < 1e-5, f"CPU和CUDA结果不一致，最大差异: {max_diff}"

    @allure.story("边界条件测试")
    @allure.title("测试大规模加法")
    @allure.description("""
    验证大规模数据下的加法性能，测试要点：
    1. 使用1000x1000大小的张量
    2. 验证大规模计算的数值稳定性
    3. 比较CPU和CUDA设备的计算结果
    4. 确保内存使用合理
    """)
    def test_add_performance(self, device):
        # 准备大规模测试数据
        dtype = torch.float32
        a = torch.randn((1000, 1000), dtype=dtype)
        b = torch.randn((1000, 1000), dtype=dtype)
        
        if device == "cpu":
            dev_obj = get_device_object("cpu")
            with allure.step("执行大规模加法 - 设备: cpu"):
                a_dev = a.to(device=dev_obj)
                b_dev = b.to(device=dev_obj)
                output = a_dev + b_dev
                
            with allure.step("验证输出"):
                expected = a + b
                assert output.shape == expected.shape, f"输出形状不符合预期: 期望 {expected.shape}, 实际 {output.shape}"
                assert torch.allclose(output, expected, rtol=1e-5), "输出结果不正确"
        
        elif device == "cuda":
            # 在CPU上运行
            cpu_output = a + b
            
            # 在CUDA上运行
            with allure.step("执行大规模加法 - 设备: cuda"):
                cuda_a = a.cuda()
                cuda_b = b.cuda()
                cuda_output = cuda_a + cuda_b
                
            with allure.step("验证输出"):
                assert cuda_output.shape == cpu_output.shape, "CUDA和CPU的输出形状不一致"
                
            with allure.step("比较CPU和CUDA结果"):
                cuda_output_cpu = cuda_output.cpu()
                max_diff = torch.max(torch.abs(cpu_output - cuda_output_cpu))
                assert max_diff < 1e-5, f"CPU和CUDA结果不一致，最大差异: {max_diff}"

    @allure.story("边界条件测试")
    @allure.title("测试广播加法")
    @allure.description("""
    验证不同形状张量之间的广播加法，测试要点：
    1. 验证广播规则的正确应用
    2. 测试不同维度组合（3x1x4 + 2x4 -> 3x2x4）
    3. 验证输出张量形状的正确性
    4. CPU和CUDA设备结果一致性
    """)
    def test_add_broadcast(self, device):
        # 准备测试数据
        dtype = torch.float32
        a = torch.randn((3, 1, 4), dtype=dtype)  # shape: (3, 1, 4)
        b = torch.randn((2, 4), dtype=dtype)     # shape: (2, 4)
        # 结果shape应该是: (3, 2, 4)
        
        if device == "cpu":
            dev_obj = get_device_object("cpu")
            with allure.step("执行广播加法 - 设备: cpu"):
                a_dev = a.to(device=dev_obj)
                b_dev = b.to(device=dev_obj)
                output = a_dev + b_dev
                
            with allure.step("验证输出"):
                expected = a + b
                expected_shape = (3, 2, 4)
                assert output.shape == expected_shape, f"输出形状不符合预期: 期望 {expected_shape}, 实际 {output.shape}"
                assert torch.allclose(output, expected, rtol=1e-5), "输出结果不正确"
        
        elif device == "cuda":
            # 在CPU上运行
            cpu_output = a + b
            
            # 在CUDA上运行
            with allure.step("执行广播加法 - 设备: cuda"):
                cuda_a = a.cuda()
                cuda_b = b.cuda()
                cuda_output = cuda_a + cuda_b
                
            with allure.step("验证输出"):
                assert cuda_output.shape == cpu_output.shape, "CUDA和CPU的输出形状不一致"
                
            with allure.step("比较CPU和CUDA结果"):
                cuda_output_cpu = cuda_output.cpu()
                max_diff = torch.max(torch.abs(cpu_output - cuda_output_cpu))
                assert max_diff < 1e-5, f"CPU和CUDA结果不一致，最大差异: {max_diff}"
