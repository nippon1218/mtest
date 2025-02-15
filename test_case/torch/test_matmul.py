#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import torch
import allure
import numpy as np
from .utils import get_device_object, test_dtypes

@allure.epic("PyTorch算子测试")
@allure.feature("MatMul算子")
@pytest.mark.order(3)
class TestMatMul:
    def setup_method(self, method):
        # 检查CUDA是否可用
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
        if not torch.cuda.is_available():
            pytest.skip("CUDA不可用")
            
    def teardown_method(self, method):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    @allure.story("基础功能测试")
    @allure.title("测试不同数据类型的矩阵乘法")
    @pytest.mark.parametrize("dtype", test_dtypes)
    def test_matmul_basic(self, device, dtype):
        # 检查数据类型是否在CUDA上支持
        if device == "cuda" and dtype in [torch.int32, torch.int64]:
            pytest.skip(f"数据类型 {dtype} 在CUDA上不支持")
            
        # 准备测试数据
        a = torch.tensor([[1, 2], [3, 4]], dtype=dtype)  # shape: (2, 2)
        b = torch.tensor([[5, 6], [7, 8]], dtype=dtype)  # shape: (2, 2)
        
        if device == "cpu":
            dev_obj = get_device_object("cpu")
            with allure.step(f"执行矩阵乘法 - 设备: cpu, 数据类型: {dtype}"):
                a_dev = a.to(device=dev_obj)
                b_dev = b.to(device=dev_obj)
                output = torch.matmul(a_dev, b_dev)
                
            with allure.step("验证输出"):
                expected = torch.matmul(a, b)
                assert output.shape == expected.shape, f"输出形状不符合预期: 期望 {expected.shape}, 实际 {output.shape}"
                assert output.dtype == expected.dtype, f"输出数据类型不符合预期: 期望 {expected.dtype}, 实际 {output.dtype}"
                assert torch.all(output == expected), "输出结果不正确"
        
        elif device == "cuda":
            # 在CPU上运行
            cpu_output = torch.matmul(a, b)
            
            # 在CUDA上运行
            with allure.step(f"执行矩阵乘法 - 设备: cuda, 数据类型: {dtype}"):
                cuda_a = a.cuda()
                cuda_b = b.cuda()
                cuda_output = torch.matmul(cuda_a, cuda_b)
                
            with allure.step("验证输出"):
                assert cuda_output.shape == cpu_output.shape, "CUDA和CPU的输出形状不一致"
                
            with allure.step("比较CPU和CUDA结果"):
                cuda_output_cpu = cuda_output.cpu()
                max_diff = torch.max(torch.abs(cpu_output - cuda_output_cpu))
                assert max_diff < 1e-5, f"CPU和CUDA结果不一致，最大差异: {max_diff}"

    @allure.story("边界条件测试")
    @allure.title("测试大规模矩阵乘法")
    def test_matmul_performance(self, device):
        # 准备大规模测试数据
        dtype = torch.float32
        a = torch.randn((100, 200), dtype=dtype)
        b = torch.randn((200, 100), dtype=dtype)
        
        if device == "cpu":
            dev_obj = get_device_object("cpu")
            with allure.step("执行大规模矩阵乘法 - 设备: cpu"):
                a_dev = a.to(device=dev_obj)
                b_dev = b.to(device=dev_obj)
                output = torch.matmul(a_dev, b_dev)
                
            with allure.step("验证输出"):
                expected = torch.matmul(a, b)
                assert output.shape == expected.shape, f"输出形状不符合预期: 期望 {expected.shape}, 实际 {output.shape}"
                assert torch.allclose(output, expected, rtol=1e-5), "输出结果不正确"
        
        elif device == "cuda":
            # 在CPU上运行
            cpu_output = torch.matmul(a, b)
            
            # 在CUDA上运行
            with allure.step("执行大规模矩阵乘法 - 设备: cuda"):
                cuda_a = a.cuda()
                cuda_b = b.cuda()
                cuda_output = torch.matmul(cuda_a, cuda_b)
                
            with allure.step("验证输出"):
                assert cuda_output.shape == cpu_output.shape, "CUDA和CPU的输出形状不一致"
                
            with allure.step("比较CPU和CUDA结果"):
                cuda_output_cpu = cuda_output.cpu()
                max_diff = torch.max(torch.abs(cpu_output - cuda_output_cpu))
                assert max_diff < 1e-4, f"CPU和CUDA结果不一致，最大差异: {max_diff}"

    @allure.story("边界条件测试")
    @allure.title("测试批量矩阵乘法")
    def test_matmul_batch(self, device):
        # 准备批量测试数据
        dtype = torch.float32
        # 创建两个3D张量，第一维是批量维度
        a = torch.randn((3, 2, 4), dtype=dtype)  # shape: (batch=3, m=2, k=4)
        b = torch.randn((3, 4, 3), dtype=dtype)  # shape: (batch=3, k=4, n=3)
        # 结果shape应该是: (3, 2, 3)
        
        if device == "cpu":
            dev_obj = get_device_object("cpu")
            with allure.step("执行批量矩阵乘法 - 设备: cpu"):
                a_dev = a.to(device=dev_obj)
                b_dev = b.to(device=dev_obj)
                output = torch.matmul(a_dev, b_dev)
                
            with allure.step("验证输出"):
                expected = torch.matmul(a, b)
                expected_shape = (3, 2, 3)
                assert output.shape == expected_shape, f"输出形状不符合预期: 期望 {expected_shape}, 实际 {output.shape}"
                assert torch.allclose(output, expected, rtol=1e-5), "输出结果不正确"
        
        elif device == "cuda":
            # 在CPU上运行
            cpu_output = torch.matmul(a, b)
            
            # 在CUDA上运行
            with allure.step("执行批量矩阵乘法 - 设备: cuda"):
                cuda_a = a.cuda()
                cuda_b = b.cuda()
                cuda_output = torch.matmul(cuda_a, cuda_b)
                
            with allure.step("验证输出"):
                assert cuda_output.shape == cpu_output.shape, "CUDA和CPU的输出形状不一致"
                
            with allure.step("比较CPU和CUDA结果"):
                cuda_output_cpu = cuda_output.cpu()
                max_diff = torch.max(torch.abs(cpu_output - cuda_output_cpu))
                assert max_diff < 1e-5, f"CPU和CUDA结果不一致，最大差异: {max_diff}"
