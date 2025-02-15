#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import torch
import allure
import numpy as np
from .utils import get_device_object, test_dtypes

@allure.epic("PyTorch算子测试")
@allure.feature("Mul算子")
@pytest.mark.order(2)
class TestMul:
    def teardown_method(self, method):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    @allure.story("基础功能测试")
    @allure.title("测试不同数据类型的标量乘法")
    @pytest.mark.parametrize("dtype", test_dtypes)
    def test_mul_scalar(self, device, dtype):
        # 准备测试数据
        a = torch.tensor([[1, 2], [3, 4]], dtype=dtype)
        scalar = 2
        
        if device == "cpu":
            dev_obj = get_device_object("cpu")
            with allure.step(f"执行标量乘法 - 设备: cpu, 数据类型: {dtype}"):
                a_dev = a.to(device=dev_obj)
                output = a_dev * scalar
                
            with allure.step("验证输出"):
                expected = a * scalar
                assert output.shape == expected.shape, f"输出形状不符合预期: 期望 {expected.shape}, 实际 {output.shape}"
                assert output.dtype == expected.dtype, f"输出数据类型不符合预期: 期望 {expected.dtype}, 实际 {output.dtype}"
                assert torch.all(output == expected), "输出结果不正确"
        
        elif device == "cuda":
            # 在CPU上运行
            cpu_output = a * scalar
            
            # 在CUDA上运行
            with allure.step(f"执行标量乘法 - 设备: cuda, 数据类型: {dtype}"):
                cuda_a = a.cuda()
                cuda_output = cuda_a * scalar
                
            with allure.step("验证输出"):
                assert cuda_output.shape == cpu_output.shape, "CUDA和CPU的输出形状不一致"
                
            with allure.step("比较CPU和CUDA结果"):
                cuda_output_cpu = cuda_output.cpu()
                max_diff = torch.max(torch.abs(cpu_output - cuda_output_cpu))
                assert max_diff < 1e-5, f"CPU和CUDA结果不一致，最大差异: {max_diff}"

    @allure.story("基础功能测试")
    @allure.title("测试不同数据类型的张量乘法")
    @pytest.mark.parametrize("dtype", test_dtypes)
    def test_mul_tensor(self, device, dtype):
        # 准备测试数据
        a = torch.tensor([[1, 2], [3, 4]], dtype=dtype)
        b = torch.tensor([[5, 6], [7, 8]], dtype=dtype)
        
        if device == "cpu":
            dev_obj = get_device_object("cpu")
            with allure.step(f"执行张量乘法 - 设备: cpu, 数据类型: {dtype}"):
                a_dev = a.to(device=dev_obj)
                b_dev = b.to(device=dev_obj)
                output = a_dev * b_dev
                
            with allure.step("验证输出"):
                expected = a * b
                assert output.shape == expected.shape, f"输出形状不符合预期: 期望 {expected.shape}, 实际 {output.shape}"
                assert output.dtype == expected.dtype, f"输出数据类型不符合预期: 期望 {expected.dtype}, 实际 {output.dtype}"
                assert torch.all(output == expected), "输出结果不正确"
        
        elif device == "cuda":
            # 在CPU上运行
            cpu_output = a * b
            
            # 在CUDA上运行
            with allure.step(f"执行张量乘法 - 设备: cuda, 数据类型: {dtype}"):
                cuda_a = a.cuda()
                cuda_b = b.cuda()
                cuda_output = cuda_a * cuda_b
                
            with allure.step("验证输出"):
                assert cuda_output.shape == cpu_output.shape, "CUDA和CPU的输出形状不一致"
                
            with allure.step("比较CPU和CUDA结果"):
                cuda_output_cpu = cuda_output.cpu()
                max_diff = torch.max(torch.abs(cpu_output - cuda_output_cpu))
                assert max_diff < 1e-5, f"CPU和CUDA结果不一致，最大差异: {max_diff}"

    @allure.story("边界条件测试")
    @allure.title("测试大规模乘法")
    def test_mul_performance(self, device):
        # 准备大规模测试数据
        dtype = torch.float32
        a = torch.randn((1000, 1000), dtype=dtype)
        b = torch.randn((1000, 1000), dtype=dtype)
        
        if device == "cpu":
            dev_obj = get_device_object("cpu")
            with allure.step("执行大规模乘法 - 设备: cpu"):
                a_dev = a.to(device=dev_obj)
                b_dev = b.to(device=dev_obj)
                output = a_dev * b_dev
                
            with allure.step("验证输出"):
                expected = a * b
                assert output.shape == expected.shape, f"输出形状不符合预期: 期望 {expected.shape}, 实际 {output.shape}"
                assert torch.allclose(output, expected, rtol=1e-5), "输出结果不正确"
        
        elif device == "cuda":
            # 在CPU上运行
            cpu_output = a * b
            
            # 在CUDA上运行
            with allure.step("执行大规模乘法 - 设备: cuda"):
                cuda_a = a.cuda()
                cuda_b = b.cuda()
                cuda_output = cuda_a * cuda_b
                
            with allure.step("验证输出"):
                assert cuda_output.shape == cpu_output.shape, "CUDA和CPU的输出形状不一致"
                
            with allure.step("比较CPU和CUDA结果"):
                cuda_output_cpu = cuda_output.cpu()
                max_diff = torch.max(torch.abs(cpu_output - cuda_output_cpu))
                assert max_diff < 1e-5, f"CPU和CUDA结果不一致，最大差异: {max_diff}"

    @allure.story("边界条件测试")
    @allure.title("测试广播乘法")
    def test_mul_broadcast(self, device):
        # 准备测试数据
        dtype = torch.float32
        a = torch.randn((3, 1, 4), dtype=dtype)  # shape: (3, 1, 4)
        b = torch.randn((2, 4), dtype=dtype)     # shape: (2, 4)
        # 结果shape应该是: (3, 2, 4)
        
        if device == "cpu":
            dev_obj = get_device_object("cpu")
            with allure.step("执行广播乘法 - 设备: cpu"):
                a_dev = a.to(device=dev_obj)
                b_dev = b.to(device=dev_obj)
                output = a_dev * b_dev
                
            with allure.step("验证输出"):
                expected = a * b
                expected_shape = (3, 2, 4)
                assert output.shape == expected_shape, f"输出形状不符合预期: 期望 {expected_shape}, 实际 {output.shape}"
                assert torch.allclose(output, expected, rtol=1e-5), "输出结果不正确"
        
        elif device == "cuda":
            # 在CPU上运行
            cpu_output = a * b
            
            # 在CUDA上运行
            with allure.step("执行广播乘法 - 设备: cuda"):
                cuda_a = a.cuda()
                cuda_b = b.cuda()
                cuda_output = cuda_a * cuda_b
                
            with allure.step("验证输出"):
                assert cuda_output.shape == cpu_output.shape, "CUDA和CPU的输出形状不一致"
                
            with allure.step("比较CPU和CUDA结果"):
                cuda_output_cpu = cuda_output.cpu()
                max_diff = torch.max(torch.abs(cpu_output - cuda_output_cpu))
                assert max_diff < 1e-5, f"CPU和CUDA结果不一致，最大差异: {max_diff}"
