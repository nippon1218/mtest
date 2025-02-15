#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import torch
import allure
import math
from .utils import get_device_object, test_dtypes

def rmsnorm(x, weight=None, eps=1e-5):
    """实现RMSNorm（Root Mean Square Layer Normalization）
    
    Args:
        x: 输入张量
        weight: 可选的缩放参数
        eps: 用于数值稳定性的小值
    
    Returns:
        归一化后的张量
    """
    # 计算均方根
    norm_dims = tuple(range(1, x.dim()))  # 在除了batch维度外的所有维度上计算
    variance = torch.mean(x * x, dim=norm_dims, keepdim=True)
    rms = torch.sqrt(variance + eps)
    
    # 归一化
    x_normed = x / rms
    
    # 如果提供了weight参数，应用缩放
    if weight is not None:
        while weight.dim() < x.dim():
            weight = weight.unsqueeze(0)
        x_normed = x_normed * weight
        
    return x_normed

@allure.epic("PyTorch算子测试")
@allure.feature("RMSNorm算子")
@pytest.mark.order(7)
class TestRMSNorm:
    def setup_method(self, method):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
    def teardown_method(self, method):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    @allure.story("基础功能测试")
    @allure.title("测试不同数据类型的RMSNorm")
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_rmsnorm_basic(self, device, dtype):
        # 准备测试数据
        x = torch.tensor([
            [1.0, -2.0, 3.0],
            [-4.0, 5.0, -6.0]
        ], dtype=dtype)
        weight = torch.ones(3, dtype=dtype)
        
        if device == "cpu":
            dev_obj = get_device_object("cpu")
            with allure.step(f"执行RMSNorm - 设备: cpu, 数据类型: {dtype}"):
                x_dev = x.to(device=dev_obj)
                weight_dev = weight.to(device=dev_obj)
                output = rmsnorm(x_dev, weight_dev)
                
            with allure.step("验证输出"):
                # 验证形状
                assert output.shape == x.shape, f"输出形状不符合预期: 期望 {x.shape}, 实际 {output.shape}"
                assert output.dtype == dtype, f"输出数据类型不符合预期: 期望 {dtype}, 实际 {output.dtype}"
                
                # 验证均方根是否接近1
                rms = torch.sqrt(torch.mean(output * output, dim=1))
                assert torch.allclose(rms, torch.ones_like(rms), rtol=1e-5), "归一化后的均方根应接近1"
        
        elif device == "cuda":
            # 在CPU上运行
            cpu_output = rmsnorm(x, weight)
            
            # 在CUDA上运行
            with allure.step(f"执行RMSNorm - 设备: cuda, 数据类型: {dtype}"):
                cuda_x = x.cuda()
                cuda_weight = weight.cuda()
                cuda_output = rmsnorm(cuda_x, cuda_weight)
                
            with allure.step("验证输出"):
                assert cuda_output.shape == cpu_output.shape, "CUDA和CPU的输出形状不一致"
                
            with allure.step("比较CPU和CUDA结果"):
                cuda_output_cpu = cuda_output.cpu()
                max_diff = torch.max(torch.abs(cpu_output - cuda_output_cpu))
                assert max_diff < 1e-5, f"CPU和CUDA结果不一致，最大差异: {max_diff}"

    @allure.story("边界条件测试")
    @allure.title("测试RMSNorm的边界值")
    def test_rmsnorm_edge_cases(self, device):
        # 准备测试数据
        dtype = torch.float32
        # 测试不同量级的输入
        x_normal = torch.tensor([
            [1e-6, 1e-6, 1e-6],  # 非常小的值
            [1e6, 1e6, 1e6],     # 非常大的值
            [0, 0, 0]            # 全零输入
        ], dtype=dtype)
        weight = torch.ones(3, dtype=dtype)
        
        if device == "cpu":
            dev_obj = get_device_object("cpu")
            with allure.step("执行RMSNorm边界值测试 - 设备: cpu"):
                x_normal_dev = x_normal.to(device=dev_obj)
                weight_dev = weight.to(device=dev_obj)
                output = rmsnorm(x_normal_dev, weight_dev)
                
            with allure.step("验证输出"):
                # 验证非零输入的归一化结果
                for i in range(2):
                    # 当输入全部相同时，输出应该也全部相同
                    if torch.allclose(x_normal[i], x_normal[i][0].expand_as(x_normal[i])):
                        assert torch.allclose(output[i], output[i][0].expand_as(output[i])), \
                            f"第{i}行的输出应该全部相同"
                    else:
                        assert torch.std(output[i]) > 0, f"第{i}行的输出不应该全为0"
                
                # 验证全零输入的处理（应该全为0）
                assert torch.allclose(output[2], torch.zeros(3), atol=1e-6), "全零输入的输出应接近0"
        
        elif device == "cuda":
            # 在CPU上运行
            cpu_output = rmsnorm(x_normal, weight)
            
            # 在CUDA上运行
            with allure.step("执行RMSNorm边界值测试 - 设备: cuda"):
                cuda_x = x_normal.cuda()
                cuda_weight = weight.cuda()
                cuda_output = rmsnorm(cuda_x, cuda_weight)
                
            with allure.step("验证输出"):
                assert cuda_output.shape == cpu_output.shape, "CUDA和CPU的输出形状不一致"
                
            with allure.step("比较CPU和CUDA结果"):
                cuda_output_cpu = cuda_output.cpu()
                max_diff = torch.max(torch.abs(cpu_output - cuda_output_cpu))
                assert max_diff < 1e-5, f"CPU和CUDA结果不一致，最大差异: {max_diff}"

    @allure.story("边界条件测试")
    @allure.title("测试大规模RMSNorm")
    def test_rmsnorm_performance(self, device):
        # 准备大规模测试数据
        dtype = torch.float32
        batch_size = 32
        seq_len = 512
        hidden_size = 1024
        
        x = torch.randn(batch_size, seq_len, hidden_size, dtype=dtype)
        weight = torch.ones(hidden_size, dtype=dtype)
        
        if device == "cpu":
            dev_obj = get_device_object("cpu")
            with allure.step("执行大规模RMSNorm - 设备: cpu"):
                x_dev = x.to(device=dev_obj)
                weight_dev = weight.to(device=dev_obj)
                output = rmsnorm(x_dev, weight_dev)
                
            with allure.step("验证输出"):
                assert output.shape == x.shape, f"输出形状不符合预期: 期望 {x.shape}, 实际 {output.shape}"
                # 验证归一化后的标准差
                std = torch.std(output, dim=-1)
                assert torch.all(std > 0), "归一化后的输出不应该全为0"
        
        elif device == "cuda":
            # 在CPU上运行
            cpu_output = rmsnorm(x, weight)
            
            # 在CUDA上运行
            with allure.step("执行大规模RMSNorm - 设备: cuda"):
                cuda_x = x.cuda()
                cuda_weight = weight.cuda()
                cuda_output = rmsnorm(cuda_x, cuda_weight)
                
            with allure.step("验证输出"):
                assert cuda_output.shape == cpu_output.shape, "CUDA和CPU的输出形状不一致"
                
            with allure.step("比较CPU和CUDA结果"):
                cuda_output_cpu = cuda_output.cpu()
                max_diff = torch.max(torch.abs(cpu_output - cuda_output_cpu))
                assert max_diff < 1e-4, f"CPU和CUDA结果不一致，最大差异: {max_diff}"

    @allure.story("基础功能测试")
    @allure.title("测试不同维度的RMSNorm")
    def test_rmsnorm_dimensions(self, device):
        # 准备不同维度的测试数据
        dtype = torch.float32
        test_cases = [
            ((2, 3), torch.ones(3)),           # 2D
            ((2, 3, 4), torch.ones(4)),        # 3D
            ((2, 3, 4, 5), torch.ones(5))      # 4D
        ]
        
        for x_shape, weight in test_cases:
            x = torch.randn(*x_shape, dtype=dtype)
            
            if device == "cpu":
                dev_obj = get_device_object("cpu")
                with allure.step(f"执行{len(x_shape)}D RMSNorm - 设备: cpu"):
                    x_dev = x.to(device=dev_obj)
                    weight_dev = weight.to(device=dev_obj)
                    output = rmsnorm(x_dev, weight_dev)
                    
                with allure.step("验证输出"):
                    assert output.shape == x.shape, f"输出形状不符合预期: 期望 {x.shape}, 实际 {output.shape}"
                    # 验证归一化后的标准差
                    std = torch.std(output, dim=-1)
                    assert torch.all(std > 0), "归一化后的输出不应该全为0"
            
            elif device == "cuda":
                # 在CPU上运行
                cpu_output = rmsnorm(x, weight)
                
                # 在CUDA上运行
                with allure.step(f"执行{len(x_shape)}D RMSNorm - 设备: cuda"):
                    cuda_x = x.cuda()
                    cuda_weight = weight.cuda()
                    cuda_output = rmsnorm(cuda_x, cuda_weight)
                    
                with allure.step("验证输出"):
                    assert cuda_output.shape == cpu_output.shape, "CUDA和CPU的输出形状不一致"
                    
                with allure.step("比较CPU和CUDA结果"):
                    cuda_output_cpu = cuda_output.cpu()
                    max_diff = torch.max(torch.abs(cpu_output - cuda_output_cpu))
                    assert max_diff < 1e-5, f"CPU和CUDA结果不一致，最大差异: {max_diff}"
