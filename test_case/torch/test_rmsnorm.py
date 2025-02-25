#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import torch
import allure
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
@allure.description("""
该测试模块验证PyTorch中RMSNorm（Root Mean Square Layer Normalization）算子的功能正确性，包括：
1. 基本功能：验证不同数据类型和维度的RMSNorm计算
2. 边界情况：验证数值稳定性，包括处理大值、小值和零值
3. 性能测试：验证大规模数据的计算
4. 设备兼容性：验证在CPU和CUDA设备上的一致性

所有测试都在CPU和CUDA设备上执行，并验证结果的一致性。
""")
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
    @allure.description("""
    验证RMSNorm的基本功能，测试要点：
    1. 支持float32和float64数据类型
    2. 验证归一化后的均方根是否接近1
    3. 验证输出形状和数据类型的正确性
    4. 比较CPU和CUDA结果的一致性
    """)
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_rmsnorm_basic(self, device, dtype):
        device_obj = get_device_object(device)
        
        # 准备测试数据
        x = torch.tensor([
            [1.0, -2.0, 3.0],
            [-4.0, 5.0, -6.0]
        ], dtype=dtype)
        weight = torch.ones(3, dtype=dtype)
        
        # 在CPU上计算参考结果
        x_cpu = x.clone()
        weight_cpu = weight.clone()
        output_cpu = rmsnorm(x_cpu, weight_cpu)
        
        # 在指定设备上计算
        with allure.step(f"执行RMSNorm - 设备: {device}, 数据类型: {dtype}"):
            x_dev = x.to(device=device_obj)
            weight_dev = weight.to(device=device_obj)
            output = rmsnorm(x_dev, weight_dev)
        
        with allure.step("验证输出"):
            # 验证形状
            assert output.shape == x.shape, f"输出形状不符合预期: 期望 {x.shape}, 实际 {output.shape}"
            assert output.dtype == dtype, f"输出数据类型不符合预期: 期望 {dtype}, 实际 {output.dtype}"
            
            # 验证均方根是否接近1
            rms = torch.sqrt(torch.mean(output * output, dim=1))
            assert torch.allclose(rms, torch.ones_like(rms), rtol=1e-5), "归一化后的均方根应接近1"
        
        # 如果在CUDA设备上运行，比较与CPU结果的一致性
        if device == "cuda":
            with allure.step("比较CPU和CUDA结果"):
                output_cpu_compare = output.cpu()
                torch.testing.assert_close(output_cpu_compare, output_cpu, rtol=1e-4, atol=1e-4)

    @allure.story("基础功能测试")
    @allure.title("测试不同维度的RMSNorm")
    @allure.description("""
    验证RMSNorm在不同维度输入上的表现，测试要点：
    1. 支持2D、3D、4D等不同维度的输入
    2. 验证各维度下输出形状的正确性
    3. 验证归一化后的标准差
    4. 比较CPU和CUDA结果的一致性
    """)
    def test_rmsnorm_dimensions(self, device):
        device_obj = get_device_object(device)
        
        # 准备不同维度的测试数据
        dtype = torch.float32
        test_cases = [
            ((2, 3), torch.ones(3)),           # 2D
            ((2, 3, 4), torch.ones(4)),        # 3D
            ((2, 3, 4, 5), torch.ones(5))      # 4D
        ]
        
        for x_shape, weight in test_cases:
            x = torch.randn(*x_shape, dtype=dtype)
            
            # 在CPU上计算参考结果
            x_cpu = x.clone()
            weight_cpu = weight.clone()
            output_cpu = rmsnorm(x_cpu, weight_cpu)
            
            # 在指定设备上计算
            with allure.step(f"执行{len(x_shape)}D RMSNorm - 设备: {device}"):
                x_dev = x.to(device=device_obj)
                weight_dev = weight.to(device=device_obj)
                output = rmsnorm(x_dev, weight_dev)
            
            with allure.step("验证输出"):
                assert output.shape == x.shape, f"输出形状不符合预期: 期望 {x.shape}, 实际 {output.shape}"
                # 验证归一化后的标准差
                std = torch.std(output, dim=-1)
                assert torch.all(std > 0), "归一化后的输出不应该全为0"
            
            # 如果在CUDA设备上运行，比较与CPU结果的一致性
            if device == "cuda":
                with allure.step("比较CPU和CUDA结果"):
                    output_cpu_compare = output.cpu()
                    torch.testing.assert_close(output_cpu_compare, output_cpu, rtol=1e-4, atol=1e-4)

    @allure.story("边界条件测试")
    @allure.title("测试RMSNorm的边界值")
    @allure.description("""
    验证RMSNorm在处理边界值时的稳定性，测试要点：
    1. 处理非常小的值（1e-6）
    2. 处理非常大的值（1e6）
    3. 处理全零输入
    4. 验证输出的数值稳定性
    5. 比较CPU和CUDA结果的一致性
    """)
    def test_rmsnorm_edge_cases(self, device):
        device_obj = get_device_object(device)
        
        # 准备测试数据
        dtype = torch.float32
        # 测试不同量级的输入
        x_normal = torch.tensor([
            [1e-6, 1e-6, 1e-6],  # 非常小的值
            [1e6, 1e6, 1e6],     # 非常大的值
            [0, 0, 0]            # 全零输入
        ], dtype=dtype)
        weight = torch.ones(3, dtype=dtype)
        
        # 在CPU上计算参考结果
        x_cpu = x_normal.clone()
        weight_cpu = weight.clone()
        output_cpu = rmsnorm(x_cpu, weight_cpu)
        
        # 在指定设备上计算
        with allure.step(f"执行RMSNorm边界值测试 - 设备: {device}"):
            x_dev = x_normal.to(device=device_obj)
            weight_dev = weight.to(device=device_obj)
            output = rmsnorm(x_dev, weight_dev)
        
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
            assert torch.allclose(output[2], torch.zeros(3, device=device_obj), atol=1e-6), "全零输入的输出应接近0"
        
        # 如果在CUDA设备上运行，比较与CPU结果的一致性
        if device == "cuda":
            with allure.step("比较CPU和CUDA结果"):
                output_cpu_compare = output.cpu()
                torch.testing.assert_close(output_cpu_compare, output_cpu, rtol=1e-4, atol=1e-4)

    @allure.story("边界条件测试")
    @allure.title("测试大规模RMSNorm")
    @allure.description("""
    验证RMSNorm在处理大规模数据时的表现，测试要点：
    1. 处理大规模张量（batch_size=32, seq_len=512, hidden_size=1024）
    2. 验证输出形状的正确性
    3. 验证归一化后的标准差
    4. 比较CPU和CUDA结果的一致性
    """)
    def test_rmsnorm_performance(self, device):
        device_obj = get_device_object(device)
        
        # 准备大规模测试数据
        dtype = torch.float32
        batch_size = 32
        seq_len = 512
        hidden_size = 1024
        
        x = torch.randn(batch_size, seq_len, hidden_size, dtype=dtype)
        weight = torch.ones(hidden_size, dtype=dtype)
        
        # 在CPU上计算参考结果
        x_cpu = x.clone()
        weight_cpu = weight.clone()
        output_cpu = rmsnorm(x_cpu, weight_cpu)
        
        # 在指定设备上计算
        with allure.step(f"执行大规模RMSNorm - 设备: {device}"):
            x_dev = x.to(device=device_obj)
            weight_dev = weight.to(device=device_obj)
            output = rmsnorm(x_dev, weight_dev)
        
        with allure.step("验证输出"):
            assert output.shape == x.shape, f"输出形状不符合预期: 期望 {x.shape}, 实际 {output.shape}"
            # 验证归一化后的标准差
            std = torch.std(output, dim=-1)
            assert torch.all(std > 0), "归一化后的输出不应该全为0"
        
        # 如果在CUDA设备上运行，比较与CPU结果的一致性
        if device == "cuda":
            with allure.step("比较CPU和CUDA结果"):
                output_cpu_compare = output.cpu()
                torch.testing.assert_close(output_cpu_compare, output_cpu, rtol=1e-4, atol=1e-4)

