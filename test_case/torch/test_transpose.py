#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
# 使用我们的导入辅助模块替代直接导入
from .torch_import import torch, torch_import_failed
import allure
import numpy as np
from .utils import get_device_object, test_dtypes

@allure.epic("PyTorch算子测试")
@allure.feature("Transpose算子")
@allure.description("""
该测试模块验证PyTorch中Transpose算子的功能正确性，包括：
1. 基本功能：验证不同数据类型的转置计算
2. 边界情况：验证特殊形状和特殊值的处理
3. 性能测试：验证大规模数据的转置
4. 多维度测试：验证不同维度组合的转置

所有测试都在CPU和CUDA设备上执行，并验证结果的一致性。
""")
class TestTranspose:
    def setup_method(self, method):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
    def teardown_method(self, method):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    @allure.story("基础功能测试")
    @allure.title("测试不同数据类型的Transpose")
    @allure.description("""
    验证基本的转置功能，测试要点：
    1. 支持多种数据类型
    2. 验证输出形状的正确性
    3. 验证转置结果的准确性
    4. 比较CPU和CUDA结果的一致性
    """)
    @pytest.mark.parametrize("dtype", test_dtypes)
    def test_transpose_basic(self, device, dtype):
        device_obj = get_device_object(device)
        
        # 准备测试数据
        x = torch.tensor([
            [1, 2, 3],
            [4, 5, 6]
        ], dtype=dtype).to(device_obj)
        
        with allure.step(f"执行Transpose - 设备: {device}, 数据类型: {dtype}"):
            output = torch.transpose(x, 0, 1)
            
        with allure.step("验证输出"):
            # 验证形状
            assert output.shape == (3, 2), f"输出形状不符合预期: 期望 (3, 2), 实际 {output.shape}"
            assert output.dtype == dtype, f"输出数据类型不符合预期: 期望 {dtype}, 实际 {output.dtype}"
            
            # 验证转置是否正确
            expected = torch.tensor([
                [1, 4],
                [2, 5],
                [3, 6]
            ], dtype=dtype).to(device_obj)
            assert torch.all(output == expected), "转置结果不正确"
        
        # 与CPU结果比较（仅当使用CUDA设备时）
        if device == "cuda":
            with allure.step("比较CPU和CUDA结果"):
                x_cpu = x.cpu()
                cpu_output = torch.transpose(x_cpu, 0, 1)
                cuda_output_cpu = output.cpu()
                assert torch.all(cpu_output == cuda_output_cpu), "CPU和CUDA结果不一致"

    @allure.story("边界条件测试")
    @allure.title("测试特殊情况的Transpose")
    @allure.description("""
    验证特殊情况下的转置操作，测试要点：
    1. 转置空张量
    2. 转置相同维度
    3. 处理全零张量
    """)
    def test_transpose_edge_cases(self, device):
        device_obj = get_device_object(device)
        
        # 测试用例列表
        test_cases = [
            ("空张量", torch.zeros((0, 3), device=device_obj), 0, 1),
            ("转置相同维度", torch.ones((2, 3), device=device_obj), 1, 1),
            ("全零张量", torch.zeros((5, 5), device=device_obj), 0, 1)
        ]
        
        for case_name, tensor, dim1, dim2 in test_cases:
            with allure.step(f"测试 {case_name} - 设备: {device}"):
                # 保存原始形状以供检查
                original_shape = tensor.shape
                
                # 执行转置
                output = torch.transpose(tensor, dim1, dim2)
                
                # 验证结果
                if dim1 == dim2:
                    # 如果转置相同维度，形状应该保持不变
                    assert output.shape == original_shape, f"相同维度转置后形状应保持不变: 期望 {original_shape}, 实际 {output.shape}"
                else:
                    # 否则，dim1和dim2应该交换
                    expected_shape = list(original_shape)
                    if dim1 < len(expected_shape) and dim2 < len(expected_shape):
                        expected_shape[dim1], expected_shape[dim2] = expected_shape[dim2], expected_shape[dim1]
                    expected_shape = tuple(expected_shape)
                    assert output.shape == expected_shape, f"转置后形状不符合预期: 期望 {expected_shape}, 实际 {output.shape}"
                
        # 测试一维张量
        one_dim = torch.tensor([1, 2, 3], device=device_obj)
        with allure.step(f"测试一维张量转置 - 设备: {device}"):
            # 一维张量的转置不改变形状
            output = torch.transpose(one_dim, 0, 0)
            assert output.shape == one_dim.shape, "一维张量转置后形状应保持不变"
            assert torch.all(output == one_dim), "一维张量转置后内容应保持不变"

    @allure.story("性能测试")
    @allure.title("测试Transpose性能")
    @allure.description("""
    测试不同大小张量的转置性能，测试要点：
    1. 小规模张量转置（10x10）
    2. 中规模张量转置（100x100）
    3. 大规模张量转置（1000x1000）
    4. 测量执行时间
    5. 比较CPU和CUDA的性能差异
    """)
    def test_transpose_performance(self, device):
        import time
        device_obj = get_device_object(device)
        
        # 测试不同大小的矩阵
        sizes = [(10, 10), (100, 100), (1000, 1000)]
        
        for size in sizes:
            with allure.step(f"测试 {size[0]}x{size[1]} 矩阵转置 - 设备: {device}"):
                # 创建随机张量
                x = torch.randn(*size, device=device_obj)
                
                # 同步设备并测量时间
                if device == "cuda":
                    torch.cuda.synchronize()
                start_time = time.time()
                
                # 执行多次转置操作以获得平均性能
                num_iterations = 100
                for _ in range(num_iterations):
                    output = torch.transpose(x, 0, 1)
                    if device == "cuda":
                        torch.cuda.synchronize()
                
                end_time = time.time()
                avg_time = (end_time - start_time) / num_iterations
                
                # 记录性能数据
                allure.attach(f"平均执行时间: {avg_time:.6f} 秒", name=f"{size[0]}x{size[1]} 矩阵转置时间", attachment_type=allure.attachment_type.TEXT)
                
                # 验证结果正确性
                output_cpu = output.cpu().numpy() if device == "cuda" else output.numpy()
                x_cpu = x.cpu().numpy() if device == "cuda" else x.numpy()
                expected = np.transpose(x_cpu, (1, 0))
                assert np.allclose(output_cpu, expected), "转置结果不正确"

    @allure.story("多维度测试")
    @allure.title("测试多维度Transpose")
    @allure.description("""
    测试多维张量的转置操作，测试要点：
    1. 3D张量转置
    2. 4D张量转置
    3. 5D张量转置
    4. 不同维度组合的转置
    5. 验证结果与numpy.transpose一致
    """)
    def test_transpose_multidim(self, device):
        device_obj = get_device_object(device)
        
        # 测试用例：(形状, dim1, dim2)
        test_cases = [
            ((2, 3, 4), 0, 2),  # 3D张量
            ((2, 3, 4, 5), 1, 3),  # 4D张量
            ((2, 3, 4, 5, 6), 2, 4)  # 5D张量
        ]
        
        for input_shape, dim1, dim2 in test_cases:
            with allure.step(f"测试{len(input_shape)}D Transpose - 设备: {device}, 维度: {input_shape}, 转置维度: ({dim1},{dim2})"):
                # 创建测试张量
                x = torch.randn(*input_shape, device=device_obj)
                
                # 执行转置
                output = torch.transpose(x, dim1, dim2)
                
                # 验证形状
                expected_shape = list(input_shape)
                expected_shape[dim1], expected_shape[dim2] = expected_shape[dim2], expected_shape[dim1]
                assert output.shape == tuple(expected_shape), f"转置后形状不符合预期: 期望 {expected_shape}, 实际 {output.shape}"
                
                # 验证转置后的内容（与numpy比较）
                x_np = x.cpu().numpy()
                output_np = output.cpu().numpy()
                expected_np = np.transpose(x_np, [dim2 if i == dim1 else dim1 if i == dim2 else i 
                                                for i in range(len(input_shape))])
                assert np.allclose(output_np, expected_np), "转置结果不正确"
