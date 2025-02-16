#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import torch
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
@pytest.mark.order(8)
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
        # 准备测试数据
        x = torch.tensor([
            [1, 2, 3],
            [4, 5, 6]
        ], dtype=dtype)
        
        if device == "cpu":
            dev_obj = get_device_object("cpu")
            with allure.step(f"执行Transpose - 设备: cpu, 数据类型: {dtype}"):
                x_dev = x.to(device=dev_obj)
                output = torch.transpose(x_dev, 0, 1)
                
            with allure.step("验证输出"):
                # 验证形状
                assert output.shape == (3, 2), f"输出形状不符合预期: 期望 (3, 2), 实际 {output.shape}"
                assert output.dtype == dtype, f"输出数据类型不符合预期: 期望 {dtype}, 实际 {output.dtype}"
                
                # 验证转置是否正确
                expected = torch.tensor([
                    [1, 4],
                    [2, 5],
                    [3, 6]
                ], dtype=dtype)
                assert torch.all(output == expected), "转置结果不正确"
        
        elif device == "cuda":
            # 在CPU上运行
            cpu_output = torch.transpose(x, 0, 1)
            
            # 在CUDA上运行
            with allure.step(f"执行Transpose - 设备: cuda, 数据类型: {dtype}"):
                cuda_x = x.cuda()
                cuda_output = torch.transpose(cuda_x, 0, 1)
                
            with allure.step("验证输出"):
                assert cuda_output.shape == cpu_output.shape, "CUDA和CPU的输出形状不一致"
                
            with allure.step("比较CPU和CUDA结果"):
                cuda_output_cpu = cuda_output.cpu()
                assert torch.all(cpu_output == cuda_output_cpu), "CPU和CUDA结果不一致"

    @allure.story("边界条件测试")
    @allure.title("测试特殊情况的Transpose")
    @allure.description("""
    验证特殊情况下的转置操作，测试要点：
    1. 单维度张量的转置
    2. 全零张量的转置
    3. 单元素张量的转置
    4. 比较CPU和CUDA的计算结果
    """)
    def test_transpose_edge_cases(self, device):
        # 准备测试数据
        dtype = torch.float32
        
        # 测试1：单维度张量
        x1 = torch.tensor([1, 2, 3], dtype=dtype)
        
        # 测试2：全零张量
        x2 = torch.zeros((2, 3, 4), dtype=dtype)
        
        # 测试3：单元素张量
        x3 = torch.tensor([[1]], dtype=dtype)
        
        if device == "cpu":
            dev_obj = get_device_object("cpu")
            with allure.step("执行Transpose边界值测试 - 设备: cpu"):
                # 测试1：单维度张量
                x1_dev = x1.to(device=dev_obj)
                output1 = torch.transpose(x1_dev, 0, 0)
                assert torch.all(output1 == x1_dev), "单维度张量转置应该不变"
                
                # 测试2：全零张量
                x2_dev = x2.to(device=dev_obj)
                output2 = torch.transpose(x2_dev, 1, 2)
                assert output2.shape == (2, 4, 3), "全零张量转置形状不正确"
                assert torch.all(output2 == 0), "全零张量转置后应该仍然全为0"
                
                # 测试3：单元素张量
                x3_dev = x3.to(device=dev_obj)
                output3 = torch.transpose(x3_dev, 0, 1)
                assert output3.shape == (1, 1), "单元素张量转置形状不正确"
                assert output3[0, 0] == x3[0, 0], "单元素张量转置值不正确"
        
        elif device == "cuda":
            # 测试1：单维度张量
            cpu_output1 = torch.transpose(x1, 0, 0)
            cuda_x1 = x1.cuda()
            cuda_output1 = torch.transpose(cuda_x1, 0, 0)
            assert torch.all(cpu_output1 == cuda_output1.cpu()), "单维度张量CUDA和CPU结果不一致"
            
            # 测试2：全零张量
            cpu_output2 = torch.transpose(x2, 1, 2)
            cuda_x2 = x2.cuda()
            cuda_output2 = torch.transpose(cuda_x2, 1, 2)
            assert torch.all(cpu_output2 == cuda_output2.cpu()), "全零张量CUDA和CPU结果不一致"
            
            # 测试3：单元素张量
            cpu_output3 = torch.transpose(x3, 0, 1)
            cuda_x3 = x3.cuda()
            cuda_output3 = torch.transpose(cuda_x3, 0, 1)
            assert torch.all(cpu_output3 == cuda_output3.cpu()), "单元素张量CUDA和CPU结果不一致"

    @allure.story("性能测试")
    @allure.title("测试大规模Transpose")
    @allure.description("""
    验证大规模数据的转置操作，测试要点：
    1. 处理大规模数据（batch=32, seq_len=512, hidden_size=1024）
    2. 验证转置的准确性
    3. 验证输出形状的一致性
    4. 比较CPU和CUDA的计算结果
    """)
    def test_transpose_performance(self, device):
        # 准备大规模测试数据
        dtype = torch.float32
        batch_size = 32
        seq_len = 512
        hidden_size = 1024
        
        x = torch.randn(batch_size, seq_len, hidden_size, dtype=dtype)
        
        if device == "cpu":
            dev_obj = get_device_object("cpu")
            with allure.step("执行大规模Transpose - 设备: cpu"):
                x_dev = x.to(device=dev_obj)
                output = torch.transpose(x_dev, 1, 2)
                
            with allure.step("验证输出"):
                assert output.shape == (batch_size, hidden_size, seq_len), \
                    f"输出形状不符合预期: 期望 {(batch_size, hidden_size, seq_len)}, 实际 {output.shape}"
                
                # 验证转置是否正确
                for i in range(min(5, batch_size)):  # 只检查前5个batch以节省时间
                    for j in range(min(5, seq_len)):
                        for k in range(min(5, hidden_size)):
                            assert x[i, j, k] == output[i, k, j], \
                                f"位置 ({i}, {j}, {k}) 的转置结果不正确"
        
        elif device == "cuda":
            # 在CPU上运行
            cpu_output = torch.transpose(x, 1, 2)
            
            # 在CUDA上运行
            with allure.step("执行大规模Transpose - 设备: cuda"):
                cuda_x = x.cuda()
                cuda_output = torch.transpose(cuda_x, 1, 2)
                
            with allure.step("验证输出"):
                assert cuda_output.shape == cpu_output.shape, "CUDA和CPU的输出形状不一致"
                
            with allure.step("比较CPU和CUDA结果"):
                # 只比较部分结果以节省内存
                sample_indices = torch.randint(0, batch_size, (5,))
                for idx in sample_indices:
                    assert torch.all(cpu_output[idx] == cuda_output[idx].cpu()), \
                        f"第{idx}个batch的CPU和CUDA结果不一致"

    @allure.story("多维度测试")
    @allure.title("测试不同维度组合的Transpose")
    @allure.description("""
    验证不同维度组合的转置操作，测试要点：
    1. 2D到2D的转置
    2. 3D到3D的转置
    3. 4D到4D的转置
    4. 验证不同维度下的结果准确性
    """)
    def test_transpose_dimensions(self, device):
        # 准备不同维度的测试数据
        dtype = torch.float32
        test_cases = [
            # (输入形状, 转置维度1, 转置维度2, 期望输出形状)
            ((2, 3), 0, 1, (3, 2)),
            ((2, 3, 4), 0, 2, (4, 3, 2)),
            ((2, 3, 4, 5), 1, 3, (2, 5, 4, 3)),
        ]
        
        for input_shape, dim1, dim2, expected_shape in test_cases:
            x = torch.randn(*input_shape, dtype=dtype)
            
            if device == "cpu":
                dev_obj = get_device_object("cpu")
                with allure.step(f"执行{len(input_shape)}D Transpose - 设备: cpu"):
                    x_dev = x.to(device=dev_obj)
                    output = torch.transpose(x_dev, dim1, dim2)
                    
                with allure.step("验证输出"):
                    assert output.shape == expected_shape, \
                        f"输出形状不符合预期: 期望 {expected_shape}, 实际 {output.shape}"
                    
                    # 验证转置后的内容
                    x_np = x.numpy()
                    output_np = output.numpy()
                    expected_np = np.transpose(x_np, [dim2 if i == dim1 else dim1 if i == dim2 else i 
                                                    for i in range(len(input_shape))])
                    assert np.allclose(output_np, expected_np), "转置结果不正确"
            
            elif device == "cuda":
                # 在CPU上运行
                cpu_output = torch.transpose(x, dim1, dim2)
                
                # 在CUDA上运行
                with allure.step(f"执行{len(input_shape)}D Transpose - 设备: cuda"):
                    cuda_x = x.cuda()
                    cuda_output = torch.transpose(cuda_x, dim1, dim2)
                    
                with allure.step("验证输出"):
                    assert cuda_output.shape == cpu_output.shape, "CUDA和CPU的输出形状不一致"
                    
                with allure.step("比较CPU和CUDA结果"):
                    cuda_output_cpu = cuda_output.cpu()
                    assert torch.all(cpu_output == cuda_output_cpu), "CPU和CUDA结果不一致"
