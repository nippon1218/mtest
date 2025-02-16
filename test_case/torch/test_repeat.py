#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import torch
import allure
import numpy as np
from .utils import get_device_object, test_dtypes

@allure.epic("PyTorch算子测试")
@allure.feature("Repeat算子")
@allure.description("""
该测试模块验证PyTorch中Repeat算子的功能正确性，包括：
1. 基本功能：验证不同数据类型的张量重复
2. 边界情况：验证特殊情况下的重复操作
3. 性能测试：验证大规模张量的重复
4. 多维度测试：验证不同维度组合的重复

所有测试都在CPU和CUDA设备上执行，并验证结果的一致性。
""")
@pytest.mark.order(9)
class TestRepeat:
    def setup_method(self, method):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
    def teardown_method(self, method):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    @allure.story("基础功能测试")
    @allure.title("测试不同数据类型的Repeat")
    @allure.description("""
    验证基本的张量重复功能，测试要点：
    1. 支持多种数据类型（float32、float64等）
    2. 验证输出形状和数据类型的正确性
    3. 验证重复结果的准确性
    4. 比较CPU和CUDA结果的一致性
    """)
    @pytest.mark.parametrize("dtype", test_dtypes)
    def test_repeat_basic(self, device, dtype):
        # 准备测试数据
        x = torch.tensor([1, 2, 3], dtype=dtype)
        
        if device == "cpu":
            dev_obj = get_device_object("cpu")
            with allure.step(f"执行Repeat - 设备: cpu, 数据类型: {dtype}"):
                x_dev = x.to(device=dev_obj)
                # 在第一个维度上重复2次
                output = x_dev.repeat(2)
                
            with allure.step("验证输出"):
                # 验证形状
                assert output.shape == (6,), f"输出形状不符合预期: 期望 (6,), 实际 {output.shape}"
                assert output.dtype == dtype, f"输出数据类型不符合预期: 期望 {dtype}, 实际 {output.dtype}"
                
                # 验证重复结果
                expected = torch.tensor([1, 2, 3, 1, 2, 3], dtype=dtype)
                assert torch.all(output == expected), "重复结果不正确"
        
        elif device == "cuda":
            # 在CPU上运行
            cpu_output = x.repeat(2)
            
            # 在CUDA上运行
            with allure.step(f"执行Repeat - 设备: cuda, 数据类型: {dtype}"):
                cuda_x = x.cuda()
                cuda_output = cuda_x.repeat(2)
                
            with allure.step("验证输出"):
                assert cuda_output.shape == cpu_output.shape, "CUDA和CPU的输出形状不一致"
                
            with allure.step("比较CPU和CUDA结果"):
                cuda_output_cpu = cuda_output.cpu()
                assert torch.all(cpu_output == cuda_output_cpu), "CPU和CUDA结果不一致"

    @allure.story("边界条件测试")
    @allure.title("测试特殊情况的Repeat")
    @allure.description("""
    验证特殊情况下的重复操作，测试要点：
    1. 重复次数为1（相当于复制）
    2. 重复次数为0（空张量）
    3. 单元素张量的重复
    4. 验证CPU和CUDA结果的一致性
    """)
    def test_repeat_edge_cases(self, device):
        # 准备测试数据
        dtype = torch.float32
        
        # 测试1：重复次数为1（相当于复制）
        x1 = torch.tensor([1, 2, 3], dtype=dtype)
        
        # 测试2：重复次数为0（空张量）
        x2 = torch.tensor([1, 2], dtype=dtype)
        
        # 测试3：单元素张量重复
        x3 = torch.tensor([1], dtype=dtype)
        
        if device == "cpu":
            dev_obj = get_device_object("cpu")
            with allure.step("执行Repeat边界值测试 - 设备: cpu"):
                # 测试1：重复次数为1
                x1_dev = x1.to(device=dev_obj)
                output1 = x1_dev.repeat(1)
                assert torch.all(output1 == x1_dev), "重复次数为1时应该返回原始张量"
                
                # 测试2：重复次数为0
                x2_dev = x2.to(device=dev_obj)
                output2 = x2_dev.repeat(0)
                assert output2.shape == (0,), "重复次数为0时应该返回空张量"
                assert output2.dtype == dtype, "空张量的数据类型应该保持不变"
                
                # 测试3：单元素张量重复
                x3_dev = x3.to(device=dev_obj)
                output3 = x3_dev.repeat(3)
                expected3 = torch.tensor([1, 1, 1], dtype=dtype)
                assert torch.all(output3 == expected3), "单元素重复结果不正确"
        
        elif device == "cuda":
            # 测试1：重复次数为1
            cpu_output1 = x1.repeat(1)
            cuda_x1 = x1.cuda()
            cuda_output1 = cuda_x1.repeat(1)
            assert torch.all(cpu_output1 == cuda_output1.cpu()), "重复次数为1时CUDA和CPU结果不一致"
            
            # 测试2：重复次数为0
            cpu_output2 = x2.repeat(0)
            cuda_x2 = x2.cuda()
            cuda_output2 = cuda_x2.repeat(0)
            assert cpu_output2.shape == cuda_output2.shape, "重复次数为0时CUDA和CPU形状不一致"
            
            # 测试3：单元素张量重复
            cpu_output3 = x3.repeat(3)
            cuda_x3 = x3.cuda()
            cuda_output3 = cuda_x3.repeat(3)
            assert torch.all(cpu_output3 == cuda_output3.cpu()), "单元素重复CUDA和CPU结果不一致"

    @allure.story("性能测试")
    @allure.title("测试大规模Repeat")
    @allure.description("""
    验证大规模张量的重复操作，测试要点：
    1. 处理大规模张量（100x128）
    2. 多维度重复（2次和3次）
    3. 验证重复结果的准确性
    4. 比较CPU和CUDA的计算结果
    """)
    def test_repeat_performance(self, device):
        # 准备大规模测试数据
        dtype = torch.float32
        x = torch.randn(100, 128, dtype=dtype)  # 100x128的矩阵
        
        if device == "cpu":
            dev_obj = get_device_object("cpu")
            with allure.step("执行大规模Repeat - 设备: cpu"):
                x_dev = x.to(device=dev_obj)
                # 在第一维重复2次，第二维重复3次
                output = x_dev.repeat(2, 3)
                
            with allure.step("验证输出"):
                assert output.shape == (200, 384), \
                    f"输出形状不符合预期: 期望 (200, 384), 实际 {output.shape}"
                
                # 验证重复的正确性（只检查部分数据）
                for i in range(min(5, x.shape[0])):
                    for j in range(min(5, x.shape[1])):
                        # 验证第一个重复块
                        assert x[i, j] == output[i, j]
                        # 验证水平方向的重复
                        assert x[i, j] == output[i, j + x.shape[1]]
                        # 验证垂直方向的重复
                        assert x[i, j] == output[i + x.shape[0], j]
        
        elif device == "cuda":
            # 在CPU上运行
            cpu_output = x.repeat(2, 3)
            
            # 在CUDA上运行
            with allure.step("执行大规模Repeat - 设备: cuda"):
                cuda_x = x.cuda()
                cuda_output = cuda_x.repeat(2, 3)
                
            with allure.step("验证输出"):
                assert cuda_output.shape == cpu_output.shape, "CUDA和CPU的输出形状不一致"
                
            with allure.step("比较CPU和CUDA结果"):
                # 只比较部分结果以节省内存
                sample_indices = torch.randint(0, cuda_output.shape[0], (5,))
                for idx in sample_indices:
                    assert torch.all(cpu_output[idx] == cuda_output[idx].cpu()), \
                        f"第{idx}行的CPU和CUDA结果不一致"

    @allure.story("多维度测试")
    @allure.title("测试不同维度组合的Repeat")
    @allure.description("""
    验证不同维度组合的重复操作，测试要点：
    1. 1D到2D的重复
    2. 2D到3D的重复
    3. 带有1的维度的重复
    4. 验证不同维度下的结果准确性
    """)
    def test_repeat_dimensions(self, device):
        # 准备不同维度的测试数据
        dtype = torch.float32
        test_cases = [
            # (输入形状, 重复次数, 期望输出形状)
            ((2,), (3,), (6,)),  # 1D -> 1D
            ((2, 3), (2, 2), (4, 6)),  # 2D -> 2D
            ((2, 3, 4), (2, 1, 3), (4, 3, 12)),  # 3D -> 3D
            ((2, 1, 3), (1, 4, 1), (2, 4, 3)),  # 3D带1的维度
        ]
        
        for input_shape, repeats, expected_shape in test_cases:
            x = torch.randn(*input_shape, dtype=dtype)
            
            if device == "cpu":
                dev_obj = get_device_object("cpu")
                with allure.step(f"执行{len(input_shape)}D Repeat - 设备: cpu"):
                    x_dev = x.to(device=dev_obj)
                    output = x_dev.repeat(*repeats)
                    
                with allure.step("验证输出"):
                    assert output.shape == expected_shape, \
                        f"输出形状不符合预期: 期望 {expected_shape}, 实际 {output.shape}"
                    
                    # 验证重复后的内容（仅检查原始数据在重复后的正确性）
                    original_slice = tuple(slice(0, dim) for dim in input_shape)
                    assert torch.all(output[original_slice] == x), "重复后原始数据部分不正确"
            
            elif device == "cuda":
                # 在CPU上运行
                cpu_output = x.repeat(*repeats)
                
                # 在CUDA上运行
                with allure.step(f"执行{len(input_shape)}D Repeat - 设备: cuda"):
                    cuda_x = x.cuda()
                    cuda_output = cuda_x.repeat(*repeats)
                    
                with allure.step("验证输出"):
                    assert cuda_output.shape == cpu_output.shape, "CUDA和CPU的输出形状不一致"
                    
                with allure.step("比较CPU和CUDA结果"):
                    cuda_output_cpu = cuda_output.cpu()
                    assert torch.all(cpu_output == cuda_output_cpu), "CPU和CUDA结果不一致"
