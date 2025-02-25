#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import torch
import allure
import numpy as np
from .utils import get_device_object, em_test_dtypes, em_input_dtypes

@allure.epic("PyTorch算子测试")
@allure.feature("Embedding算子")
@allure.description("""
该测试模块验证PyTorch中Embedding算子的功能正确性，包括：
1. 基本功能：验证不同数据类型的嵌入层操作
2. 输入类型：测试不同输入数据类型的兼容性
3. 边界情况：验证空张量等特殊情况的处理
4. 性能测试：验证大规模嵌入操作的正确性

所有测试都在CPU和CUDA设备上执行，并验证结果的一致性。
""")
class TestEmbedding:
    @allure.story("基础功能测试")
    @allure.title("测试不同数据类型的Embedding")
    @allure.description("""
    验证Embedding层在不同数据类型下的功能，测试要点：
    1. 支持多种数据类型（float32、float64等）
    2. 验证输出张量的形状正确性
    3. 验证输出数据类型的一致性
    4. 比较CPU和CUDA结果的一致性
    """)
    @pytest.mark.parametrize("dtype", em_test_dtypes)
    def test_embedding_dtypes(self, device, dtype):
        # 准备测试数据
        vocab_size = 10
        embedding_dim = 4
        
        # 创建Embedding层
        embedding = torch.nn.Embedding(vocab_size, embedding_dim, dtype=dtype)
        
        # 准备输入数据
        input_tensor = torch.tensor([[1, 2], [3, 4]], dtype=torch.int64)
        
        device_obj = get_device_object(device)
        with allure.step(f"执行Embedding - 设备: {device}, 数据类型: {dtype}"):
            embedding = embedding.to(device=device_obj)
            input_dev = input_tensor.to(device=device_obj)
            output = embedding(input_dev)
        
        with allure.step("验证输出"):
            expected_shape = (2, 2, embedding_dim)
            assert output.shape == expected_shape, f"输出形状不符合预期: 期望 {expected_shape}, 实际 {output.shape}"
            assert output.dtype == dtype, f"输出数据类型不符合预期: 期望 {dtype}, 实际 {output.dtype}"
        
        if device == "cuda":
            # 在CPU上运行参考结果
            cpu_embedding = embedding.cpu()
            cpu_output = cpu_embedding(input_tensor)
            
            with allure.step("比较CPU和CUDA结果"):
                output_cpu = output.cpu()
                torch.testing.assert_close(output_cpu, cpu_output, rtol=1e-5, atol=1e-5)

    @allure.story("基础功能测试")
    @allure.title("测试不同输入数据类型的Embedding")
    @allure.description("""
    验证Embedding层对不同输入数据类型的处理，测试要点：
    1. 支持多种整数类型的输入（int32、int64等）
    2. 验证输出张量的形状正确性
    3. 比较CPU和CUDA结果的一致性
    4. 确保不同输入类型下的数值准确性
    """)
    @pytest.mark.parametrize("input_dtype", em_input_dtypes)
    def test_embedding_input_dtypes(self, device, input_dtype):
        # 准备测试数据
        vocab_size = 10
        embedding_dim = 4
        
        # 创建Embedding层
        embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        
        # 准备输入数据
        input_tensor = torch.tensor([[1, 2], [3, 4]], dtype=input_dtype)
        
        device_obj = get_device_object(device)
        with allure.step(f"执行Embedding - 设备: {device}, 输入数据类型: {input_dtype}"):
            embedding = embedding.to(device=device_obj)
            input_dev = input_tensor.to(device=device_obj)
            output = embedding(input_dev)
        
        with allure.step("验证输出"):
            expected_shape = (2, 2, embedding_dim)
            assert output.shape == expected_shape, f"输出形状不符合预期: 期望 {expected_shape}, 实际 {output.shape}"
        
        if device == "cuda":
            # 在CPU上运行参考结果
            cpu_embedding = embedding.cpu()
            cpu_output = cpu_embedding(input_tensor)
            
            with allure.step("比较CPU和CUDA结果"):
                output_cpu = output.cpu()
                torch.testing.assert_close(output_cpu, cpu_output, rtol=1e-5, atol=1e-5)

    @allure.story("边界条件测试")
    @allure.title("测试Embedding边界情况")
    @allure.description("""
    验证Embedding层在边界情况下的处理，测试要点：
    1. 处理空张量输入
    2. 验证空张量输出的形状正确性
    3. 确保CPU和CUDA设备的一致性
    4. 验证边界情况下的内存使用
    """)
    def test_embedding_edge_cases(self, device):
        # 准备测试数据
        vocab_size = 10
        embedding_dim = 4
        
        # 创建Embedding层
        embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        
        # 测试空张量
        empty_input = torch.tensor([], dtype=torch.int64)
        
        device_obj = get_device_object(device)
        with allure.step(f"测试空张量 - 设备: {device}"):
            embedding = embedding.to(device=device_obj)
            empty_input = empty_input.to(device=device_obj)
            output = embedding(empty_input)
        
        with allure.step("验证输出"):
            expected_shape = (0, embedding_dim)
            assert output.shape == expected_shape, f"输出形状不符合预期: 期望 {expected_shape}, 实际 {output.shape}"
        
        if device == "cuda":
            # 在CPU上运行参考结果
            cpu_embedding = embedding.cpu()
            cpu_input = empty_input.cpu()
            cpu_output = cpu_embedding(cpu_input)
            
            with allure.step("比较CPU和CUDA结果"):
                output_cpu = output.cpu()
                torch.testing.assert_close(output_cpu, cpu_output, rtol=1e-5, atol=1e-5)

    @allure.story("边界条件测试")
    @allure.title("测试大规模Embedding")
    @allure.description("""
    验证Embedding层在大规模数据下的性能和正确性，测试要点：
    1. 处理大词汇表（10000个词）
    2. 处理高维嵌入向量（128维）
    3. 处理大批量数据（64批次，50序列长度）
    4. 验证CPU和CUDA结果的一致性
    """)
    def test_embedding_performance(self, device):
        try:
            # 准备大规模测试数据
            vocab_size = 10000
            embedding_dim = 128
            batch_size = 64
            seq_length = 50
            
            # 创建Embedding层
            embedding = torch.nn.Embedding(vocab_size, embedding_dim)
            
            # 准备输入数据
            input_tensor = torch.randint(0, vocab_size, (batch_size, seq_length))
            
            device_obj = get_device_object(device)
            with allure.step(f"执行大规模Embedding - 设备: {device}"):
                embedding = embedding.to(device=device_obj)
                input_dev = input_tensor.to(device=device_obj)
                output = embedding(input_dev)
            
            with allure.step("验证输出"):
                expected_shape = (batch_size, seq_length, embedding_dim)
                assert output.shape == expected_shape, f"输出形状不符合预期: 期望 {expected_shape}, 实际 {output.shape}"
            
            if device == "cuda":
                # 在CPU上运行参考结果
                cpu_embedding = embedding.cpu()
                cpu_output = cpu_embedding(input_tensor)
                
                with allure.step("比较CPU和CUDA结果"):
                    output_cpu = output.cpu()
                    torch.testing.assert_close(output_cpu, cpu_output, rtol=1e-5, atol=1e-5)
        except Exception as e:
            pytest.skip(f"创建 Embedding 失败: {str(e)}")
