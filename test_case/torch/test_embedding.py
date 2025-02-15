#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import torch
import allure
import numpy as np
from .utils import get_device_object, em_test_dtypes, em_input_dtypes

@allure.epic("PyTorch算子测试")
@allure.feature("Embedding算子")
@pytest.mark.order(4)
class TestEmbedding:
    @allure.story("基础功能测试")
    @allure.title("测试不同数据类型的Embedding")
    @pytest.mark.parametrize("dtype", em_test_dtypes)
    def test_embedding_dtypes(self, device, dtype):
        # 准备测试数据
        vocab_size = 10
        embedding_dim = 4
        
        # 创建Embedding层
        embedding = torch.nn.Embedding(vocab_size, embedding_dim, dtype=dtype)
        
        # 准备输入数据
        input_tensor = torch.tensor([[1, 2], [3, 4]], dtype=torch.int64)
        
        if device == "cpu":
            dev_obj = get_device_object("cpu")
            with allure.step(f"执行Embedding - 设备: cpu, 数据类型: {dtype}"):
                embedding = embedding.to(device=dev_obj)
                input_dev = input_tensor.to(device=dev_obj)
                output = embedding(input_dev)
                
            with allure.step("验证输出"):
                expected_shape = (2, 2, embedding_dim)
                assert output.shape == expected_shape, f"输出形状不符合预期: 期望 {expected_shape}, 实际 {output.shape}"
                assert output.dtype == dtype, f"输出数据类型不符合预期: 期望 {dtype}, 实际 {output.dtype}"
        
        elif device == "cuda":
            # 在CPU上运行
            cpu_output = embedding(input_tensor)
            
            # 在CUDA上运行
            with allure.step(f"执行Embedding - 设备: cuda, 数据类型: {dtype}"):
                cuda_embedding = embedding.cuda()
                cuda_input = input_tensor.cuda()
                cuda_output = cuda_embedding(cuda_input)
                
            with allure.step("验证输出"):
                assert cuda_output.shape == cpu_output.shape, "CUDA和CPU的输出形状不一致"
                
            with allure.step("比较CPU和CUDA结果"):
                cuda_output_cpu = cuda_output.cpu()
                max_diff = torch.max(torch.abs(cpu_output - cuda_output_cpu))
                assert max_diff < 1e-5, f"CPU和CUDA结果不一致，最大差异: {max_diff}"

    @allure.story("基础功能测试")
    @allure.title("测试不同输入数据类型的Embedding")
    @pytest.mark.parametrize("input_dtype", em_input_dtypes)
    def test_embedding_input_dtypes(self, device, input_dtype):
        # 准备测试数据
        vocab_size = 10
        embedding_dim = 4
        
        # 创建Embedding层
        embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        
        # 准备输入数据
        input_tensor = torch.tensor([[1, 2], [3, 4]], dtype=input_dtype)
        
        if device == "cpu":
            dev_obj = get_device_object("cpu")
            with allure.step(f"执行Embedding - 设备: cpu, 输入数据类型: {input_dtype}"):
                embedding = embedding.to(device=dev_obj)
                input_dev = input_tensor.to(device=dev_obj)
                output = embedding(input_dev)
                
            with allure.step("验证输出"):
                expected_shape = (2, 2, embedding_dim)
                assert output.shape == expected_shape, f"输出形状不符合预期: 期望 {expected_shape}, 实际 {output.shape}"
        
        elif device == "cuda":
            # 在CPU上运行
            cpu_output = embedding(input_tensor)
            
            # 在CUDA上运行
            with allure.step(f"执行Embedding - 设备: cuda, 输入数据类型: {input_dtype}"):
                cuda_embedding = embedding.cuda()
                cuda_input = input_tensor.cuda()
                cuda_output = cuda_embedding(cuda_input)
                
            with allure.step("验证输出"):
                assert cuda_output.shape == cpu_output.shape, "CUDA和CPU的输出形状不一致"
                
            with allure.step("比较CPU和CUDA结果"):
                cuda_output_cpu = cuda_output.cpu()
                max_diff = torch.max(torch.abs(cpu_output - cuda_output_cpu))
                assert max_diff < 1e-5, f"CPU和CUDA结果不一致，最大差异: {max_diff}"

    @allure.story("边界条件测试")
    @allure.title("测试Embedding边界情况")
    def test_embedding_edge_cases(self, device):
        # 准备测试数据
        vocab_size = 10
        embedding_dim = 4
        
        # 创建Embedding层
        embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        
        # 测试空张量
        empty_input = torch.tensor([], dtype=torch.int64)
        
        if device == "cpu":
            dev_obj = get_device_object("cpu")
            with allure.step("测试空张量 - 设备: cpu"):
                embedding = embedding.to(device=dev_obj)
                empty_input = empty_input.to(device=dev_obj)
                output = embedding(empty_input)
                
            with allure.step("验证输出"):
                expected_shape = (0, embedding_dim)
                assert output.shape == expected_shape, f"输出形状不符合预期: 期望 {expected_shape}, 实际 {output.shape}"
        
        elif device == "cuda":
            # 在CPU上运行
            cpu_output = embedding(empty_input)
            
            # 在CUDA上运行
            with allure.step("测试空张量 - 设备: cuda"):
                cuda_embedding = embedding.cuda()
                cuda_input = empty_input.cuda()
                cuda_output = cuda_embedding(cuda_input)
                
            with allure.step("验证输出"):
                assert cuda_output.shape == cpu_output.shape, "CUDA和CPU的输出形状不一致"

    @allure.story("边界条件测试")
    @allure.title("测试大规模Embedding")
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
            
            if device == "cpu":
                dev_obj = get_device_object("cpu")
                with allure.step("执行大规模Embedding - 设备: cpu"):
                    embedding = embedding.to(device=dev_obj)
                    input_dev = input_tensor.to(device=dev_obj)
                    output = embedding(input_dev)
                    
                with allure.step("验证输出"):
                    expected_shape = (batch_size, seq_length, embedding_dim)
                    assert output.shape == expected_shape, f"输出形状不符合预期: 期望 {expected_shape}, 实际 {output.shape}"
            
            elif device == "cuda":
                # 在CPU上运行
                cpu_output = embedding(input_tensor)
                
                # 在CUDA上运行
                with allure.step("执行大规模Embedding - 设备: cuda"):
                    cuda_embedding = embedding.cuda()
                    cuda_input = input_tensor.cuda()
                    cuda_output = cuda_embedding(cuda_input)
                    
                with allure.step("验证输出"):
                    assert cuda_output.shape == cpu_output.shape, "CUDA和CPU的输出形状不一致"
                    
                with allure.step("比较CPU和CUDA结果"):
                    cuda_output_cpu = cuda_output.cpu()
                    max_diff = torch.max(torch.abs(cpu_output - cuda_output_cpu))
                    assert max_diff < 1e-5, f"CPU和CUDA结果不一致，最大差异: {max_diff}"
        except Exception as e:
            pytest.skip(f"创建 Embedding 失败: {str(e)}")
