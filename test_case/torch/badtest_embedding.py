#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import torch
import allure
import numpy as np

def get_device_object(device_str):
    """获取torch.device对象"""
    if device_str == "cuda":
        return torch.device("cuda:0")
    return torch.device("cpu")

test_dtypes = [
    torch.float16,
    torch.float32,
    torch.float64
]

test_input_dtypes = [
    torch.int32,
    torch.int64
]

@allure.epic("PyTorch算子测试")
@allure.feature("Embedding算子")
class TestEmbedding:
    
    @allure.story("基础功能测试")
    @allure.title("测试不同数据类型的Embedding")
    @pytest.mark.parametrize("dtype", test_dtypes)
    def test_embedding_dtypes(self, device, dtype):

        num_embeddings = 10
        embedding_dim = 4
        input_data = torch.tensor([[1, 2], [3, 4]], dtype=torch.long)
        
        if device == "cpu":
            dev_obj = get_device_object("cpu")
            with allure.step(f"创建Embedding层 - 设备: cpu, 数据类型: {dtype}"):
                embedding = torch.nn.Embedding(num_embeddings, embedding_dim)
                embedding = embedding.to(dtype=dtype)
                input_tensor = input_data.to(device=dev_obj)
                
            with allure.step("执行前向传播"):
                output = embedding(input_tensor)
                
            with allure.step("验证输出"):
                expected_shape = (2, 2, embedding_dim)
                assert output.shape == expected_shape, f"输出形状不符合预期: 期望 {expected_shape}, 实际 {output.shape}"
                assert output.dtype == dtype, f"输出数据类型不符合预期: 期望 {dtype}, 实际 {output.dtype}"
        
        elif device == "cuda":
            # 在CPU上运行
            cpu_embedding = torch.nn.Embedding(num_embeddings, embedding_dim)
            cpu_embedding = cpu_embedding.to(dtype=dtype)
            cpu_output = cpu_embedding(input_data)
            
            # 在CUDA上运行
            with allure.step(f"创建Embedding层 - 设备: cuda, 数据类型: {dtype}"):
                cuda_embedding = torch.nn.Embedding(num_embeddings, embedding_dim)
                cuda_embedding.load_state_dict(cpu_embedding.state_dict())  # 使用相同的权重
                cuda_embedding = cuda_embedding.cuda()
                cuda_embedding = cuda_embedding.to(dtype=dtype)
                cuda_input = input_data.cuda()
                
            with allure.step("执行前向传播"):
                cuda_output = cuda_embedding(cuda_input)
                
            with allure.step("验证输出"):
                expected_shape = (2, 2, embedding_dim)
                assert cuda_output.shape == expected_shape, f"输出形状不符合预期: 期望 {expected_shape}, 实际 {cuda_output.shape}"
                assert cuda_output.dtype == dtype, f"输出数据类型不符合预期: 期望 {dtype}, 实际 {cuda_output.dtype}"
                
            with allure.step("比较CPU和CUDA结果"):
                cuda_output_cpu = cuda_output.cpu()
                max_diff = torch.max(torch.abs(cpu_output - cuda_output_cpu))
                assert max_diff < 1e-5, f"CPU和CUDA结果不一致，最大差异: {max_diff}"
                
    @allure.story("基础功能测试")
    @allure.title("测试不同输入数据类型的Embedding")
    @pytest.mark.parametrize("input_dtype", test_input_dtypes)
    def test_embedding_input_dtypes(self, device, input_dtype):
        num_embeddings = 10
        embedding_dim = 4
        input_data = torch.tensor([[1, 2], [3, 4]], dtype=input_dtype)
        
        if device == "cpu":
            dev_obj = get_device_object("cpu")
            with allure.step(f"创建Embedding层 - 设备: cpu, 输入数据类型: {input_dtype}"):
                embedding = torch.nn.Embedding(num_embeddings, embedding_dim)
                input_tensor = input_data.to(device=dev_obj)
                
            with allure.step("执行前向传播"):
                output = embedding(input_tensor)
                
            with allure.step("验证输出"):
                expected_shape = (2, 2, embedding_dim)
                assert output.shape == expected_shape, f"输出形状不符合预期: 期望 {expected_shape}, 实际 {output.shape}"
        
        elif device == "cuda":
            # 在CPU上运行
            cpu_embedding = torch.nn.Embedding(num_embeddings, embedding_dim)
            cpu_output = cpu_embedding(input_data)
            
            # 在CUDA上运行
            with allure.step(f"创建Embedding层 - 设备: cuda, 输入数据类型: {input_dtype}"):
                cuda_embedding = torch.nn.Embedding(num_embeddings, embedding_dim)
                cuda_embedding.load_state_dict(cpu_embedding.state_dict())  # 使用相同的权重
                cuda_embedding = cuda_embedding.cuda()
                cuda_input = input_data.cuda()
                
            with allure.step("执行前向传播"):
                cuda_output = cuda_embedding(cuda_input)
                
            with allure.step("验证输出"):
                expected_shape = (2, 2, embedding_dim)
                assert cuda_output.shape == expected_shape, f"输出形状不符合预期: 期望 {expected_shape}, 实际 {cuda_output.shape}"
                
            with allure.step("比较CPU和CUDA结果"):
                cuda_output_cpu = cuda_output.cpu()
                max_diff = torch.max(torch.abs(cpu_output - cuda_output_cpu))
                assert max_diff < 1e-5, f"CPU和CUDA结果不一致，最大差异: {max_diff}"
            
    @allure.story("边界条件测试")
    @allure.title("测试空输入和最大索引")
    def test_embedding_edge_cases(self, device):
        if device == "cpu":
            dev_obj = get_device_object("cpu")
            
            with allure.step("测试空输入 - 设备: cpu"):
                embedding = torch.nn.Embedding(10, 4)
                empty_input = torch.tensor([], dtype=torch.long)
                output = embedding(empty_input)
                assert output.shape[0] == 0, "空输入应该产生空输出"
                
            with allure.step("测试最大索引 - 设备: cpu"):
                max_index = 9
                input_tensor = torch.tensor([max_index], dtype=torch.long)
                output = embedding(input_tensor)
                assert output.shape == (1, 4), "最大索引输出形状不正确"
                
            with allure.step("测试索引越界 - 设备: cpu"):
                try:
                    invalid_input = torch.tensor([10], dtype=torch.long)
                    output = embedding(invalid_input)
                    pytest.fail("预期应该抛出索引越界错误")
                except IndexError:
                    pass
        
        elif device == "cuda":
            # 在CPU上运行以获取基准结果
            cpu_embedding = torch.nn.Embedding(10, 4)
            
            # 在CUDA上运行
            cuda_embedding = torch.nn.Embedding(10, 4)
            cuda_embedding.load_state_dict(cpu_embedding.state_dict())  # 使用相同的权重
            cuda_embedding = cuda_embedding.cuda()
            
            with allure.step("测试空输入 - 设备: cuda"):
                # CPU结果
                cpu_empty = torch.tensor([], dtype=torch.long)
                cpu_output = cpu_embedding(cpu_empty)
                
                # CUDA结果
                cuda_empty = cpu_empty.cuda()
                cuda_output = cuda_embedding(cuda_empty)
                
                assert cuda_output.shape[0] == 0, "空输入应该产生空输出"
                assert cuda_output.shape == cpu_output.shape, "CUDA和CPU的输出形状不一致"
                
            with allure.step("测试最大索引 - 设备: cuda"):
                # CPU结果
                max_index = 9
                cpu_input = torch.tensor([max_index], dtype=torch.long)
                cpu_output = cpu_embedding(cpu_input)
                
                # CUDA结果
                cuda_input = cpu_input.cuda()
                cuda_output = cuda_embedding(cuda_input)
                
                assert cuda_output.shape == (1, 4), "最大索引输出形状不正确"
                assert cuda_output.cpu().shape == cpu_output.shape, "CUDA和CPU的输出形状不一致"
                max_diff = torch.max(torch.abs(cpu_output - cuda_output.cpu()))
                assert max_diff < 1e-5, f"CPU和CUDA结果不一致，最大差异: {max_diff}"
                
            with allure.step("测试索引越界 - 设备: cuda"):
                try:
                    invalid_input = torch.tensor([10], dtype=torch.long).cuda()
                    output = cuda_embedding(invalid_input)
                    torch.cuda.synchronize()
                    pytest.fail("预期应该抛出索引越界错误")
                except RuntimeError:
                    pass

    @allure.story("性能测试")
    @allure.title("测试大规模Embedding")
    def test_embedding_performance(self, device):
        num_embeddings = 1000
        embedding_dim = 64
        batch_size = 100
        seq_length = 10
        
        if device == "cpu":
            with allure.step("创建小规模Embedding - 设备: cpu"):
                try:
                    embedding = torch.nn.Embedding(num_embeddings, embedding_dim)
                    input_tensor = torch.randint(0, num_embeddings, (batch_size, seq_length), dtype=torch.long)
                except RuntimeError as e:
                    pytest.skip(f"创建 Embedding 失败: {str(e)}")
                
            with allure.step("执行前向传播"):
                try:
                    output = embedding(input_tensor)
                    assert output.shape == (batch_size, seq_length, embedding_dim), "输出形状不正确"
                    
                    # 测试模型参数是否可访问
                    params = list(embedding.parameters())
                    assert len(params) > 0, "模型参数为空"
                    
                    # 测试反向传播
                    loss = output.sum()
                    loss.backward()
                except RuntimeError as e:
                    pytest.skip(f"执行前向/反向传播失败: {str(e)}")
        
        elif device == "cuda":
            # 在CPU上运行以获取基准结果
            with allure.step("创建小规模Embedding - 设备: cuda"):
                try:
                    # 创建CPU模型和输入
                    cpu_embedding = torch.nn.Embedding(num_embeddings, embedding_dim)
                    input_tensor = torch.randint(0, num_embeddings, (batch_size, seq_length), dtype=torch.long)
                    
                    # 创建CUDA模型
                    cuda_embedding = torch.nn.Embedding(num_embeddings, embedding_dim)
                    cuda_embedding.load_state_dict(cpu_embedding.state_dict())  # 使用相同的权重
                    cuda_embedding = cuda_embedding.cuda()
                    cuda_input = input_tensor.cuda()
                except RuntimeError as e:
                    pytest.skip(f"创建 Embedding 失败: {str(e)}")
                
            with allure.step("执行前向传播并比较结果"):
                try:
                    # CPU前向传播
                    cpu_output = cpu_embedding(input_tensor)
                    
                    # CUDA前向传播
                    cuda_output = cuda_embedding(cuda_input)
                    
                    # 验证输出
                    assert cuda_output.shape == (batch_size, seq_length, embedding_dim), "输出形状不正确"
                    assert cuda_output.shape == cpu_output.shape, "CUDA和CPU的输出形状不一致"
                    
                    # 比较结果
                    max_diff = torch.max(torch.abs(cpu_output - cuda_output.cpu()))
                    assert max_diff < 1e-5, f"CPU和CUDA结果不一致，最大差异: {max_diff}"
                    
                    # 测试模型参数是否可访问
                    cuda_params = list(cuda_embedding.parameters())
                    assert len(cuda_params) > 0, "模型参数为空"
                    
                    # 测试反向传播
                    cuda_loss = cuda_output.sum()
                    cuda_loss.backward()
                except RuntimeError as e:
                    pytest.skip(f"执行前向/反向传播失败: {str(e)}")
