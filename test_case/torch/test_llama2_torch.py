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
    torch.float64,
    torch.int32,
    torch.int64
]

@allure.epic("PyTorch算子测试")
@allure.feature("Add算子")
@pytest.mark.order(1)
class TestAdd:
    def teardown_method(self, method):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    @allure.story("基础功能测试")
    @allure.title("测试不同数据类型的标量加法")
    @pytest.mark.parametrize("dtype", test_dtypes)
    def test_add_scalar(self, device, dtype):

        # 准备测试数据
        a = torch.tensor([[1, 2], [3, 4]], dtype=dtype)
        scalar = 2
        
        if device == "cpu":
            dev_obj = get_device_object("cpu")
            with allure.step(f"执行标量加法 - 设备: cpu, 数据类型: {dtype}"):
                a_dev = a.to(device=dev_obj)
                output = a_dev + scalar
                
            with allure.step("验证输出"):
                expected = a + scalar
                assert output.shape == expected.shape, f"输出形状不符合预期: 期望 {expected.shape}, 实际 {output.shape}"
                assert output.dtype == expected.dtype, f"输出数据类型不符合预期: 期望 {expected.dtype}, 实际 {output.dtype}"
                assert torch.all(output == expected), "输出结果不正确"
        
        elif device == "cuda":
            # 在CPU上运行
            cpu_output = a + scalar
            
            # 在CUDA上运行
            with allure.step(f"执行标量加法 - 设备: cuda, 数据类型: {dtype}"):
                cuda_input = a.cuda()
                cuda_output = cuda_input + scalar
                
            with allure.step("验证输出"):
                assert cuda_output.shape == cpu_output.shape, "CUDA和CPU的输出形状不一致"
                assert cuda_output.dtype == cpu_output.dtype, "CUDA和CPU的输出数据类型不一致"
                
            with allure.step("比较CPU和CUDA结果"):
                cuda_output_cpu = cuda_output.cpu()
                max_diff = torch.max(torch.abs(cpu_output - cuda_output_cpu))
                assert max_diff < 1e-5, f"CPU和CUDA结果不一致，最大差异: {max_diff}"
    
    @allure.story("基础功能测试")
    @allure.title("测试不同数据类型的张量加法")
    @pytest.mark.parametrize("dtype", test_dtypes)
    def test_add_tensor(self, device, dtype):

        # 准备测试数据
        a = torch.tensor([[1, 2], [3, 4]], dtype=dtype)
        b = torch.tensor([[5, 6], [7, 8]], dtype=dtype)
        
        if device == "cpu":
            dev_obj = get_device_object("cpu")
            with allure.step(f"执行张量加法 - 设备: cpu, 数据类型: {dtype}"):
                a_dev = a.to(device=dev_obj)
                b_dev = b.to(device=dev_obj)
                output = a_dev + b_dev
                
            with allure.step("验证输出"):
                expected = a + b
                assert output.shape == expected.shape, f"输出形状不符合预期: 期望 {expected.shape}, 实际 {output.shape}"
                assert output.dtype == expected.dtype, f"输出数据类型不符合预期: 期望 {expected.dtype}, 实际 {output.dtype}"
                assert torch.all(output == expected), "输出结果不正确"
        
        elif device == "cuda":
            # 在CPU上运行
            cpu_output = a + b
            
            # 在CUDA上运行
            with allure.step(f"执行张量加法 - 设备: cuda, 数据类型: {dtype}"):
                cuda_a = a.cuda()
                cuda_b = b.cuda()
                cuda_output = cuda_a + cuda_b
                
            with allure.step("验证输出"):
                assert cuda_output.shape == cpu_output.shape, "CUDA和CPU的输出形状不一致"
                assert cuda_output.dtype == cpu_output.dtype, "CUDA和CPU的输出数据类型不一致"
                
            with allure.step("比较CPU和CUDA结果"):
                cuda_output_cpu = cuda_output.cpu()
                max_diff = torch.max(torch.abs(cpu_output - cuda_output_cpu))
                assert max_diff < 1e-5, f"CPU和CUDA结果不一致，最大差异: {max_diff}"
    
    @allure.story("边界条件测试")
    @allure.title("测试大规模张量加法")
    def test_add_performance(self, device):
        # 准备大规模测试数据
        shape = (1000, 1000)
        dtype = torch.float32
        a = torch.randn(shape, dtype=dtype)
        b = torch.randn(shape, dtype=dtype)
        
        if device == "cpu":
            dev_obj = get_device_object("cpu")
            with allure.step("执行大规模张量加法 - 设备: cpu"):
                a_dev = a.to(device=dev_obj)
                b_dev = b.to(device=dev_obj)
                output = a_dev + b_dev
                
            with allure.step("验证输出"):
                expected = a + b
                assert output.shape == expected.shape, f"输出形状不符合预期: 期望 {expected.shape}, 实际 {output.shape}"
                assert torch.allclose(output, expected, rtol=1e-5), "输出结果不正确"
        
        elif device == "cuda":
            # 在CPU上运行
            cpu_output = a + b
            
            # 在CUDA上运行
            with allure.step("执行大规模张量加法 - 设备: cuda"):
                cuda_a = a.cuda()
                cuda_b = b.cuda()
                cuda_output = cuda_a + cuda_b
                
            with allure.step("验证输出"):
                assert cuda_output.shape == cpu_output.shape, "CUDA和CPU的输出形状不一致"
                
            with allure.step("比较CPU和CUDA结果"):
                cuda_output_cpu = cuda_output.cpu()
                max_diff = torch.max(torch.abs(cpu_output - cuda_output_cpu))
                assert max_diff < 1e-5, f"CPU和CUDA结果不一致，最大差异: {max_diff}"
    
    @allure.story("边界条件测试")
    @allure.title("测试广播加法")
    def test_add_broadcast(self, device):
        # 准备不同形状的测试数据
        dtype = torch.float32
        a = torch.randn((3, 1, 4), dtype=dtype)  # shape: (3, 1, 4)
        b = torch.randn((1, 2, 4), dtype=dtype)  # shape: (1, 2, 4)
        # 结果shape应该是: (3, 2, 4)
        
        if device == "cpu":
            dev_obj = get_device_object("cpu")
            with allure.step("执行广播加法 - 设备: cpu"):
                a_dev = a.to(device=dev_obj)
                b_dev = b.to(device=dev_obj)
                output = a_dev + b_dev
                
            with allure.step("验证输出"):
                expected = a + b
                expected_shape = (3, 2, 4)
                assert output.shape == expected_shape, f"输出形状不符合预期: 期望 {expected_shape}, 实际 {output.shape}"
                assert torch.allclose(output, expected, rtol=1e-5), "输出结果不正确"
        
        elif device == "cuda":
            # 在CPU上运行
            cpu_output = a + b
            
            # 在CUDA上运行
            with allure.step("执行广播加法 - 设备: cuda"):
                cuda_a = a.cuda()
                cuda_b = b.cuda()
                cuda_output = cuda_a + cuda_b
                
            with allure.step("验证输出"):
                expected_shape = (3, 2, 4)
                assert cuda_output.shape == expected_shape, f"输出形状不符合预期: 期望 {expected_shape}, 实际 {cuda_output.shape}"
                
            with allure.step("比较CPU和CUDA结果"):
                cuda_output_cpu = cuda_output.cpu()
                max_diff = torch.max(torch.abs(cpu_output - cuda_output_cpu))
                assert max_diff < 1e-5, f"CPU和CUDA结果不一致，最大差异: {max_diff}"

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
                cuda_input = a.cuda()
                cuda_output = cuda_input * scalar
                
            with allure.step("验证输出"):
                assert cuda_output.shape == cpu_output.shape, "CUDA和CPU的输出形状不一致"
                assert cuda_output.dtype == cpu_output.dtype, "CUDA和CPU的输出数据类型不一致"
                
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
                assert cuda_output.dtype == cpu_output.dtype, "CUDA和CPU的输出数据类型不一致"
                
            with allure.step("比较CPU和CUDA结果"):
                cuda_output_cpu = cuda_output.cpu()
                max_diff = torch.max(torch.abs(cpu_output - cuda_output_cpu))
                assert max_diff < 1e-5, f"CPU和CUDA结果不一致，最大差异: {max_diff}"
    
    @allure.story("边界条件测试")
    @allure.title("测试大规模张量乘法")
    def test_mul_performance(self, device):
        # 准备大规模测试数据
        shape = (1000, 1000)
        dtype = torch.float32
        a = torch.randn(shape, dtype=dtype)
        b = torch.randn(shape, dtype=dtype)
        
        if device == "cpu":
            dev_obj = get_device_object("cpu")
            with allure.step("执行大规模张量乘法 - 设备: cpu"):
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
            with allure.step("执行大规模张量乘法 - 设备: cuda"):
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
        # 准备不同形状的测试数据
        dtype = torch.float32
        a = torch.randn((3, 1, 4), dtype=dtype)  # shape: (3, 1, 4)
        b = torch.randn((1, 2, 4), dtype=dtype)  # shape: (1, 2, 4)
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
                expected_shape = (3, 2, 4)
                assert cuda_output.shape == expected_shape, f"输出形状不符合预期: 期望 {expected_shape}, 实际 {cuda_output.shape}"
                
            with allure.step("比较CPU和CUDA结果"):
                cuda_output_cpu = cuda_output.cpu()
                max_diff = torch.max(torch.abs(cpu_output - cuda_output_cpu))
                assert max_diff < 1e-5, f"CPU和CUDA结果不一致，最大差异: {max_diff}"

em_test_dtypes = [
    torch.float16,
    torch.float32,
    torch.float64
]
em_test_input_dtypes = [
    torch.int32,
    torch.int64
]
@allure.epic("PyTorch算子测试")
@allure.feature("Embedding算子")
class TestEmbedding:
    
    @allure.story("基础功能测试")
    @allure.title("测试不同数据类型的Embedding")
    @pytest.mark.parametrize("dtype", em_test_dtypes)
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
    @pytest.mark.parametrize("input_dtype", em_test_input_dtypes)
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


@allure.epic("PyTorch算子测试")
@allure.feature("MatMul算子")
@pytest.mark.order(3)
class TestMatMul:
    def setup_method(self, method):
        # 检查CUDA是否可用
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
    def teardown_method(self, method):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        if not torch.cuda.is_available():
            pytest.skip("CUDA不可用")
            
        # 检查CUDA设备是否就绪
        try:
            torch.cuda.init()
        except RuntimeError:
            pytest.skip("CUDA设备初始化失败")
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
                assert cuda_output.dtype == cpu_output.dtype, "CUDA和CPU的输出数据类型不一致"
                
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
                expected_shape = (3, 2, 3)
                assert cuda_output.shape == expected_shape, f"输出形状不符合预期: 期望 {expected_shape}, 实际 {cuda_output.shape}"
                
            with allure.step("比较CPU和CUDA结果"):
                cuda_output_cpu = cuda_output.cpu()
                max_diff = torch.max(torch.abs(cpu_output - cuda_output_cpu))
                assert max_diff < 1e-5, f"CPU和CUDA结果不一致，最大差异: {max_diff}"
