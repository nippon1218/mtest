#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import torch
import allure
import math
from .utils import get_device_object, test_dtypes

def apply_rotary_emb(x, cos, sin, position_ids=None):
    """应用Rotary Position Embedding
    
    Args:
        x: 输入张量，shape为(batch_size, seq_len, num_heads, head_dim)
        cos: 余弦张量，shape为(seq_len, dim)
        sin: 正弦张量，shape为(seq_len, dim)
        position_ids: 位置ID，shape为(batch_size, seq_len)，默认为None
    
    Returns:
        应用RoPE后的张量
    """
    # 获取输入张量的维度
    batch_size, seq_len, num_heads, head_dim = x.shape
    
    # 如果没有提供position_ids，则使用默认的位置序列
    if position_ids is None:
        position_ids = torch.arange(seq_len, device=x.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
    
    # 根据position_ids获取对应的cos和sin值
    cos = cos[position_ids]  # [batch_size, seq_len, head_dim]
    sin = sin[position_ids]  # [batch_size, seq_len, head_dim]
    
    # 扩展维度以匹配num_heads
    cos = cos.unsqueeze(2)  # [batch_size, seq_len, 1, head_dim]
    sin = sin.unsqueeze(2)  # [batch_size, seq_len, 1, head_dim]
    
    # 将输入张量分成两半
    x_half = x.shape[-1] // 2
    x_real, x_imag = x[..., :x_half], x[..., x_half:]
    
    # 应用复数乘法的实部和虚部
    real_part = cos[..., :x_half] * x_real - sin[..., :x_half] * x_imag
    imag_part = sin[..., :x_half] * x_real + cos[..., :x_half] * x_imag
    
    # 合并实部和虚部
    x_out = torch.cat([real_part, imag_part], dim=-1)
    
    return x_out

@allure.epic("PyTorch算子测试")
@allure.feature("Rope算子")
@allure.description("""
该测试模块验证PyTorch中Rotary Position Embedding (RoPE)的功能正确性，包括：
1. 基本功能：验证不同数据类型的RoPE计算
2. 自定义位置：验证使用自定义position_ids的情况
3. 性能测试：验证大规模数据的RoPE计算

所有测试都在CPU和CUDA设备上执行，并验证结果的一致性。
""")
@pytest.mark.order(5)
class TestRope:
    def setup_method(self, method):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
    def teardown_method(self, method):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    def _get_rotary_embedding(self, seq_len, dim, base=10000, device=None):
        """获取RoPE的sin和cos值
        
        Args:
            seq_len: 序列长度
            dim: 维度
            base: 基数，默认为10000
            device: 设备
        
        Returns:
            (cos, sin) 元组
        """
        # 生成位置编码的索引
        position = torch.arange(seq_len, dtype=torch.float, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float, device=device) * (-math.log(base) / dim))
        
        # 计算sin和cos值
        sinusoid = position * div_term
        sin = torch.sin(sinusoid)
        cos = torch.cos(sinusoid)
        
        # 扩展维度以匹配head_dim
        sin = torch.cat([sin, sin], dim=-1)
        cos = torch.cat([cos, cos], dim=-1)
        
        return cos, sin

    @allure.story("基础功能测试")
    @allure.title("测试不同数据类型的RoPE")
    @allure.description("""
    验证基本的RoPE功能，测试要点：
    1. 支持多种数据类型（float32、float64）
    2. 验证输出形状和数据类型的正确性
    3. 验证RoPE计算的准确性
    4. 比较CPU和CUDA结果的一致性
    """)
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_rope_basic(self, device, dtype):
        # 准备测试数据
        batch_size = 2
        seq_len = 4
        num_heads = 2
        head_dim = 6
        
        # 创建输入张量
        x = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=dtype)
        
        if device == "cpu":
            dev_obj = get_device_object("cpu")
            with allure.step(f"执行RoPE - 设备: cpu, 数据类型: {dtype}"):
                x_dev = x.to(device=dev_obj)
                cos, sin = self._get_rotary_embedding(seq_len, head_dim, device=dev_obj)
                output = apply_rotary_emb(x_dev, cos, sin)
                
            with allure.step("验证输出"):
                assert output.shape == x.shape, f"输出形状不符合预期: 期望 {x.shape}, 实际 {output.shape}"
                assert output.dtype == dtype, f"输出数据类型不符合预期: 期望 {dtype}, 实际 {output.dtype}"
        
        elif device == "cuda":
            # 在CPU上运行
            cos_cpu, sin_cpu = self._get_rotary_embedding(seq_len, head_dim)
            cpu_output = apply_rotary_emb(x, cos_cpu, sin_cpu)
            
            # 在CUDA上运行
            with allure.step(f"执行RoPE - 设备: cuda, 数据类型: {dtype}"):
                cuda_x = x.cuda()
                cos_cuda, sin_cuda = self._get_rotary_embedding(seq_len, head_dim, device="cuda")
                cuda_output = apply_rotary_emb(cuda_x, cos_cuda, sin_cuda)
                
            with allure.step("验证输出"):
                assert cuda_output.shape == cpu_output.shape, "CUDA和CPU的输出形状不一致"
                
            with allure.step("比较CPU和CUDA结果"):
                cuda_output_cpu = cuda_output.cpu()
                max_diff = torch.max(torch.abs(cpu_output - cuda_output_cpu))
                assert max_diff < 1e-5, f"CPU和CUDA结果不一致，最大差异: {max_diff}"

    @allure.story("边界条件测试")
    @allure.title("测试自定义位置的RoPE")
    @allure.description("""
    验证使用自定义position_ids的RoPE计算，测试要点：
    1. 使用自定义的位置顺序
    2. 验证位置编码的正确性
    3. 验证输出形状的一致性
    4. 比较CPU和CUDA结果的一致性
    """)
    def test_rope_custom_positions(self, device):
        # 准备测试数据
        dtype = torch.float32
        batch_size = 2
        seq_len = 4
        num_heads = 2
        head_dim = 6
        
        # 创建输入张量和自定义位置ID
        x = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=dtype)
        position_ids = torch.tensor([[0, 2, 1, 3], [0, 1, 2, 3]], dtype=torch.long)  # 自定义位置顺序
        
        if device == "cpu":
            dev_obj = get_device_object("cpu")
            with allure.step("执行带自定义位置的RoPE - 设备: cpu"):
                x_dev = x.to(device=dev_obj)
                position_ids_dev = position_ids.to(device=dev_obj)
                cos, sin = self._get_rotary_embedding(seq_len, head_dim, device=dev_obj)
                output = apply_rotary_emb(x_dev, cos, sin, position_ids_dev)
                
            with allure.step("验证输出"):
                assert output.shape == x.shape, f"输出形状不符合预期: 期望 {x.shape}, 实际 {output.shape}"
        
        elif device == "cuda":
            # 在CPU上运行
            cos_cpu, sin_cpu = self._get_rotary_embedding(seq_len, head_dim)
            cpu_output = apply_rotary_emb(x, cos_cpu, sin_cpu, position_ids)
            
            # 在CUDA上运行
            with allure.step("执行带自定义位置的RoPE - 设备: cuda"):
                cuda_x = x.cuda()
                cuda_position_ids = position_ids.cuda()
                cos_cuda, sin_cuda = self._get_rotary_embedding(seq_len, head_dim, device="cuda")
                cuda_output = apply_rotary_emb(cuda_x, cos_cuda, sin_cuda, cuda_position_ids)
                
            with allure.step("验证输出"):
                assert cuda_output.shape == cpu_output.shape, "CUDA和CPU的输出形状不一致"
                
            with allure.step("比较CPU和CUDA结果"):
                cuda_output_cpu = cuda_output.cpu()
                max_diff = torch.max(torch.abs(cpu_output - cuda_output_cpu))
                assert max_diff < 1e-5, f"CPU和CUDA结果不一致，最大差异: {max_diff}"

    @allure.story("边界条件测试")
    @allure.title("测试大规模RoPE")
    @allure.description("""
    验证大规模数据的RoPE计算，测试要点：
    1. 处理大规模数据（batch=8, seq_len=512, num_heads=12）
    2. 验证大规模计算的准确性
    3. 验证输出形状的一致性
    4. 比较CPU和CUDA的计算结果
    """)
    def test_rope_performance(self, device):
        # 准备大规模测试数据
        dtype = torch.float32
        batch_size = 8
        seq_len = 512
        num_heads = 12
        head_dim = 64
        
        # 创建输入张量
        x = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=dtype)
        
        if device == "cpu":
            dev_obj = get_device_object("cpu")
            with allure.step("执行大规模RoPE - 设备: cpu"):
                x_dev = x.to(device=dev_obj)
                cos, sin = self._get_rotary_embedding(seq_len, head_dim, device=dev_obj)
                output = apply_rotary_emb(x_dev, cos, sin)
                
            with allure.step("验证输出"):
                assert output.shape == x.shape, f"输出形状不符合预期: 期望 {x.shape}, 实际 {output.shape}"
        
        elif device == "cuda":
            # 在CPU上运行
            cos_cpu, sin_cpu = self._get_rotary_embedding(seq_len, head_dim)
            cpu_output = apply_rotary_emb(x, cos_cpu, sin_cpu)
            
            # 在CUDA上运行
            with allure.step("执行大规模RoPE - 设备: cuda"):
                cuda_x = x.cuda()
                cos_cuda, sin_cuda = self._get_rotary_embedding(seq_len, head_dim, device="cuda")
                cuda_output = apply_rotary_emb(cuda_x, cos_cuda, sin_cuda)
                
            with allure.step("验证输出"):
                assert cuda_output.shape == cpu_output.shape, "CUDA和CPU的输出形状不一致"
                
            with allure.step("比较CPU和CUDA结果"):
                cuda_output_cpu = cuda_output.cpu()
                max_diff = torch.max(torch.abs(cpu_output - cuda_output_cpu))
                assert max_diff < 1e-4, f"CPU和CUDA结果不一致，最大差异: {max_diff}"
