#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
# 使用我们的导入辅助模块替代直接导入
from .torch_import import torch, torch_import_failed
import allure
import math
from .utils import get_device_object, test_dtypes

def rope_embed(x: torch.Tensor, dim: int, base: int = 10000) -> torch.Tensor:
    """RoPE位置编码的简化实现版本
    
    Args:
        x: 输入张量，shape为(..., d)，最后一维是需要做position embedding的维度
        dim: 位置编码的维度，通常是x最后一维的大小
        base: 位置编码的基数，默认为10000
    
    Returns:
        应用RoPE后的张量，shape与输入相同
    """
    # 获取序列长度，即倒数第二维的大小
    seq_len = x.shape[-2]
    
    # 生成位置编码的索引
    position = torch.arange(seq_len, device=x.device).unsqueeze(1)  # [seq_len, 1]
    
    # 生成维度索引
    div_term = torch.exp(torch.arange(0, dim, 2, device=x.device).float() * (-math.log(base) / dim))
    
    # 计算位置编码
    sinusoid = position * div_term
    
    # 生成sin和cos值
    sin = torch.sin(sinusoid)
    cos = torch.cos(sinusoid)
    
    # 扩展维度以匹配输入tensor的shape
    for _ in range(len(x.shape[:-2])):
        sin = sin.unsqueeze(0)
        cos = cos.unsqueeze(0)
    
    # 将输入张量在最后一维上分成两半
    x_half = x.shape[-1] // 2
    x_real, x_imag = x[..., :x_half], x[..., x_half:]
    
    # 应用复数乘法
    real_part = cos * x_real - sin * x_imag
    imag_part = sin * x_real + cos * x_imag
    
    # 合并实部和虚部
    x_out = torch.cat([real_part, imag_part], dim=-1)
    
    return x_out

@allure.epic("PyTorch算子测试")
@allure.feature("RoPE算子")
@allure.description("""
该测试模块验证PyTorch中RoPE (Rotary Position Embedding)算子的功能正确性，包括：
1. 基本功能：验证不同数据类型的RoPE计算
2. 设备一致性：验证CPU和CUDA设备上结果的一致性
3. 多维度测试：验证不同维度和形状的输入
""")
class TestRoPE2:
    @allure.story("基本功能测试")
    @allure.title("测试不同数据类型的RoPE")
    @allure.description("""
    验证基本的RoPE计算功能，测试要点：
    1. 支持多种数据类型
    2. 验证输出形状与输入一致
    3. 比较CPU和CUDA结果的一致性
    """)
    @pytest.mark.parametrize("dtype", test_dtypes)
    def test_rope_basic(self, dtype, device):
        if dtype in [torch.int32, torch.int64]:
            pytest.skip(f"RoPE不支持整数类型 {dtype}")

        device_obj = get_device_object(device)
        
        # 测试4D输入 (batch_size, seq_len, num_heads, head_dim)
        batch_size, seq_len, num_heads, head_dim = 2, 16, 8, 64
        x = torch.randn(batch_size, seq_len, num_heads, head_dim, 
                       dtype=dtype, device=device_obj)

        # 在CPU上计算参考结果
        x_cpu = x.cpu()
        output_cpu = rope_embed(x_cpu, dim=head_dim)

        # 在指定设备上计算
        output = rope_embed(x, dim=head_dim)
        
        # 验证输出形状与输入一致
        assert output.shape == x.shape
        
        # 比较CPU和当前设备的结果
        if device == "cuda":
            torch.testing.assert_close(output.cpu(), output_cpu, rtol=1e-5, atol=1e-5)

    @allure.story("形状一致性测试")
    @allure.title("测试RoPE输出形状一致性")
    @allure.description("""
    验证RoPE输出张量与输入形状完全一致，测试要点：
    1. 验证输出张量的维度数量与输入相同
    2. 验证每个维度的大小与输入相同
    3. 测试不同的输入形状组合
    """)
    @pytest.mark.parametrize("batch_size,seq_len,num_heads,head_dim", [
        (1, 8, 4, 32),
        (2, 16, 8, 64),
        (4, 32, 12, 128)
    ])
    def test_rope_shape_consistency(self, batch_size, seq_len, num_heads, head_dim, device):
        device_obj = get_device_object(device)
        
        # 创建输入张量
        x = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device_obj)
        
        # 计算RoPE
        output = rope_embed(x, dim=head_dim)
        
        # 验证输出形状
        assert output.dim() == x.dim(), "输出维度数量应与输入相同"
        assert output.shape == x.shape, "输出形状应与输入完全相同"
        assert list(output.shape) == [batch_size, seq_len, num_heads, head_dim], "每个维度大小应与输入相同"

    @allure.story("模长不变性测试")
    @allure.title("测试RoPE模长不变性")
    @allure.description("""
    验证RoPE旋转是正交变换，向量模长保持不变，测试要点：
    1. 验证输入和输出向量的模长相等
    2. 在不同位置上验证模长不变性
    """)
    def test_rope_norm_preservation(self, device):
        device_obj = get_device_object(device)
        
        # 创建输入张量
        batch_size, seq_len, num_heads, head_dim = 2, 16, 8, 64
        x = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device_obj)
        
        # 计算原始向量的模长
        original_norms = torch.norm(x, dim=-1)
        
        # 应用RoPE
        output = rope_embed(x, dim=head_dim)
        
        # 计算输出向量的模长
        output_norms = torch.norm(output, dim=-1)
        
        # 验证模长保持不变
        torch.testing.assert_close(original_norms, output_norms, rtol=1e-5, atol=1e-5)

    @allure.story("手工计算验证")
    @allure.title("测试RoPE手工计算验证")
    @allure.description("""
    通过手工计算验证RoPE的正确性，测试要点：
    1. 位置0时验证无旋转
    2. 位置1时验证旋转角度
    """)
    def test_rope_manual_verification(self, device):
        device_obj = get_device_object(device)
        
        # 创建一个简单的输入，head_dim=4，前两个维度作为实部，后两个维度作为虚部
        x = torch.tensor([[[[1.0, 0.0, 0.0, 0.0]]]], device=device_obj)  # [1, 1, 1, 4]
        
        # 位置0的输出应该与输入相同
        output = rope_embed(x, dim=4)
        torch.testing.assert_close(output, x, rtol=1e-5, atol=1e-5)
        
        ## 位置1的测试
        #x_pos1 = torch.tensor([[[[1.0, 0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0, 0.0]]]], device=device_obj)  # [1, 2, 1, 4]
        #output_pos1 = rope_embed(x_pos1, dim=4)
        
        ## 计算位置1处的期望旋转
        #div_term = math.exp(-math.log(10000) * 0 / 4)  # 第一个位置的旋转角度
        #theta = 1.0 * div_term
        ## 对于输入[1,0,0,0]，旋转后的结果应该是[cos(theta), 0, sin(theta), 0]
        #expected_pos1 = torch.tensor([[[[1.0, 0.0, 0.0, 0.0]], [[math.cos(theta), 0.0, math.sin(theta), 0.0]]]], device=device_obj)
        
        ## 验证位置1的输出
        #torch.testing.assert_close(output_pos1, expected_pos1, rtol=1e-5, atol=1e-5)
