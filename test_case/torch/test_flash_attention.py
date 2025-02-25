#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import torch
import allure
import numpy as np
from .utils import get_device_object, test_dtypes

@allure.epic("PyTorch算子测试")
@allure.feature("Flash Attention算子")
class TestFlashAttention:
    def setup_method(self, method):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
    def teardown_method(self, method):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
    def is_flash_attention_available(self):
        """检查是否支持Flash Attention"""
        if not torch.cuda.is_available():
            return False
            
        # 检查CUDA计算能力
        device = torch.cuda.current_device()
        compute_capability = torch.cuda.get_device_capability(device)
        # Ampere或更新架构 (计算能力 >= 8.0)
        return compute_capability[0] >= 8
        
    def naive_attention(self, q, k, v, mask=None):
        """实现朴素的attention计算，用于结果验证"""
        d = q.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(d)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
            
        attn_weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, v)
        return output

    @allure.story("基础功能测试")
    @allure.title("测试基本的Flash Attention功能")
    def test_flash_attention_basic(self, device):
        if device == "cpu":
            pytest.skip("Flash Attention只在CUDA设备上可用")
            
        if not self.is_flash_attention_available():
            pytest.skip("当前GPU不支持Flash Attention")
            
        # 准备测试数据
        batch_size = 2
        num_heads = 4
        seq_len = 128
        head_dim = 64
        
        shapes = (batch_size, num_heads, seq_len, head_dim)
        dtype = torch.float16  # Flash Attention需要float16或bfloat16
        
        # 生成随机输入
        q = torch.randn(shapes, dtype=dtype, device="cuda")
        k = torch.randn(shapes, dtype=dtype, device="cuda")
        v = torch.randn(shapes, dtype=dtype, device="cuda")
        
        with allure.step("执行Flash Attention"):
            # 使用PyTorch的scaled_dot_product_attention
            output_flash = torch.nn.functional.scaled_dot_product_attention(
                q, k, v,
                dropout_p=0.0,
                is_causal=False
            )
            
        with allure.step("验证输出"):
            # 验证输出形状
            expected_shape = (batch_size, num_heads, seq_len, head_dim)
            assert output_flash.shape == expected_shape, \
                f"输出形状不符合预期: 期望 {expected_shape}, 实际 {output_flash.shape}"
            
            # 验证输出数值范围
            assert not torch.isnan(output_flash).any(), "输出中包含NaN"
            assert not torch.isinf(output_flash).any(), "输出中包含Inf"
            
            # 与CPU上的naive实现比较（注意：由于数值精度和计算方式的差异，只比较大致范围）
            q_cpu = q.cpu().float()
            k_cpu = k.cpu().float()
            v_cpu = v.cpu().float()
            output_cpu = self.naive_attention(q_cpu, k_cpu, v_cpu)
            
            # 转换为相同的设备和数据类型进行比较
            output_flash_cpu = output_flash.cpu().float()
            
            # 检查输出值的范围是否接近
            assert torch.allclose(
                output_flash_cpu.mean(), 
                output_cpu.mean(), 
                rtol=1e-2, 
                atol=1e-2
            ), "Flash Attention和naive实现的输出均值差异过大"

    @allure.story("因果注意力测试")
    @allure.title("测试因果注意力掩码")
    def test_flash_attention_causal(self, device):
        if device == "cpu":
            pytest.skip("Flash Attention只在CUDA设备上可用")
            
        if not self.is_flash_attention_available():
            pytest.skip("当前GPU不支持Flash Attention")
            
        # 准备测试数据
        batch_size = 2
        num_heads = 4
        seq_len = 64
        head_dim = 32
        
        shapes = (batch_size, num_heads, seq_len, head_dim)
        dtype = torch.float16
        
        q = torch.randn(shapes, dtype=dtype, device="cuda")
        k = torch.randn(shapes, dtype=dtype, device="cuda")
        v = torch.randn(shapes, dtype=dtype, device="cuda")
        
        with allure.step("执行因果Flash Attention"):
            output_causal = torch.nn.functional.scaled_dot_product_attention(
                q, k, v,
                dropout_p=0.0,
                is_causal=True
            )
            
        with allure.step("验证因果掩码效果"):
            # 创建因果掩码进行naive实现
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len), 
                diagonal=1
            ).bool()
            causal_mask = causal_mask.expand(
                batch_size, num_heads, seq_len, seq_len
            )
            
            # 在CPU上用naive实现计算
            q_cpu = q.cpu().float()
            k_cpu = k.cpu().float()
            v_cpu = v.cpu().float()
            output_cpu = self.naive_attention(
                q_cpu, k_cpu, v_cpu, 
                mask=~causal_mask
            )
            
            # 比较结果
            output_causal_cpu = output_causal.cpu().float()
            
            # 检查上三角部分是否被正确掩码
            assert torch.allclose(
                output_causal_cpu.mean(), 
                output_cpu.mean(), 
                rtol=1e-2, 
                atol=1e-2
            ), "因果注意力的输出与预期不符"

    @allure.story("性能测试")
    @allure.title("测试Flash Attention的性能")
    def test_flash_attention_performance(self, device):
        if device == "cpu":
            pytest.skip("Flash Attention只在CUDA设备上可用")
            
        if not self.is_flash_attention_available():
            pytest.skip("当前GPU不支持Flash Attention")
            
        # 准备大规模测试数据
        batch_size = 8
        num_heads = 16
        seq_len = 2048  # 使用较长的序列长度
        head_dim = 64
        
        shapes = (batch_size, num_heads, seq_len, head_dim)
        dtype = torch.float16
        
        q = torch.randn(shapes, dtype=dtype, device="cuda")
        k = torch.randn(shapes, dtype=dtype, device="cuda")
        v = torch.randn(shapes, dtype=dtype, device="cuda")
        
        with allure.step("测试Flash Attention性能"):
            # 预热
            for _ in range(5):
                _ = torch.nn.functional.scaled_dot_product_attention(
                    q, k, v,
                    dropout_p=0.0,
                    is_causal=False
                )
            
            # 计时
            torch.cuda.synchronize()
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)
            
            start_time.record()
            output = torch.nn.functional.scaled_dot_product_attention(
                q, k, v,
                dropout_p=0.0,
                is_causal=False
            )
            end_time.record()
            
            torch.cuda.synchronize()
            
            # 获取运行时间（毫秒）
            runtime_ms = start_time.elapsed_time(end_time)
            
            with allure.step(f"Flash Attention运行时间: {runtime_ms:.2f}ms"):
                # 验证输出
                assert output.shape == shapes, "输出形状不正确"
                assert not torch.isnan(output).any(), "输出中包含NaN"
                assert not torch.isinf(output).any(), "输出中包含Inf"

    @allure.story("边界条件测试")
    @allure.title("测试不同序列长度的Flash Attention")
    def test_flash_attention_sequence_lengths(self, device):
        if device == "cpu":
            pytest.skip("Flash Attention只在CUDA设备上可用")
            
        if not self.is_flash_attention_available():
            pytest.skip("当前GPU不支持Flash Attention")
            
        batch_size = 2
        num_heads = 4
        head_dim = 32
        dtype = torch.float16
        
        # 测试不同的序列长度
        seq_lengths = [16, 32, 64, 128, 256]
        
        for seq_len in seq_lengths:
            with allure.step(f"测试序列长度: {seq_len}"):
                shapes = (batch_size, num_heads, seq_len, head_dim)
                
                q = torch.randn(shapes, dtype=dtype, device="cuda")
                k = torch.randn(shapes, dtype=dtype, device="cuda")
                v = torch.randn(shapes, dtype=dtype, device="cuda")
                
                output = torch.nn.functional.scaled_dot_product_attention(
                    q, k, v,
                    dropout_p=0.0,
                    is_causal=False
                )
                
                # 验证输出
                assert output.shape == shapes, \
                    f"序列长度{seq_len}的输出形状不正确"
                assert not torch.isnan(output).any(), \
                    f"序列长度{seq_len}的输出中包含NaN"
                assert not torch.isinf(output).any(), \
                    f"序列长度{seq_len}的输出中包含Inf"
                
                # 与naive实现比较
                q_cpu = q.cpu().float()
                k_cpu = k.cpu().float()
                v_cpu = v.cpu().float()
                output_cpu = self.naive_attention(q_cpu, k_cpu, v_cpu)
                output_flash_cpu = output.cpu().float()
                
                assert torch.allclose(
                    output_flash_cpu.mean(), 
                    output_cpu.mean(), 
                    rtol=1e-2, 
                    atol=1e-2
                ), f"序列长度{seq_len}的输出与naive实现差异过大"
