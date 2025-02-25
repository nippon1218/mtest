#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import torch
import allure
import numpy as np
import math
from .utils import get_device_object, test_dtypes

@allure.epic("PyTorch算子测试")
@allure.feature("FFN Megatron算子")
class TestFFNMegatron:
    def setup_method(self, method):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
    def teardown_method(self, method):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
    def naive_ffn(self, x, fc1_weight, fc1_bias, fc2_weight, fc2_bias):
        """实现朴素的FFN计算，用于结果验证"""
        hidden = torch.nn.functional.linear(x, fc1_weight, fc1_bias)
        hidden = torch.nn.functional.gelu(hidden)
        output = torch.nn.functional.linear(hidden, fc2_weight, fc2_bias)
        return output

    def megatron_ffn(self, x, fc1_weight, fc1_bias, fc2_weight, fc2_bias):
        """Megatron风格的FFN实现"""
        # 分块计算以优化内存使用
        chunk_size = 1024
        hidden_size = fc1_weight.size(0)
        
        outputs = []
        for chunk_start in range(0, x.size(0), chunk_size):
            chunk_end = min(chunk_start + chunk_size, x.size(0))
            chunk = x[chunk_start:chunk_end]
            
            # 第一个线性层
            hidden = torch.nn.functional.linear(chunk, fc1_weight, fc1_bias)
            
            # 使用fast_gelu（如果可用）
            if hasattr(torch.nn.functional, 'gelu') and \
               torch.nn.functional.gelu.__name__ == 'gelu':
                hidden = torch.nn.functional.gelu(hidden)
            else:
                # 使用更稳定的gelu实现
                hidden = torch.clamp(hidden, -10, 10)  # 限制输入范围
                hidden = 0.5 * hidden * (
                    1.0 + torch.tanh(
                        torch.tensor(0.7978845608028654, device=hidden.device) * 
                        (hidden + 0.044715 * hidden.pow(3))
                    )
                )
            
            # 第二个线性层
            output = torch.nn.functional.linear(hidden, fc2_weight, fc2_bias)
            outputs.append(output)
            
        return torch.cat(outputs, dim=0)

    @allure.story("基础功能测试")
    @allure.title("测试基本的FFN功能")
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
    def test_ffn_basic(self, device, dtype):
        # 准备测试数据
        batch_size = 32
        seq_len = 64
        hidden_size = 256
        ffn_hidden_size = 1024
        
        # 初始化权重
        fc1_weight = torch.randn(ffn_hidden_size, hidden_size, dtype=dtype)
        fc1_bias = torch.randn(ffn_hidden_size, dtype=dtype)
        fc2_weight = torch.randn(hidden_size, ffn_hidden_size, dtype=dtype)
        fc2_bias = torch.randn(hidden_size, dtype=dtype)
        
        # 输入数据
        x = torch.randn(batch_size * seq_len, hidden_size, dtype=dtype)
        
        if device == "cuda" and torch.cuda.is_available():
            # 移动数据到GPU
            x = x.cuda()
            fc1_weight = fc1_weight.cuda()
            fc1_bias = fc1_bias.cuda()
            fc2_weight = fc2_weight.cuda()
            fc2_bias = fc2_bias.cuda()
            
            with allure.step(f"执行Megatron FFN - 设备: cuda, 数据类型: {dtype}"):
                output_mega = self.megatron_ffn(
                    x, fc1_weight, fc1_bias, fc2_weight, fc2_bias
                )
                
            with allure.step("验证输出"):
                # 验证形状
                expected_shape = (batch_size * seq_len, hidden_size)
                assert output_mega.shape == expected_shape, \
                    f"输出形状不符合预期: 期望 {expected_shape}, 实际 {output_mega.shape}"
                
                # 验证数值范围
                assert not torch.isnan(output_mega).any(), "输出中包含NaN"
                assert not torch.isinf(output_mega).any(), "输出中包含Inf"
                
                # 与naive实现比较
                output_naive = self.naive_ffn(
                    x, fc1_weight, fc1_bias, fc2_weight, fc2_bias
                )
                
                # 由于数值精度的差异，使用较大的容差
                rtol = 1e-2 if dtype == torch.float16 else 1e-3
                atol = 1e-2 if dtype == torch.float16 else 1e-3
                
                assert torch.allclose(
                    output_mega, output_naive, rtol=rtol, atol=atol
                ), "Megatron实现与naive实现的结果不一致"
        
        else:
            # CPU实现
            with allure.step(f"执行FFN - 设备: cpu, 数据类型: {dtype}"):
                output = self.naive_ffn(
                    x, fc1_weight, fc1_bias, fc2_weight, fc2_bias
                )
                
            with allure.step("验证输出"):
                expected_shape = (batch_size * seq_len, hidden_size)
                assert output.shape == expected_shape, \
                    f"输出形状不符合预期: 期望 {expected_shape}, 实际 {output.shape}"

    @allure.story("性能测试")
    @allure.title("测试FFN的性能")
    def test_ffn_performance(self, device):
        if device != "cuda" or not torch.cuda.is_available():
            pytest.skip("性能测试需要CUDA设备")
            
        # 准备大规模测试数据
        dtype = torch.float16  # 使用float16以获得更好的性能
        batch_size = 32
        seq_len = 512
        hidden_size = 1024
        ffn_hidden_size = 4096
        
        # 初始化权重
        fc1_weight = torch.randn(
            ffn_hidden_size, hidden_size, 
            dtype=dtype, device="cuda"
        )
        fc1_bias = torch.randn(
            ffn_hidden_size, 
            dtype=dtype, device="cuda"
        )
        fc2_weight = torch.randn(
            hidden_size, ffn_hidden_size, 
            dtype=dtype, device="cuda"
        )
        fc2_bias = torch.randn(
            hidden_size, 
            dtype=dtype, device="cuda"
        )
        
        x = torch.randn(
            batch_size * seq_len, hidden_size, 
            dtype=dtype, device="cuda"
        )
        
        with allure.step("测试Megatron FFN性能"):
            # 预热
            for _ in range(5):
                _ = self.megatron_ffn(
                    x, fc1_weight, fc1_bias, fc2_weight, fc2_bias
                )
            
            # 计时
            torch.cuda.synchronize()
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)
            
            start_time.record()
            output = self.megatron_ffn(
                x, fc1_weight, fc1_bias, fc2_weight, fc2_bias
            )
            end_time.record()
            
            torch.cuda.synchronize()
            
            runtime_ms = start_time.elapsed_time(end_time)
            
            with allure.step(f"Megatron FFN运行时间: {runtime_ms:.2f}ms"):
                # 验证输出
                assert output.shape == (batch_size * seq_len, hidden_size), \
                    "输出形状不正确"
                assert not torch.isnan(output).any(), "输出中包含NaN"
                assert not torch.isinf(output).any(), "输出中包含Inf"

    @allure.story("边界条件测试")
    @allure.title("测试不同输入大小的FFN")
    def test_ffn_input_sizes(self, device):
        if device != "cuda" or not torch.cuda.is_available():
            pytest.skip("边界条件测试需要CUDA设备")
            
        dtype = torch.float16
        hidden_size = 256
        ffn_hidden_size = 1024
        
        # 初始化权重
        fc1_weight = torch.randn(
            ffn_hidden_size, hidden_size, 
            dtype=dtype, device="cuda"
        )
        fc1_bias = torch.randn(
            ffn_hidden_size, 
            dtype=dtype, device="cuda"
        )
        fc2_weight = torch.randn(
            hidden_size, ffn_hidden_size, 
            dtype=dtype, device="cuda"
        )
        fc2_bias = torch.randn(
            hidden_size, 
            dtype=dtype, device="cuda"
        )
        
        # 测试不同的输入大小
        input_sizes = [
            (1, 1),      # 最小输入
            (16, 32),    # 小输入
            (64, 64),    # 中等输入
            (128, 128),  # 大输入
        ]
        
        for batch_size, seq_len in input_sizes:
            with allure.step(f"测试输入大小: batch_size={batch_size}, seq_len={seq_len}"):
                x = torch.randn(
                    batch_size * seq_len, hidden_size, 
                    dtype=dtype, device="cuda"
                )
                
                output = self.megatron_ffn(
                    x, fc1_weight, fc1_bias, fc2_weight, fc2_bias
                )
                
                # 验证输出
                expected_shape = (batch_size * seq_len, hidden_size)
                assert output.shape == expected_shape, \
                    f"输入大小({batch_size},{seq_len})的输出形状不正确"
                assert not torch.isnan(output).any(), \
                    f"输入大小({batch_size},{seq_len})的输出中包含NaN"
                assert not torch.isinf(output).any(), \
                    f"输入大小({batch_size},{seq_len})的输出中包含Inf"
                
                # 与naive实现比较
                output_naive = self.naive_ffn(
                    x, fc1_weight, fc1_bias, fc2_weight, fc2_bias
                )
                
                assert torch.allclose(
                    output, output_naive, rtol=1e-2, atol=1e-2
                ), f"输入大小({batch_size},{seq_len})的结果与naive实现不一致"

    @allure.story("数值稳定性测试")
    @allure.title("测试FFN的数值稳定性")
    def test_ffn_numerical_stability(self, device):
        if device != "cuda" or not torch.cuda.is_available():
            pytest.skip("数值稳定性测试需要CUDA设备")
            
        dtype = torch.float16
        batch_size = 32
        seq_len = 64
        hidden_size = 256
        ffn_hidden_size = 1024
        
        # 测试不同范围的输入值
        test_cases = [
            ("小数值", 1e-2),
            ("正常数值", 1.0),
            ("大数值", 1e1),  # 对于float16，使用较小的scale
        ]
        
        for case_name, scale in test_cases:
            with allure.step(f"测试{case_name}"):
                # 初始化权重，使用Xavier初始化的缩放
                fc1_weight = torch.randn(
                    ffn_hidden_size, hidden_size, 
                    dtype=dtype, device="cuda"
                ) * (scale / math.sqrt(hidden_size))
                fc1_bias = torch.zeros(
                    ffn_hidden_size, 
                    dtype=dtype, device="cuda"
                )
                fc2_weight = torch.randn(
                    hidden_size, ffn_hidden_size, 
                    dtype=dtype, device="cuda"
                ) * (scale / math.sqrt(ffn_hidden_size))
                fc2_bias = torch.zeros(
                    hidden_size, 
                    dtype=dtype, device="cuda"
                )
                
                x = torch.randn(
                    batch_size * seq_len, hidden_size, 
                    dtype=dtype, device="cuda"
                ) * scale
                
                output = self.megatron_ffn(
                    x, fc1_weight, fc1_bias, fc2_weight, fc2_bias
                )
                
                # 验证输出
                assert not torch.isnan(output).any(), \
                    f"{case_name}的输出中包含NaN"
                assert not torch.isinf(output).any(), \
                    f"{case_name}的输出中包含Inf"
                
                # 检查输出范围
                max_abs_val = torch.max(torch.abs(output))
                assert max_abs_val < 1e6, \
                    f"{case_name}的输出值过大: {max_abs_val}"
