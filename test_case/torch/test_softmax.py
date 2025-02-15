#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import torch
import allure
import numpy as np
import math
from .utils import get_device_object, test_dtypes

@allure.epic("PyTorch算子测试")
@allure.feature("Softmax算子")
class TestSoftmax:
    def softmax(self, x, dim=-1):
        """
        实现Softmax函数
        """
        # 为了数值稳定性，减去最大值
        x_max = torch.max(x, dim=dim, keepdim=True)[0]
        exp_x = torch.exp(x - x_max)
        return exp_x / torch.sum(exp_x, dim=dim, keepdim=True)

    @allure.story("基本功能测试")
    @allure.title("测试Softmax基本功能 - {dtype}")
    @pytest.mark.parametrize("dtype", test_dtypes)
    def test_softmax_basic(self, dtype, device):
        if dtype in [torch.int32, torch.int64] and device == "cuda":
            pytest.skip(f"数据类型 {dtype} 在CUDA上不支持")

        device_obj = get_device_object(device)
        
        # 测试2D输入
        if dtype in [torch.int32, torch.int64]:
            # 对于整数类型，使用randint而不是randn
            x = torch.randint(-10, 10, (32, 64), dtype=dtype, device=device_obj)
        else:
            x = torch.randn(32, 64, dtype=dtype, device=device_obj)
            
        # 对于整数类型，需要先转换为浮点型
        if dtype in [torch.int32, torch.int64]:
            x = x.float()
            
        output = self.softmax(x)
        
        # 验证输出和为1
        sums = torch.sum(output, dim=-1)
        torch.testing.assert_close(sums, torch.ones_like(sums))
        
        # 验证值域在[0,1]之间
        assert torch.all(output >= 0) and torch.all(output <= 1)

    @allure.story("数值稳定性测试")
    @allure.title("测试Softmax数值稳定性")
    def test_softmax_numerical_stability(self, device):
        if device != "cuda" or not torch.cuda.is_available():
            pytest.skip("数值稳定性测试需要CUDA设备")
            
        dtype = torch.float16
        device_obj = get_device_object(device)
        
        test_cases = [
            ("小数值", 1e-4),
            ("正常数值", 1.0),
            ("大数值", 1e4)
        ]
        
        for case_name, scale in test_cases:
            with allure.step(f"测试{case_name}"):
                x = torch.randn(32, 64, dtype=dtype, device=device_obj) * scale
                output = self.softmax(x)
                
                # 验证输出中没有NaN或Inf
                assert not torch.isnan(output).any(), f"{case_name}的输出中包含NaN"
                assert not torch.isinf(output).any(), f"{case_name}的输出中包含Inf"
                
                # 验证输出和为1
                sums = torch.sum(output, dim=-1)
                torch.testing.assert_close(
                    sums, 
                    torch.ones_like(sums),
                    rtol=1e-3,  # 对于float16，使用较大的容差
                    atol=1e-3
                )

    @allure.story("性能测试")
    @allure.title("测试Softmax性能")
    def test_softmax_performance(self, device):
        if device != "cuda" or not torch.cuda.is_available():
            pytest.skip("性能测试需要CUDA设备")
            
        device_obj = get_device_object(device)
        dtype = torch.float16
        
        # 使用较大的输入尺寸
        batch_size = 32
        seq_len = 512
        hidden_size = 1024
        
        x = torch.randn(
            batch_size, seq_len, hidden_size,
            dtype=dtype, device=device_obj
        )
        
        # 预热
        for _ in range(10):
            _ = self.softmax(x)
            
        torch.cuda.synchronize()
        
        # 测试100次迭代的平均时间
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        for _ in range(100):
            _ = self.softmax(x)
        end_event.record()
        
        torch.cuda.synchronize()
        avg_time = start_event.elapsed_time(end_event) / 100
        
        # 记录性能指标
        allure.attach(
            f"平均执行时间: {avg_time:.3f} ms",
            name="性能指标",
            attachment_type=allure.attachment_type.TEXT
        )

    @allure.story("维度测试")
    @allure.title("测试Softmax不同维度")
    def test_softmax_dimensions(self, device):
        device_obj = get_device_object(device)
        dtype = torch.float32
        
        test_shapes = [
            (32,),           # 1D
            (16, 32),        # 2D
            (8, 16, 32),     # 3D
            (4, 8, 16, 32)   # 4D
        ]
        
        for shape in test_shapes:
            with allure.step(f"测试shape={shape}"):
                x = torch.randn(*shape, dtype=dtype, device=device_obj)
                
                # 对最后一个维度做softmax
                output = self.softmax(x)
                
                # 验证输出形状
                assert output.shape == shape
                
                # 验证输出和为1
                sums = torch.sum(output, dim=-1)
                torch.testing.assert_close(sums, torch.ones_like(sums))
