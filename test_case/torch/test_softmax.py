#!/usr/bin/env python
# -*- coding: utf-8 -*-

from itertools import product
import pytest
import torch
import allure
import numpy as np
from .utils import get_device_object, test_dtypes

@allure.epic("PyTorch算子测试")
@allure.feature("Softmax算子")
@allure.description("""
该测试模块验证PyTorch中tensor.softmax算子的功能正确性，包括：
1. 基本功能：验证不同数据类型的softmax计算
2. 边界情况：验证数值稳定性
3. 性能测试：验证大规模数据的计算
4. 多维度测试：验证不同维度的计算

所有测试都在CPU和CUDA设备上执行，并验证结果的一致性。
""")
@pytest.mark.order(7)
class TestSoftmax:
    @allure.story("基本功能测试")
    @allure.title("测试不同数据类型的Softmax")
    @allure.description("""
    验证基本的softmax计算功能，测试要点：
    1. 支持多种数据类型
    2. 验证输出和为1
    3. 验证值域在[0,1]之间
    4. 比较CPU和CUDA结果的一致性
    """)
    @pytest.mark.parametrize("dtype", test_dtypes)
    def test_softmax_basic(self, dtype, device):
        if dtype in [torch.int32, torch.int64] and device == "cuda":
            pytest.skip(f"数据类型 {dtype} 在CUDA上不支持")

        device_obj = get_device_object(device)
        
        # 测试2D输入
        if dtype in [torch.int32, torch.int64]:
            x = torch.randint(-10, 10, (32, 64), dtype=dtype, device=device_obj)
            x = x.float()
        else:
            x = torch.randn(32, 64, dtype=dtype, device=device_obj)

        # 在CPU上计算参考结果
        x_cpu = x.cpu()
        output_cpu = x_cpu.softmax(dim=-1)

        # 在指定设备上计算
        output = x.softmax(dim=-1)
        
        # 验证输出和为1
        sums = torch.sum(output, dim=-1)
        torch.testing.assert_close(sums, torch.ones_like(sums), rtol=1e-5, atol=1e-5)
        
        # 验证值域在[0,1]之间
        assert torch.all(output >= 0) and torch.all(output <= 1)

        # 比较CPU和当前设备的结果
        if device == "cuda":
            torch.testing.assert_close(output.cpu(), output_cpu, rtol=1e-4, atol=1e-4)

    @allure.story("数值稳定性测试")
    @allure.title("测试Softmax数值稳定性")
    @allure.description("""
    验证softmax在不同数值范围的稳定性，测试要点：
    1. 处理小数值输入（1e-4）
    2. 处理正常范围输入（1.0）
    3. 处理大数值输入（1e4）
    4. 验证没有NaN和Inf
    5. 比较CPU和CUDA的计算结果
    """)
    def test_softmax_numerical_stability(self, device):
        device_obj = get_device_object(device)
        dtype = torch.float32
        
        test_cases = [
            ("小数值", 1e-4),
            ("正常数值", 1.0),
            ("大数值", 1e4)
        ]
        
        for case_name, scale in test_cases:
            with allure.step(f"测试{case_name}"):
                # 在CPU上计算参考结果
                x_cpu = torch.randn(32, 64, dtype=dtype) * scale
                output_cpu = x_cpu.softmax(dim=-1)

                # 在指定设备上计算
                x = x_cpu.to(device_obj)
                output = x.softmax(dim=-1)
                
                # 验证输出中没有NaN或Inf
                assert not torch.isnan(output).any(), f"{case_name}的输出中包含NaN"
                assert not torch.isinf(output).any(), f"{case_name}的输出中包含Inf"
                
                # 验证输出和为1
                sums = torch.sum(output, dim=-1)
                torch.testing.assert_close(
                    sums, 
                    torch.ones_like(sums),
                    rtol=1e-5,
                    atol=1e-5
                )

                # 比较CPU和当前设备的结果
                if device == "cuda":
                    torch.testing.assert_close(output.cpu(), output_cpu, rtol=1e-4, atol=1e-4)

    @allure.story("性能测试")
    @allure.title("测试大规模Softmax")
    @allure.description("""
    验证大规模数据的softmax计算，测试要点：
    1. 处理大规模数据（batch=32, seq_len=512, hidden_size=1024）
    2. 验证计算的准确性
    3. 验证输出形状的一致性
    4. 比较CPU和CUDA的计算结果
    """)
    def test_softmax_performance(self, device):
        device_obj = get_device_object(device)
        dtype = torch.float32
        
        # 使用较大的输入尺寸
        batch_size = 32
        seq_len = 512
        hidden_size = 1024
        
        # 在CPU上计算参考结果
        x_cpu = torch.randn(batch_size, seq_len, hidden_size, dtype=dtype)
        output_cpu = x_cpu.softmax(dim=-1)

        # 在指定设备上计算
        x = x_cpu.to(device_obj)
        output = x.softmax(dim=-1)

        # 验证输出和为1
        sums = torch.sum(output, dim=-1)
        torch.testing.assert_close(
            sums,
            torch.ones_like(sums),
            rtol=1e-5,
            atol=1e-5
        )

        # 比较CPU和当前设备的结果
        if device == "cuda":
            torch.testing.assert_close(output.cpu(), output_cpu, rtol=1e-4, atol=1e-4)

    @allure.story("维度测试")
    @allure.title("测试不同维度的Softmax")
    @allure.description("""
    验证不同维度输入的softmax计算，测试要点：
    1. 处理1D到4D的输入数据
    2. 验证输出形状的一致性
    3. 验证计算的准确性
    4. 比较CPU和CUDA的计算结果
    """)
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
                # 在CPU上计算参考结果
                x_cpu = torch.randn(*shape, dtype=dtype)
                output_cpu = x_cpu.softmax(dim=-1)

                # 在指定设备上计算
                x = x_cpu.to(device_obj)
                output = x.softmax(dim=-1)
                
                # 验证输出形状
                assert output.shape == shape
                
                # 验证输出和为1
                sums = torch.sum(output, dim=-1)
                torch.testing.assert_close(
                    sums,
                    torch.ones_like(sums),
                    rtol=1e-5,
                    atol=1e-5
                )

                # 比较CPU和当前设备的结果
                if device == "cuda":
                    torch.testing.assert_close(output.cpu(), output_cpu, rtol=1e-4, atol=1e-4)


    def test_softmax_half_to_float(self, device):
        device_obj = get_device_object(device)
        _softmax = torch.softmax
        shapes = [
            [(2, 3, 5), (2, 3, 5)],
            [(7, 8, 9, 10), (9, 4)],
            [(10,), (3, 5, 7)],
            [(2, 0, 3,  5), (9, 4)],
            [(10,15), (10, 15)],
        ]
        dtypes = [(torch.half, torch.float), (torch.float, torch.float)]
        for [shape, out_shape], [dtype1, dtype2] in product(*[shapes, dtypes]):
            x = torch.randn(shape, dtype=dtype1)

            for dim in range(len(shape)):
                res_cpu = _softmax(x, dim, dtype=dtype2)
                res_cuda = _softmax(x.cuda(), dim, dtype=dtype2)
                torch.testing.assert_close(res_cpu, res_cuda.cpu().float(), rtol=2e-4, atol=1e-4)

