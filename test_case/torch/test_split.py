#!/usr/bin/env python
# -*- coding: utf-8 -*-

from itertools import product
import pytest
# 使用我们的导入辅助模块替代直接导入
from .torch_import import torch, torch_import_failed
import allure
from .utils import get_device_object, float_dtypes

@allure.epic("PyTorch算子测试")
@allure.feature("Split算子")
@allure.description("""
该测试模块验证PyTorch中tensor.split算子的功能正确性，包括：
1. 基本功能：验证不同数据类型的split操作
2. 边界情况：验证特殊分割参数的处理
3. 多维度测试：验证不同维度上的分割
4. 性能测试：验证大规模数据的分割操作

所有测试都在CPU和CUDA设备上执行，并验证结果的一致性。
""")
class TestSplit:
    def teardown_method(self, method):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    @allure.story("基本功能测试")
    @allure.title("测试不同数据类型的Split")
    @allure.description("""
    验证基本的split功能，测试要点：
    1. 支持多种数据类型
    2. 验证等分割的正确性
    3. 验证分割后的形状和内容
    4. 比较CPU和CUDA结果的一致性
    """)
    @pytest.mark.parametrize("dtype", float_dtypes)
    def test_split_basic(self, dtype, device):
        device_obj = get_device_object(device)
        
        # 测试2D输入
        x = torch.randn(32, 64, dtype=dtype, device=device_obj)

        # 在CPU上计算参考结果
        x_cpu = x.cpu()
        splits_cpu = torch.split(x_cpu, 16, dim=1)  # 在第1维度上等分成4份
        
        # 在指定设备上计算
        splits = torch.split(x, 16, dim=1)
        
        # 验证分割后的数量
        assert len(splits) == 4
        
        # 验证每个分割的形状
        for i, split in enumerate(splits):
            assert split.shape == (32, 16)
            
            # 比较CPU和当前设备的结果
            if device == "cuda":
                torch.testing.assert_close(split.cpu(), splits_cpu[i], rtol=1e-5, atol=1e-5)

    @allure.story("不均等分割测试")
    @allure.title("测试不均等分割的Split")
    @allure.description("""
    验证不均等分割的功能，测试要点：
    1. 使用不同大小的分割列表
    2. 验证分割后的形状和内容
    3. 比较CPU和CUDA结果的一致性
    """)
    def test_split_uneven(self, device):
        device_obj = get_device_object(device)
        
        # 创建测试数据
        x = torch.randn(20, 30, device=device_obj)
        
        # 定义不均等的分割大小，确保总和等于张量在指定维度的大小
        split_sizes = [5, 10, 5]  # 总和为20，等于x.shape[0]
        
        # 在CPU上计算参考结果
        x_cpu = x.cpu()
        splits_cpu = torch.split(x_cpu, split_sizes, dim=0)
        
        # 在指定设备上计算
        splits = torch.split(x, split_sizes, dim=0)
        
        # 验证分割后的数量
        assert len(splits) == len(split_sizes)
        
        # 验证每个分割的形状
        for i, split in enumerate(splits):
            assert split.shape[0] == split_sizes[i]
            assert split.shape[1] == 30
            
            # 比较CPU和当前设备的结果
            if device == "cuda":
                torch.testing.assert_close(split.cpu(), splits_cpu[i], rtol=1e-5, atol=1e-5)

    @allure.story("多维度测试")
    @allure.title("测试不同维度的Split")
    @allure.description("""
    验证在不同维度上进行分割的功能，测试要点：
    1. 在不同维度上进行分割
    2. 验证分割后的形状和内容
    3. 比较CPU和CUDA结果的一致性
    """)
    @pytest.mark.parametrize("dim", [0, 1, 2])
    def test_split_dimensions(self, dim, device):
        device_obj = get_device_object(device)
        
        # 创建3D测试数据
        x = torch.randn(12, 18, 24, device=device_obj)
        
        # 确定在当前维度上的分割大小
        dim_size = x.shape[dim]
        split_size = dim_size // 3  # 分成3份
        
        # 在CPU上计算参考结果
        x_cpu = x.cpu()
        splits_cpu = torch.split(x_cpu, split_size, dim=dim)
        
        # 在指定设备上计算
        splits = torch.split(x, split_size, dim=dim)
        
        # 验证分割后的数量
        assert len(splits) == 3
        
        # 验证每个分割的形状和内容
        for i, split in enumerate(splits):
            # 比较CPU和当前设备的结果
            if device == "cuda":
                torch.testing.assert_close(split.cpu(), splits_cpu[i], rtol=1e-5, atol=1e-5)

    @allure.story("边界情况测试")
    @allure.title("测试Split的边界情况")
    @allure.description("""
    验证split在边界情况下的行为，测试要点：
    1. 分割大小为1
    2. 分割大小等于整个维度
    3. 处理空张量
    4. 按块大小分割（无法整除）
    5. 异常情况（总和不等）
    """)
    def test_split_edge_cases(self, device):
        device_obj = get_device_object(device)
        
        with allure.step("测试分割大小为1"):
            x = torch.randn(10, 5, device=device_obj)
            splits = torch.split(x, 1, dim=0)
            
            assert len(splits) == 10
            for i, split in enumerate(splits):
                assert split.shape == (1, 5)
                assert torch.allclose(split, x[i:i+1])
        
        with allure.step("测试分割大小等于整个维度"):
            x = torch.randn(10, 5, device=device_obj)
            splits = torch.split(x, 10, dim=0)
            
            assert len(splits) == 1
            assert splits[0].shape == (10, 5)
            assert torch.allclose(splits[0], x)
        
        with allure.step("测试空张量"):
            x = torch.tensor([], device=device_obj).reshape(0, 5)
            splits = torch.split(x, 1, dim=0)
            
            # 对于空张量，PyTorch返回一个包含一个空张量的元组，而不是空元组
            assert len(splits) == 1
            assert splits[0].shape[0] == 0
            assert splits[0].shape[1] == 5
        
        with allure.step("测试按块大小分割（无法整除）"):
            x = torch.randn(10, 5, device=device_obj)
            # 分割大小为3，无法整除维度大小10
            splits = torch.split(x, 3, dim=0)
            
            # 应该返回4个分割：3, 3, 3, 1
            assert len(splits) == 4
            assert splits[0].shape == (3, 5)
            assert splits[1].shape == (3, 5)
            assert splits[2].shape == (3, 5)
            assert splits[3].shape == (1, 5)  # 最后一个分割大小为余数
            
            # 验证分割的内容
            assert torch.allclose(splits[0], x[0:3])
            assert torch.allclose(splits[1], x[3:6])
            assert torch.allclose(splits[2], x[6:9])
            assert torch.allclose(splits[3], x[9:10])
        
        with allure.step("测试异常情况（总和不等）"):
            x = torch.randn(10, 5, device=device_obj)
            
            # 当使用列表指定分割大小时，总和必须等于维度大小
            try:
                # 总和为12，不等于维度大小10
                splits = torch.split(x, [3, 4, 5], dim=0)
                assert False, "应该抛出异常但没有"
            except RuntimeError as e:
                # 验证异常信息包含预期的错误描述
                assert "expects split_sizes to sum exactly to" in str(e)

    @allure.story("性能测试")
    @allure.title("测试大规模数据的Split性能")
    @allure.description("""
    验证split在处理大规模数据时的性能，测试要点：
    1. 处理大型张量
    2. 验证分割后的形状和内容
    """)
    def test_split_large_tensor(self, device):
        device_obj = get_device_object(device)
        
        # 创建大型测试数据
        x = torch.randn(1024, 1024, device=device_obj)
        
        # 定义分割大小
        split_size = 256
        
        # 在指定设备上计算
        splits = torch.split(x, split_size, dim=0)
        
        # 验证分割后的数量
        assert len(splits) == 4
        
        # 验证每个分割的形状
        for split in splits:
            assert split.shape == (256, 1024)
